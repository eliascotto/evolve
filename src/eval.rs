use std::sync::Arc;

use crate::collections::{List, Vector};
use crate::core::native_fns::NativeRegistry;
use crate::core::{Metadata, Namespace, RecurContext, SpecialFormRegistry, Var};
use crate::env::Env;
use crate::error::{Error, SpannedResult, SyntaxError, error_at};
use crate::interner::{self, SymId};
use crate::reader::Span;
use crate::value::Value;
use crate::synthetic_span;

/// Result of executing a single trampoline step.
enum Step {
    /// The evaluator produced a value and an updated environment.
    Value(Value, Env),
    /// The evaluator wants to continue by tail-calling into the given form. If
    /// `restore` is `Some(env)`, the environment should be restored to `env`
    /// once the tail expression completes (used for constructs such as loops
    /// and function bodies that execute in a child environment).
    Tail { form: Value, env: Env, restore: Option<Env> },
    /// Internal trampoline signal indicating that a recur instruction was
    /// emitted.  Carries the recursion context, the evaluated arguments, the
    /// environment that produced them, and the span of the recur form.
    Recur(RecurContext, Arc<Vector<Value>>, Env, Span),
}

type EvalResult<T> = SpannedResult<T>;

/// A trampoline interpreter that shares infrastructure with the legacy
/// recursive evaluator but keeps evaluation inside an explicit loop.
#[derive(Debug, Clone)]
pub struct Evaluator {
    pub special_forms: Arc<SpecialFormRegistry>,
    pub native_fns: Arc<NativeRegistry>,
}

impl Evaluator {
    /// Create a new trampoline engine backed by the shared core symbols and
    /// native function registry.
    pub fn new() -> Self {
        let core_syms = Arc::new(SpecialFormRegistry::new());
        let native_fns = Arc::new(NativeRegistry::new());

        Self { special_forms: core_syms, native_fns }
    }

    /// Evaluate a form using the trampoline loop.
    ///
    /// The method never relies on the Rust call stack for tail position.  
    /// Instead it accumulates the next form to evaluate and iterates until a value is produced.
    pub fn eval(&self, form: &Value, env: &mut Env) -> EvalResult<Value> {
        let mut current_form = form.clone();
        let mut current_env = env.clone();
        // We need to keep track of the environments that we need to restore
        let mut restore_stack: Vec<Env> = Vec::new();

        loop {
            match self.eval_step(current_form.clone(), current_env.clone())? {
                Step::Value(value, mut next_env) => {
                    while let Some(restore_env) = restore_stack.pop() {
                        next_env = restore_env;
                    }
                    *env = next_env.clone();
                    return Ok(value);
                }
                Step::Tail { form: next_form, env: next_env, restore } => {
                    if let Some(restore_env) = restore {
                        restore_stack.push(restore_env);
                    }
                    current_form = next_form;
                    current_env = next_env;
                }
                Step::Recur(context, args, mut next_env, recur_span) => {
                    match context {
                        RecurContext::Loop { bindings, body, span } => {
                            if args.len() != bindings.len() {
                                return Err(error_at(
                                    recur_span.clone(),
                                    Error::SyntaxError(
                                        SyntaxError::WrongArgumentCount {
                                            error_str: format!(
                                                "Wrong number of arguments to recur. Expected {}, got {}",
                                                bindings.len(),
                                                args.len()
                                            ),
                                        },
                                    ),
                                ));
                            }

                            for (binding, new_value) in
                                bindings.iter().zip(args.iter())
                            {
                                if let Value::Symbol { value, .. } = binding {
                                    next_env =
                                        next_env.set(value.id(), new_value.clone());
                                }
                            }

                            current_form =
                                make_do_value(self, body.clone(), span.clone());
                            current_env = next_env;
                        }
                        RecurContext::Function { params, body, span } => {
                            if args.len() != params.len() {
                                return Err(error_at(
                                    recur_span,
                                    Error::SyntaxError(
                                        SyntaxError::WrongArgumentCount {
                                            error_str: format!(
                                                "Wrong number of arguments to recur. Expected {}, got {}",
                                                params.len(),
                                                args.len()
                                            ),
                                        },
                                    ),
                                ));
                            }

                            for (param, new_value) in params.iter().zip(args.iter())
                            {
                                if let Value::Symbol { value, .. } = param {
                                    next_env =
                                        next_env.set(value.id(), new_value.clone());
                                }
                            }

                            current_form =
                                make_do_value(self, body.clone(), span.clone());
                            current_env = next_env;
                        }
                    }
                }
            }
        }
    }

    fn eval_step(&self, form: Value, env: Env) -> EvalResult<Step> {
        match form {
            Value::Symbol { value: sym, span, .. } => {
                self.eval_symbol(sym.id(), span.clone(), env)
            }
            Value::List { value: list, span, .. } => {
                self.eval_list(list, span.clone(), env)
            }
            Value::Vector { value: vector, span, .. } => {
                self.eval_vector(vector, span.clone(), env)
            }
            primitive => Ok(Step::Value(primitive, env)),
        }
    }

    fn eval_symbol(&self, sym: SymId, span: Span, env: Env) -> EvalResult<Step> {
        // Look for a value pointed by sym in the environment
        if let Some(value) = env.get(sym).cloned() {
            let resolved = self.resolve_var_value(value, span.clone())?;
            return Ok(Step::Value(resolved, env));
        }

        // Look for a var binding pointer by sym in the namespace
        if let Some(var) = env.ns.get(sym).cloned() {
            let resolved = self.read_var(&var, span.clone())?;
            return Ok(Step::Value(resolved, env));
        }

        if self.is_special_form(sym) {
            return Ok(Step::Value(
                Value::SpecialForm { span: span.clone(), name: sym },
                env,
            ));
        }

        if let Some(f) = self.native_fns.resolve(sym) {
            return Ok(Step::Value(
                Value::NativeFunction { span: span.clone(), name: sym, f },
                env,
            ));
        }

        Err(error_at(
            span,
            Error::RuntimeError(format!(
                "Undefined symbol: {} ({:?})",
                interner::sym_to_str(sym),
                sym
            )),
        ))
    }

    /// Returns a value, if the value is a var, it will be resolved to the value it points to.
    fn resolve_var_value(&self, value: Value, span: Span) -> EvalResult<Value> {
        if let Value::Var { value: var, .. } = value {
            self.read_var(&var, span)
        } else {
            Ok(value)
        }
    }

    /// Reads the value from a var.
    fn read_var(&self, var: &Arc<Var>, span: Span) -> EvalResult<Value> {
        let storage = var.value.as_ref().ok_or_else(|| {
            error_at(span.clone(), Error::RuntimeError("Var is unbound".to_string()))
        })?;
        let guard = storage.read().map_err(|_| {
            error_at(
                span.clone(),
                Error::RuntimeError("Failed to read var binding".to_string()),
            )
        })?;
        Ok((*guard).clone())
    }

    fn eval_vector(
        &self,
        vector: Arc<Vector<Value>>,
        span: Span,
        mut env: Env,
    ) -> EvalResult<Step> {
        let mut items = Vec::with_capacity(vector.len());
        for form in vector.iter() {
            let value = self.eval(form, &mut env)?;
            items.push(value);
        }
        Ok(Step::Value(
            Value::Vector {
                span,
                value: Arc::new(Vector::from_iter(items)),
                meta: None,
            },
            env,
        ))
    }

    fn eval_list(
        &self,
        list: Arc<List<Value>>,
        span: Span,
        mut env: Env,
    ) -> EvalResult<Step> {
        if list.is_empty() {
            // Empty list evalue to itself
            return Ok(Step::Value(
                Value::List { span, value: list, meta: None },
                env,
            ));
        }

        let head = list.head().unwrap().clone();
        let args = list.tail().unwrap_or_else(List::new);

        // Resolve the macro variable or symbol
        let maybe_macro_var = match &head {
            // A list constructed programmatically can contain a var pointer to a macro
            Value::Var { value: var, .. } => Some(var.clone()),
            // Normal macro call receives a symbol
            Value::Symbol { value: sym, .. } => env.ns.get(sym.id()).cloned(),
            _ => None,
        };

        // Handle macro expansion before function evaluation
        if let Some(var) = maybe_macro_var {
            if var.is_macro() {
                return self.expand_macro(var, args, env, span.clone());
            }
        }

        let func = self.eval(&head, &mut env)?;
        match func {
            // Special form
            Value::SpecialForm { name: sym, .. } => {
                self.eval_special_form(sym, args, env, span.clone())
            }
            // Native function
            Value::NativeFunction { f, .. } => {
                let evaluated_args = self.collect_args(&args, &mut env)?;
                let result = f(&evaluated_args, &mut env)
                    .map_err(|error| error_at(span.clone(), error))?;
                Ok(Step::Value(result, env))
            }
            // Ordinary function call
            Value::Function { params, body, env: fn_env, span: fn_span, .. } => {
                let evaluated_args = self.collect_args(&args, &mut env)?;
                if params.len() != evaluated_args.len() {
                    return Err(error_at(
                        span.clone(),
                        Error::SyntaxError(SyntaxError::WrongArgumentCount {
                            error_str: format!(
                                "Wrong number of arguments. Expected {}, got {}",
                                params.len(),
                                evaluated_args.len()
                            ),
                        }),
                    ));
                }

                let mut binding_pairs = Vec::new();
                for (param, arg_value) in params.iter().zip(evaluated_args.iter()) {
                    binding_pairs.push(param.clone());
                    binding_pairs.push(arg_value.clone());
                }
                let bindings = Arc::new(Vector::from_iter(binding_pairs));
                let call_env = fn_env
                    .create_child_with_bindings(bindings)
                    .with_recur_context(RecurContext::Function {
                        params: params.clone(),
                        body: body.clone(),
                        span: fn_span.clone(),
                    });
                let body_form = make_do_value(self, body.clone(), fn_span.clone());
                Ok(Step::Tail { form: body_form, env: call_env, restore: Some(env) })
            }
            other => Err(error_at(
                span.clone(),
                Error::TypeError {
                    expected: "callable".to_string(),
                    actual: other.as_str().to_string(),
                },
            )),
        }
    }

    fn expand_macro(
        &self,
        macro_var: Arc<Var>,
        args: List<Value>,
        env: Env,
        call_span: Span,
    ) -> EvalResult<Step> {
        // Resolve the macro function stored in the var
        let macro_name = interner::sym_to_str(macro_var.symbol);
        let storage = macro_var.value.as_ref().ok_or_else(|| {
            error_at(
                call_span.clone(),
                Error::RuntimeError(format!("Macro '{}' is unbound", macro_name)),
            )
        })?;

        let macro_value = storage.read().map_err(|_| {
            error_at(
                call_span.clone(),
                Error::RuntimeError(format!(
                    "Failed to read macro binding for '{}'",
                    macro_name
                )),
            )
        })?;
        // Clone the value then drop the original so the lock is released before the macro expansion continues.
        let macro_fn = macro_value.clone();
        drop(macro_value);

        let (params, body, macro_env, macro_span) = match macro_fn {
            Value::Function { params, body, env: fn_env, span, .. } => {
                (params, body, fn_env, span)
            }
            other => {
                return Err(error_at(
                    call_span.clone(),
                    Error::TypeError {
                        expected: "function".to_string(),
                        actual: other.as_str().to_string(),
                    },
                ));
            }
        };

        // Bind macro parameters to the raw argument forms
        let provided_args: Vec<Value> = args.iter().cloned().collect();
        if params.len() != provided_args.len() {
            return Err(error_at(
                call_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: format!(
                        "Wrong number of arguments to macro. Expected {}, got {}",
                        params.len(),
                        provided_args.len()
                    ),
                }),
            ));
        }

        let mut binding_pairs = Vec::with_capacity(provided_args.len() * 2);
        for (param, arg_form) in params.iter().zip(provided_args.iter()) {
            binding_pairs.push(param.clone());
            binding_pairs.push(arg_form.clone());
        }
        let bindings = Arc::new(Vector::from_iter(binding_pairs));
        let mut macro_scope_env = macro_env.create_child_with_bindings(bindings);

        // Evaluate the macro body to produce the expansion form
        let expansion_form = make_do_value(self, body.clone(), macro_span.clone());
        let expanded = self.eval(&expansion_form, &mut macro_scope_env)?;

        // Evaluate the expanded form in place of the original call
        return Ok(Step::Tail { form: expanded, env, restore: None });
    }

    fn collect_args(
        &self,
        args: &List<Value>,
        env: &mut Env,
    ) -> EvalResult<Vec<Value>> {
        args.iter().map(|form| self.eval(form, env)).collect::<Result<Vec<_>, _>>()
    }

    fn eval_special_form(
        &self,
        sym: SymId,
        args: List<Value>,
        env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if sym == self.special_forms.s_def {
            return self.eval_def(args, env, form_span);
        }
        if sym == self.special_forms.s_defmacro {
            return self.eval_defmacro(args, env, form_span);
        }
        if sym == self.special_forms.s_if {
            return self.eval_if(args, env, form_span);
        }
        if sym == self.special_forms.s_let {
            return self.eval_let(args, env, form_span);
        }
        if sym == self.special_forms.s_do {
            return self.eval_do(args, env, form_span);
        }
        if sym == self.special_forms.s_quote {
            return self.eval_quote(args, env, form_span);
        }
        if sym == self.special_forms.s_loop {
            return self.eval_loop(args, env, form_span);
        }
        if sym == self.special_forms.s_recur {
            return self.eval_recur(args, env, form_span);
        }
        if sym == self.special_forms.s_fn {
            return self.eval_fn(args, env, form_span);
        }
        if sym == self.special_forms.s_ns {
            return self.eval_ns(args, env, form_span);
        }

        Err(error_at(
            form_span,
            Error::RuntimeError(format!("Unknown special form: {:?}", sym)),
        ))
    }

    fn eval_def(
        &self,
        args: List<Value>,
        env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.is_empty() {
            return Err(error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "def requires at least 1 argument".to_string(),
                }),
            ));
        }

        // (def symbol doc-string? init?)
        let mut iter = args.iter();
        let symbol_value = iter.next().ok_or_else(|| {
            error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "def requires a symbol".to_string(),
                }),
            )
        })?;

        let (sym, mut sym_meta, sym_span) = match symbol_value {
            Value::Symbol { value, span } => {
                (value.id(), value.metadata(), span.clone())
            }
            other => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidSymbol {
                        value: "First argument to def must be a symbol".to_string(),
                    }),
                ));
            }
        };

        let mut docstring: Option<String> = None;
        let expr_value = match iter.len() {
            0 => Value::Nil { span: synthetic_span!() },
            1 => iter.next().unwrap().clone(),
            2 => {
                let potential_doc = iter.next().unwrap();
                let maybe_expr = iter.next().unwrap();
                if let Value::String { value: doc, .. } = potential_doc {
                    docstring = Some(doc.to_string());
                    maybe_expr.clone()
                } else {
                    return Err(error_at(
                        potential_doc.span(),
                        Error::SyntaxError(SyntaxError::UnexpectedToken {
                            found: potential_doc.to_string(),
                            expected: "string".to_string(),
                        }),
                    ));
                }
            }
            len => {
                return Err(error_at(
                    form_span.clone(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: format!(
                            "def requires at most 3 arguments, got {}",
                            len + 1
                        ),
                    }),
                ));
            }
        };

        if let Some(doc) = docstring {
            let doc_key = Value::Keyword {
                span: synthetic_span!(),
                value: interner::intern_kw("doc"),
            };
            let doc_value =
                Value::String { span: synthetic_span!(), value: Arc::from(doc) };

            match sym_meta.as_mut() {
                Some(meta) => {
                    meta.insert(doc_key.clone(), doc_value.clone());
                }
                None => {
                    sym_meta = Some(Metadata::from([(doc_key, doc_value)]));
                }
            }
        }

        let mut value_env = env.clone();
        let evaluated = self.eval(&expr_value, &mut value_env)?;

        let var = Var::new_with_value(
            sym,
            value_env.ns.clone(),
            evaluated.clone(),
            sym_meta,
        );
        let var_arc = Arc::new(var);
        let var_value = Value::Var { span: sym_span, value: var_arc.clone() };

        // Define the variable in the namespace
        let mut new_env = value_env.define_var(sym, var_arc);
        // Set the value in the new environment
        new_env = new_env.set(sym, var_value.clone());
        Ok(Step::Value(var_value, new_env))
    }

    fn eval_defmacro(
        &self,
        args: List<Value>,
        env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.len() < 2 {
            return Err(error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "defmacro requires at least 2 arguments".to_string(),
                }),
            ));
        }

        // (defmacro name doc-string? attr-map? [params*] body)
        let mut iter = args.iter().peekable();
        let (macro_name, mut sym_meta, macro_name_span) = match iter.next() {
            Some(Value::Symbol { value, span }) => {
                (value.id(), value.metadata(), span.clone())
            }
            Some(other) => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidSymbol {
                        value: "First argument to defmacro must be a symbol"
                            .to_string(),
                    }),
                ));
            }
            None => {
                return Err(error_at(
                    form_span.clone(),
                    Error::SyntaxError(SyntaxError::InvalidSymbol {
                        value: "First argument to defmacro must be a symbol"
                            .to_string(),
                    }),
                ));
            }
        };

        let docstring = match iter.peek() {
            Some(Value::String { value, .. }) => {
                iter.next();
                Some(value.to_string())
            }
            _ => None,
        };

        // Add docstring to macro meta
        if let Some(doc) = docstring {
            let doc_key = Value::Keyword {
                span: synthetic_span!(),
                value: interner::intern_kw("doc"),
            };
            let doc_value =
                Value::String { span: synthetic_span!(), value: Arc::from(doc) };

            match sym_meta.as_mut() {
                Some(meta) => {
                    meta.insert(doc_key.clone(), doc_value.clone());
                }
                None => {
                    sym_meta = Some(Metadata::from([(doc_key, doc_value)]));
                }
            }
        }

        // Handle attr-map if present, adding it to macro meta
        if let Some(Value::Map { value: attr_map, .. }) = iter.peek() {
            iter.next();
            match sym_meta.as_mut() {
                Some(meta) => {
                    for (k, v) in attr_map.iter() {
                        meta.insert(k.clone(), v.clone());
                    }
                }
                None => {
                    let mut new_meta = Metadata::new();
                    for (k, v) in attr_map.iter() {
                        new_meta.insert(k.clone(), v.clone());
                    }
                    sym_meta = Some(new_meta);
                }
            }
        };

        let params = match iter.next() {
            Some(Value::Vector { value, .. }) => value.clone(),
            Some(other) => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "defmacro requires a vector of parameters"
                            .to_string(),
                    }),
                ));
            }
            None => {
                return Err(error_at(
                    form_span.clone(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "defmacro requires a vector of parameters"
                            .to_string(),
                    }),
                ));
            }
        };

        let body = iter.cloned().collect::<List<Value>>();
        let body_arc = if body.is_empty() {
            Arc::new(List::new().prepend(Value::Nil { span: synthetic_span!() }))
        } else {
            Arc::new(body)
        };

        let func = Value::Function {
            span: form_span.clone(),
            name: Some(macro_name),
            params,
            body: body_arc,
            env: Arc::new(env.clone()),
        };

        // A macro is just a Var with is_macro=true
        let macro_var =
            Var::new(macro_name, env.ns.clone(), Some(func), sym_meta, true, false);
        let var_arc = Arc::new(macro_var);

        let new_env = env.define_var(macro_name, var_arc.clone());
        Ok(Step::Value(
            Value::Var { span: macro_name_span, value: var_arc },
            new_env,
        ))
    }

    fn eval_if(
        &self,
        args: List<Value>,
        mut env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.len() < 2 || args.len() > 3 {
            return Err(error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "Wrong number of arguments to if. Expecting 2 or 3"
                        .to_string(),
                }),
            ));
        }

        let mut iter = args.iter();
        let condition = iter.next().unwrap();
        let then_form = iter.next().unwrap().clone();
        let else_form = iter.next().cloned();

        let cond_value = self.eval(condition, &mut env)?;

        match cond_value {
            Value::Bool { value: false, .. } | Value::Nil { .. } => {
                if let Some(else_branch) = else_form {
                    Ok(Step::Tail { form: else_branch, env, restore: None })
                } else {
                    Ok(Step::Value(Value::Nil { span: form_span }, env))
                }
            }
            _ => Ok(Step::Tail { form: then_form, env, restore: None }),
        }
    }

    fn eval_do(
        &self,
        args: List<Value>,
        mut env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.is_empty() {
            return Ok(Step::Value(Value::Nil { span: form_span }, env));
        }

        let mut forms: Vec<Value> = args.iter().cloned().collect();
        let last = forms.pop().unwrap();

        for form in forms {
            let _ = self.eval(&form, &mut env)?;
        }

        Ok(Step::Tail { form: last, env, restore: None })
    }

    fn eval_quote(
        &self,
        args: List<Value>,
        env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.len() != 1 {
            return Err(error_at(
                form_span,
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "quote requires exactly 1 argument".to_string(),
                }),
            ));
        }

        Ok(Step::Value(args.head().unwrap().clone(), env))
    }

    fn eval_let(
        &self,
        args: List<Value>,
        mut env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.len() < 1 || args.len() > 2 {
            return Err(error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "Wrong number of arguments to let. Expecting 1 or 2"
                        .to_string(),
                }),
            ));
        }

        let bindings_form = args.head().unwrap();
        let bindings = match bindings_form {
            Value::Vector { value, .. } => value.clone(),
            _ => {
                return Err(error_at(
                    bindings_form.span(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "First argument to let must be a vector"
                            .to_string(),
                    }),
                ));
            }
        };

        if bindings.len() % 2 == 1 {
            return Err(error_at(
                bindings_form.span(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "let binding vector requires an even number of forms"
                        .to_string(),
                }),
            ));
        }

        let mut binding_pairs = Vec::new();
        let mut iter = bindings.iter();
        while let (Some(param), Some(val_expr)) = (iter.next(), iter.next()) {
            match param {
                Value::Symbol { .. } => {
                    let evaluated = self.eval(val_expr, &mut env)?;
                    binding_pairs.push(param.clone());
                    binding_pairs.push(evaluated);
                }
                _ => {
                    return Err(error_at(
                        param.span(),
                        Error::SyntaxError(SyntaxError::InvalidSymbol {
                            value: "let binding parameters must be symbols"
                                .to_string(),
                        }),
                    ));
                }
            }
        }

        let bindings_vec = Arc::new(Vector::from_iter(binding_pairs));
        let mut local_env = env.create_child_with_bindings(bindings_vec);

        match args.tail() {
            Some(body) => {
                if body.is_empty() {
                    Ok(Step::Value(Value::Nil { span: form_span.clone() }, env))
                } else {
                    let mut forms: Vec<Value> = body.iter().cloned().collect();
                    let last = forms.pop().unwrap();

                    for form in forms {
                        let _ = self.eval(&form, &mut local_env)?;
                    }

                    let value = self.eval(&last, &mut local_env)?;
                    Ok(Step::Value(value, env))
                }
            }
            None => Ok(Step::Value(Value::Nil { span: form_span }, env)),
        }
    }

    fn eval_fn(
        &self,
        args: List<Value>,
        env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.is_empty() {
            return Err(error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str:
                        "Wrong number of arguments given to fn. Expecting at least 1"
                            .to_string(),
                }),
            ));
        }

        // (fn name? [params* ] expr*)
        let mut iter = args.iter().peekable();
        let fn_name = match iter.peek() {
            Some(Value::Symbol { value, .. }) => {
                let sym = value.id();
                iter.next();
                Some(sym)
            }
            _ => None,
        };

        let params = match iter.next() {
            Some(Value::Vector { value, .. }) => value.clone(),
            Some(other) => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "fn arguments must be in the form of a vector"
                            .to_string(),
                    }),
                ));
            }
            None => {
                return Err(error_at(
                    form_span.clone(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "fn arguments must be in the form of a vector"
                            .to_string(),
                    }),
                ));
            }
        };

        let body = iter.cloned().collect::<List<Value>>();
        let body_arc = if body.is_empty() {
            Arc::new(List::new().prepend(Value::Nil { span: synthetic_span!() }))
        } else {
            Arc::new(body)
        };

        Ok(Step::Value(
            Value::Function {
                span: form_span,
                name: fn_name,
                params,
                body: body_arc,
                env: Arc::new(env.clone()),
            },
            env,
        ))
    }

    fn eval_loop(
        &self,
        args: List<Value>,
        mut env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        if args.is_empty() {
            return Err(error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "loop requires at least 1 argument".to_string(),
                }),
            ));
        }

        let bindings_value = args.head().unwrap();
        let (initial_bindings, loop_symbols) = match bindings_value {
            Value::Vector { value: vector, .. } => {
                if vector.len() % 2 == 1 {
                    return Err(error_at(
                        bindings_value.span(),
                        Error::SyntaxError(SyntaxError::WrongArgumentCount {
                            error_str:
                                "loop binding vector requires an even number of forms".to_string(),
                        }),
                    ));
                }

                let mut binding_pairs = Vec::new();
                let mut symbol_bindings = Vec::new();
                let mut iter = vector.iter();
                while let (Some(param), Some(val_expr)) = (iter.next(), iter.next())
                {
                    match param {
                        Value::Symbol { .. } => {
                            let evaluated = self.eval(val_expr, &mut env)?;
                            symbol_bindings.push(param.clone());
                            binding_pairs.push(param.clone());
                            binding_pairs.push(evaluated);
                        }
                        _ => {
                            return Err(error_at(
                                param.span(),
                                Error::SyntaxError(SyntaxError::InvalidSymbol {
                                    value: "loop binding parameters must be symbols"
                                        .to_string(),
                                }),
                            ));
                        }
                    }
                }

                (
                    Arc::new(Vector::from_iter(binding_pairs)),
                    Arc::new(Vector::from_iter(symbol_bindings)),
                )
            }
            _ => {
                return Err(error_at(
                    bindings_value.span(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "First argument to loop must be a vector"
                            .to_string(),
                    }),
                ));
            }
        };

        let body_list = Arc::new(args.tail().unwrap_or_else(List::new));

        let loop_env = env
            .create_child_with_bindings(initial_bindings)
            .with_recur_context(RecurContext::Loop {
                bindings: loop_symbols.clone(),
                body: body_list.clone(),
                span: form_span.clone(),
            });

        if body_list.is_empty() {
            return Ok(Step::Value(Value::Nil { span: form_span.clone() }, env));
        }

        let body_form = make_do_value(self, body_list, form_span.clone());
        Ok(Step::Tail { form: body_form, env: loop_env, restore: Some(env) })
    }

    fn eval_recur(
        &self,
        args: List<Value>,
        mut env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        let context = env.get_recur_context().cloned().ok_or_else(|| {
            error_at(
                form_span.clone(),
                Error::RuntimeError(
                    "recur can be used only inside loop or function".to_string(),
                ),
            )
        })?;

        let evaluated_args = args
            .iter()
            .map(|arg| self.eval(arg, &mut env))
            .collect::<Result<Vec<_>, _>>()?;

        let expected_len = match &context {
            RecurContext::Loop { bindings, .. } => bindings.len(),
            RecurContext::Function { params, .. } => params.len(),
        };

        if evaluated_args.len() != expected_len {
            return Err(error_at(
                form_span,
                Error::RuntimeError(format!(
                    "recur argument count mismatch: expected {}, got {}",
                    expected_len,
                    evaluated_args.len()
                )),
            ));
        }

        Ok(Step::Recur(
            context,
            Arc::new(Vector::from_iter(evaluated_args)),
            env,
            form_span,
        ))
    }

    fn eval_ns(
        &self,
        args: List<Value>,
        _env: Env,
        form_span: Span,
    ) -> EvalResult<Step> {
        use crate::core::namespace::{find_or_create_ns, set_current_ns, update_ns};

        if args.is_empty() {
            return Err(error_at(
                form_span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "ns requires at least a namespace name".to_string(),
                }),
            ));
        }

        // (ns my.namespace ...)
        // First argument must be a symbol
        let mut iter = args.iter();
        let ns_name_value = iter.next().unwrap();
        let ns_name = match ns_name_value {
            Value::Symbol { value: sym, .. } => interner::sym_to_str(sym.id()),
            other => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidSymbol {
                        value: "ns requires a symbol as namespace name".to_string(),
                    }),
                ));
            }
        };

        // Find or create the namespace
        let mut ns = find_or_create_ns(&ns_name);

        // Process remaining arguments (require declarations, etc.)
        for form in iter {
            match form {
                Value::List { value: list, .. } => {
                    if let Some(first) = list.head() {
                        match first {
                            Value::Keyword { value: kw, .. } => {
                                let kw_str = interner::kw_to_str(*kw);
                                match kw_str.as_str() {
                                    "require" => {
                                        // Process :require declarations
                                        ns = self.process_require(
                                            Arc::new(ns.as_ref().clone()),
                                            list.tail().unwrap_or_else(List::new),
                                            form_span.clone(),
                                        )?;
                                    }
                                    other => {
                                        return Err(error_at(
                                            first.span(),
                                            Error::RuntimeError(format!(
                                                "Unknown ns directive: :{}",
                                                other
                                            )),
                                        ));
                                    }
                                }
                            }
                            _ => {
                                // Skip other forms for now
                            }
                        }
                    }
                }
                _ => {
                    // Skip non-list forms
                }
            }
        }

        // Update the namespace in the registry
        let ns_arc = Arc::new(ns.as_ref().clone());
        update_ns(ns_arc.clone());

        // Set as current namespace
        set_current_ns(ns_arc.id);

        // Create new environment with the new namespace
        let new_env = Env::new(ns_arc);

        // Return the namespace symbol
        Ok(Step::Value(ns_name_value.clone(), new_env))
    }

    fn process_require(
        &self,
        mut ns: Arc<Namespace>,
        specs: List<Value>,
        form_span: Span,
    ) -> EvalResult<Arc<Namespace>> {
        use crate::core::namespace::find_or_create_ns;

        for spec in specs.iter() {
            match spec {
                // Simple require: foo.bar
                Value::Symbol { value: sym, .. } => {
                    let required_ns_name = interner::sym_to_str(sym.id());
                    let _required_ns = find_or_create_ns(&required_ns_name);
                    // Just ensure the namespace exists, no aliasing
                }
                // Vector form: [foo.bar :as fb] or [foo.bar :refer [x y z]]
                Value::Vector { value: vec, .. } => {
                    if vec.is_empty() {
                        continue;
                    }

                    let ns_sym = match vec.get(0) {
                        Some(Value::Symbol { value: sym, .. }) => sym.id(),
                        _ => {
                            return Err(error_at(
                                spec.span(),
                                Error::SyntaxError(SyntaxError::InvalidSymbol {
                                    value: "require spec must start with a symbol"
                                        .to_string(),
                                }),
                            ));
                        }
                    };

                    let required_ns_name = interner::sym_to_str(ns_sym);
                    let required_ns = find_or_create_ns(&required_ns_name);

                    // Process options (:as, :refer)
                    let mut i = 1;
                    while i < vec.len() {
                        let opt = vec.get(i);
                        match opt {
                            Some(Value::Keyword { value: kw, .. }) => {
                                let kw_str = interner::kw_to_str(*kw);
                                match kw_str.as_str() {
                                    "as" => {
                                        // :as alias
                                        i += 1;
                                        if let Some(Value::Symbol { value: alias, .. }) =
                                            vec.get(i)
                                        {
                                            let updated_ns =
                                                ns.add_alias(alias.id(), required_ns.id);
                                            ns = Arc::new(updated_ns);
                                        } else {
                                            return Err(error_at(
                                                form_span.clone(),
                                                Error::SyntaxError(
                                                    SyntaxError::InvalidSymbol {
                                                        value: ":as requires an alias symbol"
                                                            .to_string(),
                                                    },
                                                ),
                                            ));
                                        }
                                    }
                                    "refer" => {
                                        // :refer [sym1 sym2 ...]
                                        i += 1;
                                        if let Some(Value::Vector { value: refer_vec, .. }) =
                                            vec.get(i)
                                        {
                                            for refer_item in refer_vec.iter() {
                                                if let Value::Symbol { value: ref_sym, .. } =
                                                    refer_item
                                                {
                                                    // Look up the var in the required namespace
                                                    if let Some(var) =
                                                        required_ns.get(ref_sym.id())
                                                    {
                                                        let updated_ns = ns
                                                            .add_refer(ref_sym.id(), var.clone());
                                                        ns = Arc::new(updated_ns);
                                                    }
                                                    // If var not found, silently skip for now
                                                    // (could error in strict mode)
                                                }
                                            }
                                        } else {
                                            return Err(error_at(
                                                form_span.clone(),
                                                Error::SyntaxError(
                                                    SyntaxError::WrongArgumentCount {
                                                        error_str: ":refer requires a vector of symbols".to_string(),
                                                    },
                                                ),
                                            ));
                                        }
                                    }
                                    _ => {
                                        // Unknown option, skip
                                    }
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                }
                _ => {
                    return Err(error_at(
                        spec.span(),
                        Error::SyntaxError(SyntaxError::InvalidSymbol {
                            value: "require spec must be a symbol or vector".to_string(),
                        }),
                    ));
                }
            }
        }

        Ok(ns)
    }

    pub fn is_special_form(&self, sym: SymId) -> bool {
        sym == self.special_forms.s_def
            || sym == self.special_forms.s_defmacro
            || sym == self.special_forms.s_if
            || sym == self.special_forms.s_let
            || sym == self.special_forms.s_do
            || sym == self.special_forms.s_quote
            || sym == self.special_forms.s_loop
            || sym == self.special_forms.s_recur
            || sym == self.special_forms.s_fn
            || sym == self.special_forms.s_ns
    }
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

pub fn make_do_value(
    evaluator: &Evaluator,
    body: Arc<List<Value>>,
    span: Span,
) -> Value {
    if body.is_empty() {
        Value::Nil { span }
    } else {
        let forms: Vec<Value> = body.iter().cloned().collect();
        let mut list = List::new();
        for form in forms.into_iter().rev() {
            list = list.prepend(form);
        }
        list = list.prepend(Value::SpecialForm {
            span: span.clone(),
            name: evaluator.special_forms.s_do,
        });
        Value::List { span, value: Arc::new(list), meta: None }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::List;
    use crate::core::native_fns::{macroexpand, macroexpand1};
    use crate::value;
    use crate::runtime::Runtime;

    /// Setup a new evaluator and environment for testing.
    fn setup() -> (Evaluator, Env) {
        let runtime = Runtime::new();
        let eval = Evaluator::new();
        let ns = runtime.find_or_create_namespace("test");
        let env = Env::new(ns);
        (eval, env)
    }

    /// Helper function to create a special form
    /// with a list that starts with a special form symbol.
    fn make_special_form_list(sym: SymId, args: Vec<Value>) -> Value {
        let mut list = List::new();
        // Add args in reverse order
        for arg in args.into_iter().rev() {
            list = list.prepend(arg);
        }
        // Add special form at the front
        list =
            list.prepend(Value::SpecialForm { span: synthetic_span!(), name: sym });
        Value::List { span: synthetic_span!(), value: Arc::new(list), meta: None }
    }

    fn symbol(sym: SymId) -> Value {
        value::symbol(sym, None, None, synthetic_span!())
    }

    fn bool_val(value: bool) -> Value {
        Value::Bool { span: synthetic_span!(), value }
    }

    fn int_val(value: i64) -> Value {
        Value::Int { span: synthetic_span!(), value }
    }

    fn build_three_stage_recur_body(
        eval: &Evaluator,
        flag_a: SymId,
        flag_b: SymId,
        flag_c: SymId,
        result_sym: SymId,
    ) -> Value {
        let recur_stage_two = make_special_form_list(
            eval.special_forms.s_recur,
            vec![bool_val(false), bool_val(true), bool_val(false), int_val(0)],
        );

        let recur_stage_three = make_special_form_list(
            eval.special_forms.s_recur,
            vec![bool_val(false), bool_val(false), bool_val(true), int_val(1)],
        );

        let final_if = make_special_form_list(
            eval.special_forms.s_if,
            vec![symbol(flag_c), symbol(result_sym), int_val(-1)],
        );

        let inner_if = make_special_form_list(
            eval.special_forms.s_if,
            vec![symbol(flag_b), recur_stage_three, final_if],
        );

        make_special_form_list(
            eval.special_forms.s_if,
            vec![symbol(flag_a), recur_stage_two, inner_if],
        )
    }

    //===----------------------------------------------------------------------===//
    // Basic Evaluation Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_eval_integer() {
        let (eval, mut env) = setup();
        let form = Value::Int { span: synthetic_span!(), value: 42 };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int(42), got {:?}", result),
        }
    }

    #[test]
    fn test_trampoline_eval_integer() {
        let (eval, mut env) = setup();
        let form = Value::Int { span: synthetic_span!(), value: 7 };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 7),
            _ => panic!("Expected Int(7), got {:?}", result),
        }
    }

    #[test]
    fn test_eval_float() {
        let (eval, mut env) = setup();
        let form = Value::Float { span: synthetic_span!(), value: 3.14 };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Float { value, .. } => {
                assert!((value - 3.14).abs() < f64::EPSILON)
            }
            _ => panic!("Expected Float(3.14), got {:?}", result),
        }
    }

    #[test]
    fn test_eval_string() {
        let (eval, mut env) = setup();
        let form =
            Value::String { span: synthetic_span!(), value: Arc::from("hello") };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::String { value, .. } => assert_eq!(value.as_ref(), "hello"),
            _ => panic!("Expected String(\"hello\"), got {:?}", result),
        }
    }

    #[test]
    fn test_trampoline_eval_symbol_defined() {
        let (eval, mut env) = setup();
        let sym_id = interner::intern_sym("x");
        let value = Value::Int { span: synthetic_span!(), value: 21 };
        env = env.set(sym_id, value.clone());

        let form = symbol(sym_id);
        let result = eval.eval(&form, &mut env).unwrap();

        match result {
            Value::Int { value: returned, .. } => assert_eq!(returned, 21),
            _ => panic!("Expected Int(21), got {:?}", result),
        }
    }

    #[test]
    fn test_trampoline_eval_symbol_undefined() {
        let (eval, mut env) = setup();
        let sym_id = interner::intern_sym("undefined-var");
        let form = symbol(sym_id);

        let result = eval.eval(&form, &mut env);
        assert!(result.is_err());
        match result {
            Err(err) => match err.error {
                Error::RuntimeError(msg) => {
                    assert!(msg.contains("Undefined symbol"));
                }
                other => panic!("Expected RuntimeError, got {:?}", other),
            },
            _ => panic!("Expected RuntimeError, got {:?}", result),
        }
    }

    #[test]
    fn test_trampoline_if_branches() {
        let (eval, mut env) = setup();
        let form = make_special_form_list(
            eval.special_forms.s_if,
            vec![
                Value::Bool { span: synthetic_span!(), value: true },
                Value::Int { span: synthetic_span!(), value: 1 },
                Value::Int { span: synthetic_span!(), value: 2 },
            ],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    #[test]
    fn test_trampoline_do_last_value() {
        let (eval, mut env) = setup();
        let form = make_special_form_list(
            eval.special_forms.s_do,
            vec![
                Value::Int { span: synthetic_span!(), value: 1 },
                Value::Int { span: synthetic_span!(), value: 2 },
                Value::Int { span: synthetic_span!(), value: 3 },
            ],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 3),
            other => panic!("Expected Int(3), got {:?}", other),
        }
    }

    #[test]
    fn test_trampoline_quote_returns_literal() {
        let (eval, mut env) = setup();
        let quoted = Value::Int { span: synthetic_span!(), value: 10 };
        let form =
            make_special_form_list(eval.special_forms.s_quote, vec![quoted.clone()]);

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 10),
            other => panic!("Expected Int(10), got {:?}", other),
        }
    }

    #[test]
    fn test_trampoline_let_scoping() {
        let (eval, mut env) = setup();
        let x_sym = interner::intern_sym("x");
        let bindings_vec = Vector::from_iter(vec![
            symbol(x_sym),
            Value::Int { span: synthetic_span!(), value: 10 },
        ]);
        let form = make_special_form_list(
            eval.special_forms.s_let,
            vec![
                Value::Vector {
                    span: synthetic_span!(),
                    value: Arc::new(bindings_vec),
                    meta: None,
                },
                symbol(x_sym),
            ],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 10),
            other => panic!("Expected Int(10), got {:?}", other),
        }
    }

    #[test]
    fn test_trampoline_def_updates_env() {
        let (eval, mut env) = setup();
        let sym = interner::intern_sym("def-test");
        let form = make_special_form_list(
            eval.special_forms.s_def,
            vec![symbol(sym), Value::Int { span: synthetic_span!(), value: 99 }],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Var { .. } => {
                let stored = env.ns.get(sym).expect("var stored in namespace");
                let inner = stored.value.as_ref().unwrap().read().unwrap();
                match &*inner {
                    Value::Int { value, .. } => assert_eq!(*value, 99),
                    other => panic!("Expected Int(99), found {:?}", other),
                }
            }
            other => panic!("Expected Var value, got {:?}", other),
        }
    }

    #[test]
    fn test_trampoline_loop_recur_tail_calls() {
        let (eval, mut env) = setup();

        let flag_a = interner::intern_sym("flag-a");
        let flag_b = interner::intern_sym("flag-b");
        let flag_c = interner::intern_sym("flag-c");
        let result_sym = interner::intern_sym("loop-result");

        let bindings_vec = Vector::from_iter(vec![
            symbol(flag_a),
            bool_val(true),
            symbol(flag_b),
            bool_val(false),
            symbol(flag_c),
            bool_val(false),
            symbol(result_sym),
            int_val(0),
        ]);

        let bindings_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(bindings_vec),
            meta: None,
        };

        let loop_body =
            build_three_stage_recur_body(&eval, flag_a, flag_b, flag_c, result_sym);

        let loop_form = make_special_form_list(
            eval.special_forms.s_loop,
            vec![bindings_value, loop_body],
        );

        let result = eval.eval(&loop_form, &mut env).unwrap();

        match result {
            Value::Int { value, .. } => assert_eq!(value, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    #[test]
    fn test_eval_bool() {
        let (eval, mut env) = setup();
        let form = Value::Bool { span: synthetic_span!(), value: true };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Bool { value, .. } => assert_eq!(value, true),
            _ => panic!("Expected Bool(true), got {:?}", result),
        }
    }

    #[test]
    fn test_eval_nil() {
        let (eval, mut env) = setup();
        let form = Value::Nil { span: synthetic_span!() };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Nil { .. } => {}
            _ => panic!("Expected Nil, got {:?}", result),
        }
    }

    #[test]
    fn test_eval_empty_list() {
        let (eval, mut env) = setup();
        let form = Value::List {
            span: synthetic_span!(),
            value: Arc::new(List::new()),
            meta: None,
        };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::List { value: list, .. } => {
                assert!(list.is_empty());
            }
            _ => panic!("Expected Nil for empty list, got {:?}", result),
        }
    }

    #[test]
    fn test_eval_vector() {
        let (eval, mut env) = setup();
        let vector = Vector::from_iter(vec![
            Value::Int { span: synthetic_span!(), value: 1 },
            Value::Int { span: synthetic_span!(), value: 2 },
            Value::Int { span: synthetic_span!(), value: 3 },
        ]);
        let form = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(vector),
            meta: None,
        };
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Vector { value, .. } => {
                assert_eq!(value.len(), 3);
                assert_eq!(value.get(0).unwrap().to_string(), "1");
                assert_eq!(value.get(1).unwrap().to_string(), "2");
                assert_eq!(value.get(2).unwrap().to_string(), "3");
            }
            _ => panic!("Expected Vector, got {:?}", result),
        }
    }

    #[test]
    fn test_eval_native_function_symbol() {
        let (eval, mut env) = setup();
        let sym_id = interner::intern_sym("concat");
        let form = symbol(sym_id);
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::NativeFunction { name, .. } => assert_eq!(name, sym_id),
            _ => {
                panic!("Expected NativeFunction for concat symbol, got {:?}", result)
            }
        }
    }

    #[test]
    fn test_eval_native_function_application() {
        let (eval, mut env) = setup();

        let form = value::list_from_vec(
            // (concat )
            vec![
                symbol(interner::intern_sym("concat")),
                value::list_from_vec(
                    vec![
                        symbol(interner::intern_sym("quote")),
                        value::list_from_vec(
                            vec![
                                value::string("b", synthetic_span!()),
                                value::string("a", synthetic_span!()),
                            ],
                            synthetic_span!(),
                        ),
                    ],
                    synthetic_span!(),
                ),
            ],
            synthetic_span!(),
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::List { value: list, .. } => {
                assert_eq!(list.len(), 2);
                assert_eq!(list.head().unwrap().to_string(), "b");
                assert_eq!(list.tail().unwrap().head().unwrap().to_string(), "a");
            }
            _ => panic!("Expected concatenated string, got {:?}", result),
        }
    }

    //===----------------------------------------------------------------------===//
    // Symbol Evaluation Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_eval_symbol_undefined() {
        let (eval, mut env) = setup();
        let sym_id = interner::intern_sym("undefined-var");
        let form = symbol(sym_id);
        let result = eval.eval(&form, &mut env);
        assert!(result.is_err());
        match result {
            Err(err) => match err.error {
                Error::RuntimeError(msg) => {
                    assert!(msg.contains("Undefined symbol"));
                }
                other => panic!("Expected RuntimeError, got {:?}", other),
            },
            _ => panic!("Expected RuntimeError, got {:?}", result),
        }
    }

    #[test]
    fn test_eval_symbol_defined() {
        let (eval, mut env) = setup();
        let sym_id = interner::intern_sym("x");
        let value = Value::Int { span: synthetic_span!(), value: 42 };
        env = env.set(sym_id, value.clone());
        let form = symbol(sym_id);
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value: v, .. } => assert_eq!(v, 42),
            _ => panic!("Expected Int(42), got {:?}", result),
        }
    }

    //===----------------------------------------------------------------------===//
    // Special Form Tests: quote
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_eval_quote_returns_literal() {
        let (eval, mut env) = setup();
        let quoted = Value::Int { span: synthetic_span!(), value: 42 };
        let form =
            make_special_form_list(eval.special_forms.s_quote, vec![quoted.clone()]);

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 42),
            other => panic!("Expected Int(42), got {:?}", other),
        }
    }

    //===----------------------------------------------------------------------===//
    // Special Form Tests: if
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_eval_if_truthy_branch() {
        let (eval, mut env) = setup();
        let form = make_special_form_list(
            eval.special_forms.s_if,
            vec![bool_val(true), int_val(42), int_val(0)],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 42),
            other => panic!("Expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_eval_if_falsey_branch() {
        let (eval, mut env) = setup();
        let form = make_special_form_list(
            eval.special_forms.s_if,
            vec![bool_val(false), int_val(42), int_val(7)],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 7),
            other => panic!("Expected Int(7), got {:?}", other),
        }
    }

    #[test]
    fn test_eval_if_without_else_returns_nil() {
        let (eval, mut env) = setup();
        let form = make_special_form_list(
            eval.special_forms.s_if,
            vec![bool_val(false), int_val(42)],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Nil { .. } => {}
            other => panic!("Expected Nil, got {:?}", other),
        }
    }

    #[test]
    fn test_eval_if_nil_condition_treated_as_false() {
        let (eval, mut env) = setup();
        let form = make_special_form_list(
            eval.special_forms.s_if,
            vec![Value::Nil { span: synthetic_span!() }, int_val(1), int_val(2)],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 2),
            other => panic!("Expected Int(2), got {:?}", other),
        }
    }

    #[test]
    fn test_eval_if_wrong_args() {
        let (eval, mut env) = setup();
        // Empty list (just special form) should fail
        let form = Value::List {
            span: synthetic_span!(),
            value: Arc::new(List::new().prepend(Value::SpecialForm {
                span: synthetic_span!(),
                name: eval.special_forms.s_if,
            })),
            meta: None,
        };
        let result = eval.eval(&form, &mut env);
        assert!(result.is_err());
    }

    //===----------------------------------------------------------------------===//
    // Special Form Tests: do
    //===----------------------------------------------------------------------===//
    #[test]
    fn test_eval_do_returns_last() {
        let (eval, mut env) = setup();
        let form = make_special_form_list(
            eval.special_forms.s_do,
            vec![
                Value::Int { span: synthetic_span!(), value: 1 },
                Value::Int { span: synthetic_span!(), value: 2 },
                Value::Int { span: synthetic_span!(), value: 3 },
            ],
        );
        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 3),
            _ => panic!("Expected Int(3), got {:?}", result),
        }
    }

    #[test]
    fn test_eval_do_empty_returns_nil() {
        let (eval, mut env) = setup();

        let form = make_special_form_list(eval.special_forms.s_do, vec![]);
        let result = eval.eval(&form, &mut env).unwrap();

        match result {
            Value::Nil { .. } => {}
            other => panic!("Expected Nil, got {:?}", other),
        }
    }

    #[test]
    fn test_eval_do_executes_side_effects_in_order() {
        let (eval, mut env) = setup();
        let flag_a = interner::intern_sym("do-flag-a");
        let flag_b = interner::intern_sym("do-flag-b");

        let set_flag_b = make_special_form_list(
            eval.special_forms.s_let,
            vec![
                Value::Vector {
                    span: synthetic_span!(),
                    value: Arc::new(Vector::from_iter(vec![
                        symbol(flag_b),
                        bool_val(true),
                    ])),
                    meta: None,
                },
                symbol(flag_b),
            ],
        );

        let forms = vec![
            make_special_form_list(
                eval.special_forms.s_let,
                vec![
                    Value::Vector {
                        span: synthetic_span!(),
                        value: Arc::new(Vector::from_iter(vec![
                            symbol(flag_a),
                            bool_val(true),
                        ])),
                        meta: None,
                    },
                    symbol(flag_a),
                ],
            ),
            set_flag_b.clone(),
            Value::Int { span: synthetic_span!(), value: 5 },
        ];

        let form = make_special_form_list(eval.special_forms.s_do, forms);
        let result = eval.eval(&form, &mut env).unwrap();

        match result {
            Value::Int { value, .. } => assert_eq!(value, 5),
            other => panic!("Expected Int(5), got {:?}", other),
        }
    }

    #[test]
    fn test_eval_do_will_restore_env_after_tail() {
        let (eval, mut env) = setup();
        let sym = interner::intern_sym("outer");
        env = env.set(sym, int_val(1));

        let inner_binding = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(Vector::from_iter(vec![symbol(sym), int_val(2)])),
            meta: None,
        };
        let inner_form = make_special_form_list(
            eval.special_forms.s_let,
            vec![inner_binding, symbol(sym)],
        );

        let form = make_special_form_list(
            eval.special_forms.s_do,
            vec![inner_form, symbol(sym)],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    //===----------------------------------------------------------------------===//
    // Special Form Tests: let
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_eval_let_binds_locals() {
        let (eval, mut env) = setup();

        let x_sym = interner::intern_sym("x");
        let bindings_vec = Vector::from_iter(vec![symbol(x_sym), int_val(10)]);
        let bindings_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(bindings_vec),
            meta: None,
        };

        let body = symbol(x_sym);

        let form = make_special_form_list(
            eval.special_forms.s_let,
            vec![bindings_value, body],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 10),
            other => panic!("Expected Int(10), got {:?}", other),
        }

        assert!(
            env.get(x_sym).is_none(),
            "let should not leak bindings into parent env"
        );
    }

    //===----------------------------------------------------------------------===//
    // Special Form Tests: fn
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_eval_fn_creates_function() {
        let (eval, mut env) = setup();

        let fn_name = interner::intern_sym("my-fn");
        let param_sym = interner::intern_sym("x");

        let params_vec = Vector::from_iter(vec![symbol(param_sym)]);
        let params_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(params_vec),
            meta: None,
        };

        let body_list = Arc::new(List::new().prepend(symbol(param_sym)));

        let form = make_special_form_list(
            eval.special_forms.s_fn,
            vec![
                symbol(fn_name),
                params_value,
                Value::List {
                    span: synthetic_span!(),
                    value: body_list.clone(),
                    meta: None,
                },
            ],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Function { name, params, body, .. } => {
                assert_eq!(name, Some(fn_name));
                assert_eq!(params.len(), 1);
                match params.get(0).unwrap() {
                    Value::Symbol { value, .. } => assert_eq!(value.id(), param_sym),
                    other => panic!("Expected symbol param, got {:?}", other),
                }
                assert_eq!(body.len(), 1);
            }
            other => panic!("Expected Value::Function, got {:?}", other),
        }
    }

    //===----------------------------------------------------------------------===//
    // Function Application Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_eval_function_application() {
        let (eval, mut env) = setup();

        let param_sym = interner::intern_sym("apply-x");
        let params_vec = Vector::from_iter(vec![symbol(param_sym)]);
        let params_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(params_vec),
            meta: None,
        };

        let body_form = symbol(param_sym);
        let fn_form = make_special_form_list(
            eval.special_forms.s_fn,
            vec![params_value, body_form],
        );

        let mut invocation_list = List::new();
        invocation_list = invocation_list.prepend(int_val(25));
        invocation_list = invocation_list.prepend(fn_form);
        let invocation = Value::List {
            span: synthetic_span!(),
            value: Arc::new(invocation_list),
            meta: None,
        };

        let result = eval.eval(&invocation, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 25),
            other => panic!("Expected Int(25), got {:?}", other),
        }
    }

    #[test]
    fn test_eval_function_application_wrong_arity() {
        let (eval, mut env) = setup();

        let param_sym = interner::intern_sym("apply-arity-x");
        let params_vec = Vector::from_iter(vec![symbol(param_sym)]);
        let params_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(params_vec),
            meta: None,
        };

        let body_form = symbol(param_sym);
        let fn_form = make_special_form_list(
            eval.special_forms.s_fn,
            vec![params_value, body_form],
        );

        let mut invocation_list = List::new();
        invocation_list = invocation_list.prepend(fn_form);
        let invocation = Value::List {
            span: synthetic_span!(),
            value: Arc::new(invocation_list),
            meta: None,
        };

        let result = eval.eval(&invocation, &mut env);
        assert!(result.is_err());
        match result {
            Err(err) => match err.error {
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str,
                }) => {
                    assert!(error_str.contains("Wrong number of arguments"));
                }
                other => {
                    panic!("Expected WrongArgumentCount error, got {:?}", other)
                }
            },
            other => panic!("Expected WrongArgumentCount error, got {:?}", other),
        }
    }

    //===----------------------------------------------------------------------===//
    // Environment Tests: create_child_with_bindings
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_create_child_with_bindings () {
        let (_eval, env) = setup();

        println!("test_create_child_with_bindings");

        // Create bindings: [sym1 val1 sym2 val2]
        let sym1 = interner::intern_sym("x");
        let sym2 = interner::intern_sym("y");
        let val1 = Value::Int { span: synthetic_span!(), value: 10 };
        let val2 = Value::Int { span: synthetic_span!(), value: 20 };

        let bindings = Arc::new(Vector::from_iter(vec![
            symbol(sym1),
            val1.clone(),
            symbol(sym2),
            val2.clone(),
        ]));

        let child_env = env.create_child_with_bindings(bindings);

        // Verify bindings are accessible in child
        assert_eq!(child_env.get(sym1), Some(&val1));
        assert_eq!(child_env.get(sym2), Some(&val2));
    }

    #[test]
    fn test_create_child_with_bindings_preserves_parent() {
        let (_eval, env) = setup();

        // Set a value in parent
        let parent_sym = interner::intern_sym("parent_var");
        let parent_val = Value::Int { span: synthetic_span!(), value: 100 };
        let env_with_parent = env.set(parent_sym, parent_val.clone());

        // Create child with new bindings
        let child_sym = interner::intern_sym("child_var");
        let child_val = Value::Int { span: synthetic_span!(), value: 200 };

        let bindings =
            Arc::new(Vector::from_iter(vec![symbol(child_sym), child_val.clone()]));

        let child_env = env_with_parent.create_child_with_bindings(bindings);

        // Verify both parent and child bindings are accessible
        assert_eq!(child_env.get(parent_sym), Some(&parent_val));
        assert_eq!(child_env.get(child_sym), Some(&child_val));
    }

    #[test]
    fn test_let_with_bindings() {
        let (eval, mut env) = setup();

        // (let [x 10 y 20] x) - should return 10
        let x_sym = interner::intern_sym("x");
        let y_sym = interner::intern_sym("y");

        let bindings_vec = Vector::from_iter(vec![
            symbol(x_sym),
            Value::Int { span: synthetic_span!(), value: 10 },
            symbol(y_sym),
            Value::Int { span: synthetic_span!(), value: 20 },
        ]);

        // Body is just the symbol x (not a list)
        let body = symbol(x_sym);

        let form = make_special_form_list(
            eval.special_forms.s_let,
            vec![
                Value::Vector {
                    span: synthetic_span!(),
                    value: Arc::new(bindings_vec),
                    meta: None,
                },
                body,
            ],
        );

        let result = eval.eval(&form, &mut env).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 10),
            _ => panic!("Expected Int(10), got {:?}", result),
        }
    }

    #[test]
    fn test_loop_recur_tail_calls() {
        let (eval, mut env) = setup();

        let flag_a = interner::intern_sym("flag-a");
        let flag_b = interner::intern_sym("flag-b");
        let flag_c = interner::intern_sym("flag-c");
        let result_sym = interner::intern_sym("loop-result");

        let bindings_vec = Vector::from_iter(vec![
            symbol(flag_a),
            bool_val(true),
            symbol(flag_b),
            bool_val(false),
            symbol(flag_c),
            bool_val(false),
            symbol(result_sym),
            int_val(0),
        ]);

        let bindings_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(bindings_vec),
            meta: None,
        };

        let loop_body =
            build_three_stage_recur_body(&eval, flag_a, flag_b, flag_c, result_sym);

        let loop_form = make_special_form_list(
            eval.special_forms.s_loop,
            vec![bindings_value, loop_body],
        );

        let result = eval.eval(&loop_form, &mut env).unwrap();

        match result {
            Value::Int { value, .. } => assert_eq!(value, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    #[test]
    fn test_fn_recur_tail_calls() {
        let (eval, mut env) = setup();

        let flag_a = interner::intern_sym("flag-a");
        let flag_b = interner::intern_sym("flag-b");
        let flag_c = interner::intern_sym("flag-c");
        let result_sym = interner::intern_sym("fn-result");

        let params_vec = Vector::from_iter(vec![
            symbol(flag_a),
            symbol(flag_b),
            symbol(flag_c),
            symbol(result_sym),
        ]);

        let params_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(params_vec),
            meta: None,
        };

        let body_form =
            build_three_stage_recur_body(&eval, flag_a, flag_b, flag_c, result_sym);

        let fn_form = make_special_form_list(
            eval.special_forms.s_fn,
            vec![params_value, body_form],
        );

        let mut invocation_list = List::new();
        invocation_list = invocation_list.prepend(int_val(0));
        invocation_list = invocation_list.prepend(bool_val(false));
        invocation_list = invocation_list.prepend(bool_val(false));
        invocation_list = invocation_list.prepend(bool_val(true));
        invocation_list = invocation_list.prepend(fn_form);
        let invocation = Value::List {
            span: synthetic_span!(),
            value: Arc::new(invocation_list),
            meta: None,
        };

        let result = eval.eval(&invocation, &mut env).unwrap();

        match result {
            Value::Int { value, .. } => assert_eq!(value, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    #[test]
    fn test_trampoline_function_application() {
        let (eval, mut env) = setup();

        let param_sym = interner::intern_sym("x");

        let params_vec = Vector::from_iter(vec![symbol(param_sym)]);
        let params_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(params_vec),
            meta: None,
        };

        let body_value = symbol(param_sym);
        let fn_form = make_special_form_list(
            eval.special_forms.s_fn,
            vec![params_value, body_value],
        );

        let mut call_list = List::new();
        call_list =
            call_list.prepend(Value::Int { span: synthetic_span!(), value: 42 });
        call_list = call_list.prepend(fn_form);
        let call_form = Value::List {
            span: synthetic_span!(),
            value: Arc::new(call_list),
            meta: None,
        };

        let result = eval.eval(&call_form, &mut env).unwrap();

        match result {
            Value::Int { value, .. } => assert_eq!(value, 42),
            other => panic!("Expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_trampoline_fn_recur_tail_calls() {
        let (eval, mut env) = setup();

        let flag_a = interner::intern_sym("flag-a");
        let flag_b = interner::intern_sym("flag-b");
        let flag_c = interner::intern_sym("flag-c");
        let result_sym = interner::intern_sym("fn-result");
        let params_vec = Vector::from_iter(vec![
            symbol(flag_a),
            symbol(flag_b),
            symbol(flag_c),
            symbol(result_sym),
        ]);

        let params_value = Value::Vector {
            span: synthetic_span!(),
            value: Arc::new(params_vec),
            meta: None,
        };

        let body_form =
            build_three_stage_recur_body(&eval, flag_a, flag_b, flag_c, result_sym);

        let fn_form = make_special_form_list(
            eval.special_forms.s_fn,
            vec![params_value, body_form],
        );

        let mut call_list = List::new();
        call_list = call_list.prepend(int_val(0));
        call_list = call_list.prepend(bool_val(false));
        call_list = call_list.prepend(bool_val(false));
        call_list = call_list.prepend(bool_val(true));
        call_list = call_list.prepend(fn_form);
        let call_form = Value::List {
            span: synthetic_span!(),
            value: Arc::new(call_list),
            meta: None,
        };

        let result = eval.eval(&call_form, &mut env).unwrap();

        match result {
            Value::Int { value, .. } => assert_eq!(value, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    //===----------------------------------------------------------------------===//
    // Native Functions Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_native_function_macroexpand1() {
        let (_eval, mut env) = setup();

        // Define a simple macro (defmacro my-macro [x] x)
        let macro_sym = interner::intern_sym("my-macro");
        let param_sym = interner::intern_sym("x");
        let param_value = symbol(param_sym);
        let params = Arc::new(Vector::from_iter(vec![param_value.clone()]));
        let body = Arc::new(List::from_iter(vec![param_value.clone()]));

        let macro_fn = Value::Function {
            span: synthetic_span!(),
            name: Some(macro_sym),
            params: params.clone(),
            body,
            env: Arc::new(env.clone()),
        };

        let macro_var =
            Var::new(macro_sym, env.ns.clone(), Some(macro_fn), None, true, false);
        env = env.define_var(macro_sym, Arc::new(macro_var));

        // Construct a macro call form: (my-macro (+ 1 2))
        let plus_sym = interner::intern_sym("+");
        let arg_form = Value::List {
            span: synthetic_span!(),
            value: Arc::new(List::from_iter(vec![
                symbol(plus_sym),
                Value::Int { span: synthetic_span!(), value: 1 },
                Value::Int { span: synthetic_span!(), value: 2 },
            ])),
            meta: None,
        };
        let macro_call = Value::List {
            span: synthetic_span!(),
            value: Arc::new(List::from_iter(vec![
                symbol(macro_sym),
                arg_form.clone(),
            ])),
            meta: None,
        };

        let expanded = macroexpand1(&[macro_call], &mut env).unwrap();
        assert_eq!(expanded, arg_form);

        // Non-macro forms are returned unchanged
        let non_macro = Value::List {
            span: synthetic_span!(),
            value: Arc::new(List::from_iter(vec![
                symbol(plus_sym),
                Value::Int { span: synthetic_span!(), value: 1 },
                Value::Int { span: synthetic_span!(), value: 2 },
            ])),
            meta: None,
        };
        let unchanged = macroexpand1(&[non_macro.clone()], &mut env).unwrap();
        assert_eq!(unchanged, non_macro);
    }

    #[test]
    fn test_native_function_macroexpand() {
        let (eval, mut env) = setup();

        let inner_sym = interner::intern_sym("inner-macro");
        let inner_body_form = make_special_form_list(
            eval.special_forms.s_quote,
            vec![Value::Int { span: synthetic_span!(), value: 42 }],
        );
        let inner_macro_fn = Value::Function {
            span: synthetic_span!(),
            name: Some(inner_sym),
            params: Arc::new(Vector::new()),
            body: Arc::new(List::from_iter(vec![inner_body_form])),
            env: Arc::new(env.clone()),
        };
        let inner_macro_var = Var::new(
            inner_sym,
            env.ns.clone(),
            Some(inner_macro_fn),
            None,
            true,
            false,
        );
        env = env.define_var(inner_sym, Arc::new(inner_macro_var));

        let outer_sym = interner::intern_sym("outer-macro");
        let inner_call = Value::List {
            span: synthetic_span!(),
            value: Arc::new(List::from_iter(vec![symbol(inner_sym)])),
            meta: None,
        };
        let outer_body_form = make_special_form_list(
            eval.special_forms.s_quote,
            vec![inner_call.clone()],
        );
        let outer_macro_fn = Value::Function {
            span: synthetic_span!(),
            name: Some(outer_sym),
            params: Arc::new(Vector::new()),
            body: Arc::new(List::from_iter(vec![outer_body_form])),
            env: Arc::new(env.clone()),
        };
        let outer_macro_var = Var::new(
            outer_sym,
            env.ns.clone(),
            Some(outer_macro_fn),
            None,
            true,
            false,
        );
        env = env.define_var(outer_sym, Arc::new(outer_macro_var));

        let macro_call = Value::List {
            span: synthetic_span!(),
            value: Arc::new(List::from_iter(vec![symbol(outer_sym)])),
            meta: None,
        };

        let expanded = macroexpand(&[macro_call], &mut env).unwrap();
        assert_eq!(expanded, Value::Int { span: synthetic_span!(), value: 42 });

        let non_macro = Value::Int { span: synthetic_span!(), value: 7 };
        let unchanged = macroexpand(&[non_macro.clone()], &mut env).unwrap();
        assert_eq!(unchanged, non_macro);
    }
}
