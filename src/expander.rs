use std::collections::BTreeMap;

use crate::env::Env;
use crate::error::{Diagnostic, Error, SyntaxError};
use crate::interner::{self, KeywId, SymId};
use crate::reader::Source;
use crate::value::Value;
use logos::Span;

//===----------------------------------------------------------------------===//
// Core Symbols
//===----------------------------------------------------------------------===//

pub struct CoreSyms {
    pub s_def: SymId,
    pub s_if: SymId,
    pub s_let: SymId,
    pub s_do: SymId,
    pub s_quote: SymId,
    pub s_fn: SymId,
    pub s_loop: SymId,
    pub s_recur: SymId,
}

impl CoreSyms {
    pub fn new() -> Self {
        Self {
            s_def: interner::intern_sym("def"),
            s_if: interner::intern_sym("if"),
            s_let: interner::intern_sym("let"),
            s_do: interner::intern_sym("do"),
            s_quote: interner::intern_sym("quote"),
            s_fn: interner::intern_sym("fn"),
            s_loop: interner::intern_sym("loop"),
            s_recur: interner::intern_sym("recur"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    Sym(SymId),                  // single symbol
    Vector(Vec<Pattern>),        // [a b [c d]]
    Map(Vec<(KeywId, Pattern)>), // {:x a, :y b}
    Ignore,                      // _
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamList {
    pub params: Vec<Pattern>,       // fixed positional params
    pub rest: Option<Box<Pattern>>, // variadic tail (from `& name`), None if non-variadic
}

pub enum HIR {
    // Core lowered nodes:
    Def {
        name: SymId,
        value: Box<HIR>,
        doc: Option<String>,
        meta: Option<BTreeMap<KeywId, Value>>,
    },
    If {
        cond: Box<HIR>,
        then_: Box<HIR>,
        else_: Option<Box<HIR>>,
    },
    Let {
        bindings: Vec<(Pattern, HIR)>,
        body: Option<Vec<HIR>>,
    },
    Do {
        forms: Vec<HIR>,
    },
    Quote {
        datum: Value,
    }, // keep raw CST or a literal HIR
    Fn {
        params: ParamList,
        body: Vec<HIR>,
        name: Option<SymId>,
    },
    Loop {
        slots: Vec<(Pattern, HIR)>,
        body: Vec<HIR>,
    }, // supports `recur`
    Recur {
        args: Vec<HIR>,
    },

    // Non-core:
    Call {
        callee: Box<HIR>,
        args: Vec<HIR>,
    },
    Var {
        id: SymId,
    },
    Literal {
        lit: Value,
    },
}

pub struct Expander {
    pub env: Env,
    pub core_syms: CoreSyms,
    pub source: String,
    pub file: Source,
}

impl Expander {
    pub fn new(env: Env, source: String, file: Source) -> Self {
        Self { env, core_syms: CoreSyms::new(), source, file }
    }

    fn get_span(value: &Value) -> Span {
        match value {
            Value::Nil { span } => span.clone(),
            Value::Bool { span, .. } => span.clone(),
            Value::Char { span, .. } => span.clone(),
            Value::Int { span, .. } => span.clone(),
            Value::Float { span, .. } => span.clone(),
            Value::String { span, .. } => span.clone(),
            Value::Symbol { span, .. } => span.clone(),
            Value::Keyword { span, .. } => span.clone(),
            Value::List { span, .. } => span.clone(),
            Value::Vector { span, .. } => span.clone(),
            Value::Map { span, .. } => span.clone(),
            Value::Set { span, .. } => span.clone(),
            Value::Namespace { span, .. } => span.clone(),
            Value::Function { span, .. } => span.clone(),
            Value::SpecialForm { span, .. } => span.clone(),
        }
    }

    fn make_diagnostic(&self, error: Error, span: Span) -> Diagnostic {
        Diagnostic::new(error, span, self.source.clone(), self.file.clone())
    }

    pub fn expand(&self, value: &Value) -> Result<HIR, Diagnostic> {
        match value {
            // Lists:
            Value::List { span: s, value: xs, meta: _ } if !xs.is_empty() => {
                if let Value::Symbol { value: sym, .. } = &xs[0] {
                    return self.expand_head(*sym, &xs[1..], s.clone());
                }
                // ( <non-sym> ... ) => call: expand all, keep as Call
                let callee = self.expand(&xs[0])?;
                let args = xs[1..]
                    .iter()
                    .map(|x| self.expand(x))
                    .collect::<Result<_, _>>()?;
                Ok(HIR::Call { callee: Box::new(callee), args })
            }
            // Variables:
            Value::Symbol { value: sym, .. } => Ok(HIR::Var { id: *sym }),
            // Literals:
            Value::Nil { .. }
            | Value::Bool { .. }
            | Value::Int { .. }
            | Value::Float { .. }
            | Value::Char { .. }
            | Value::String { .. }
            | Value::Keyword { .. } => Ok(HIR::Literal { lit: value.clone() }),
            // Sequences:
            Value::Vector { .. } | Value::Map { .. } | Value::Set { .. } => {
                Ok(HIR::Literal { lit: value.clone() })
            }
            _ => Err(self.make_diagnostic(
                Error::RuntimeError("Unexpected form".to_string()),
                Self::get_span(value),
            )),
        }
    }

    pub fn expand_head(
        &self,
        head: SymId,
        args: &[Value],
        span: Span,
    ) -> Result<HIR, Diagnostic> {
        if head == self.core_syms.s_def {
            return self.expand_def(args, span);
        }
        if head == self.core_syms.s_if {
            return self.expand_if(args, span);
        }
        if head == self.core_syms.s_let {
            return self.expand_let(args, span);
        }
        if head == self.core_syms.s_do {
            return self.expand_do(args, span);
        }
        if head == self.core_syms.s_quote {
            return self.expand_quote(args, span);
        }
        if head == self.core_syms.s_fn {
            return self.expand_fn(args, span);
        }
        if head == self.core_syms.s_loop {
            return self.expand_loop(args, span);
        }
        if head == self.core_syms.s_recur {
            return self.expand_recur(args, span);
        }

        let callee = HIR::Var { id: head };
        let args = args.iter().map(|x| self.expand(x)).collect::<Result<_, _>>()?;
        Ok(HIR::Call { callee: Box::new(callee), args })
    }

    pub fn expand_def(&self, args: &[Value], span: Span) -> Result<HIR, Diagnostic> {
        if args.len() < 2 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "def".to_string(),
                }),
                span,
            ));
        }

        let (name, meta) = match &args[0] {
            Value::Symbol { value: sym, meta: m, .. } => (*sym, m.clone()),
            _ => {
                return Err(self.make_diagnostic(
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: format!(
                            "First argument to def must be a symbol, got {}",
                            args[0].to_string()
                        ),
                    }),
                    Self::get_span(&args[0]),
                ));
            }
        };

        let mut value_index = 1;
        let mut doc = None;

        if let Some(Value::String { value: doc_str, .. }) = args.get(1) {
            doc = Some(doc_str.clone());
            value_index = 2;
        }

        if value_index >= args.len() {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "def".to_string(),
                }),
                span,
            ));
        }

        if args.len() > value_index + 1 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooManyArguments {
                    value: "def".to_string(),
                }),
                Self::get_span(&args[value_index + 1]),
            ));
        }

        let value = self.expand(&args[value_index])?;
        Ok(HIR::Def { name, value: Box::new(value), doc, meta })
    }

    pub fn expand_if(&self, args: &[Value], span: Span) -> Result<HIR, Diagnostic> {
        if args.len() < 2 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "if".to_string(),
                }),
                span,
            ));
        }

        if args.len() > 3 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooManyArguments {
                    value: "if".to_string(),
                }),
                Self::get_span(&args[3]),
            ));
        }

        let cond = self.expand(&args[0])?;
        let then_ = self.expand(&args[1])?;
        let else_ = if args.len() == 3 {
            Some(Box::new(self.expand(&args[2])?))
        } else {
            None
        };
        Ok(HIR::If { cond: Box::new(cond), then_: Box::new(then_), else_ })
    }

    pub fn expand_let(&self, args: &[Value], span: Span) -> Result<HIR, Diagnostic> {
        if args.len() < 2 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "let".to_string(),
                }),
                span,
            ));
        }

        let bindings = match &args[0] {
            Value::Vector { value: bs, .. } => bs,
            _ => {
                return Err(self.make_diagnostic(
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: format!(
                            "First argument to let must be a vector, got {}",
                            args[0].to_string()
                        ),
                    }),
                    Self::get_span(&args[0]),
                ));
            }
        };

        let expanded_bindings =
            self.expand_bindings(bindings, "let", Self::get_span(&args[0]))?;

        let body_forms = &args[1..];
        let expanded_body = self.expand_body(body_forms)?;
        let body = if expanded_body.is_empty() { None } else { Some(expanded_body) };

        Ok(HIR::Let { bindings: expanded_bindings, body })
    }

    pub fn expand_do(&self, args: &[Value], _span: Span) -> Result<HIR, Diagnostic> {
        let forms = self.expand_body(args)?;
        Ok(HIR::Do { forms })
    }

    pub fn expand_quote(
        &self,
        args: &[Value],
        span: Span,
    ) -> Result<HIR, Diagnostic> {
        if args.is_empty() {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "quote".to_string(),
                }),
                span,
            ));
        }

        if args.len() > 1 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooManyArguments {
                    value: "quote".to_string(),
                }),
                Self::get_span(&args[1]),
            ));
        }

        Ok(HIR::Quote { datum: args[0].clone() })
    }

    pub fn expand_fn(&self, args: &[Value], span: Span) -> Result<HIR, Diagnostic> {
        if args.is_empty() {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "fn".to_string(),
                }),
                span,
            ));
        }

        let mut idx = 0;
        let mut name = None;

        if args.len() >= 2 {
            if let Value::Symbol { value: sym, .. } = &args[0] {
                if matches!(args[1], Value::Vector { .. }) {
                    name = Some(*sym);
                    idx = 1;
                }
            }
        }

        let params_value = match args.get(idx) {
            Some(v) => v,
            None => {
                return Err(self.make_diagnostic(
                    Error::SyntaxError(SyntaxError::TooFewArguments {
                        value: "fn".to_string(),
                    }),
                    span,
                ));
            }
        };

        let params =
            self.parse_params(params_value, Self::get_span(params_value))?;

        let body_values = &args[idx + 1..];
        if body_values.is_empty() {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "fn".to_string(),
                }),
                span,
            ));
        }

        let body = self.expand_body(body_values)?;

        Ok(HIR::Fn { params, body, name })
    }

    pub fn expand_loop(
        &self,
        args: &[Value],
        span: Span,
    ) -> Result<HIR, Diagnostic> {
        if args.len() < 2 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "loop".to_string(),
                }),
                span,
            ));
        }

        let raw_slots = match &args[0] {
            Value::Vector { value, .. } => value,
            other => {
                return Err(self.make_diagnostic(
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: format!(
                            "First argument to loop must be a vector, got {}",
                            other.to_string()
                        ),
                    }),
                    Self::get_span(&args[0]),
                ));
            }
        };

        let slots =
            self.expand_bindings(raw_slots, "loop", Self::get_span(&args[0]))?;

        let body = self.expand_body(&args[1..])?;
        if body.is_empty() {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::TooFewArguments {
                    value: "loop".to_string(),
                }),
                span,
            ));
        }

        Ok(HIR::Loop { slots, body })
    }

    pub fn expand_recur(
        &self,
        args: &[Value],
        _span: Span,
    ) -> Result<HIR, Diagnostic> {
        let expanded_args = self.expand_body(args)?;
        Ok(HIR::Recur { args: expanded_args })
    }

    fn expand_bindings(
        &self,
        bindings: &[Value],
        form: &str,
        span: Span,
    ) -> Result<Vec<(Pattern, HIR)>, Diagnostic> {
        if bindings.len() % 2 != 0 {
            return Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: format!(
                        "{} bindings must contain an even number of forms, got {}",
                        form,
                        bindings.len()
                    ),
                }),
                span,
            ));
        }

        let mut results = Vec::with_capacity(bindings.len() / 2);

        let mut iter = bindings.chunks_exact(2);
        for pair in &mut iter {
            let pattern = self.expand_pattern(&pair[0])?;
            let expr = self.expand(&pair[1])?;
            results.push((pattern, expr));
        }

        Ok(results)
    }

    fn expand_body(&self, forms: &[Value]) -> Result<Vec<HIR>, Diagnostic> {
        forms.iter().map(|form| self.expand(form)).collect()
    }

    fn expand_pattern(&self, value: &Value) -> Result<Pattern, Diagnostic> {
        match value {
            Value::Symbol { value: sym, .. } => {
                let name = interner::sym_to_str(*sym);
                if name == "_" {
                    Ok(Pattern::Ignore)
                } else {
                    Ok(Pattern::Sym(*sym))
                }
            }
            Value::Vector { value: items, .. } => {
                let mut patterns = Vec::with_capacity(items.len());
                for item in items.iter() {
                    patterns.push(self.expand_pattern(item)?);
                }
                Ok(Pattern::Vector(patterns))
            }
            Value::Map { value: map, .. } => {
                let mut entries = Vec::with_capacity(map.len());
                for (key, val) in map.iter() {
                    match key {
                        Value::Keyword { value: kw, .. } => {
                            let pat = self.expand_pattern(val)?;
                            entries.push((*kw, pat));
                        }
                        other => {
                            return Err(self.make_diagnostic(
                                Error::SyntaxError(SyntaxError::InvalidMap {
                                    reason: format!(
                                        "Pattern map keys must be keywords, got {}",
                                        other.to_string()
                                    ),
                                }),
                                Self::get_span(other),
                            ));
                        }
                    }
                }
                Ok(Pattern::Map(entries))
            }
            Value::List { .. } => {
                Err(self.make_diagnostic(
                    Error::SyntaxError(SyntaxError::InvalidList {
                        reason: "Lists cannot be used as patterns".to_string(),
                    }),
                    Self::get_span(value),
                ))
            }
            other => Err(self.make_diagnostic(
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: format!(
                        "Invalid pattern form: expected a symbol, vector, map, or _, got {}",
                        other.to_string()
                    ),
                }),
                Self::get_span(other),
            )),
        }
    }

    fn parse_params(
        &self,
        params_value: &Value,
        span: Span,
    ) -> Result<ParamList, Diagnostic> {
        let items = match params_value {
            Value::Vector { value, .. } => value,
            other => {
                return Err(self.make_diagnostic(
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: format!(
                            "Function parameter list must be a vector, got {}",
                            other.to_string()
                        ),
                    }),
                    Self::get_span(other),
                ));
            }
        };

        let mut params = Vec::new();
        let mut rest: Option<Box<Pattern>> = None;

        let mut idx = 0;
        while idx < items.len() {
            let item = &items[idx];
            if let Value::Symbol { value: sym, .. } = item {
                if interner::sym_to_str(*sym) == "&" {
                    if rest.is_some() {
                        return Err(self.make_diagnostic(
                            Error::SyntaxError(SyntaxError::WrongArgumentCount {
                                error_str: "Only one variadic marker '&' is allowed in parameter list".to_string(),
                            }),
                            Self::get_span(item),
                        ));
                    }

                    idx += 1;
                    if idx >= items.len() {
                        return Err(self.make_diagnostic(
                            Error::SyntaxError(SyntaxError::WrongArgumentCount {
                                error_str:
                                    "Parameter list missing binding after '&'"
                                        .to_string(),
                            }),
                            span,
                        ));
                    }

                    let rest_pattern = self.expand_pattern(&items[idx])?;
                    rest = Some(Box::new(rest_pattern));
                    idx += 1;

                    if idx < items.len() {
                        return Err(self.make_diagnostic(
                            Error::SyntaxError(SyntaxError::WrongArgumentCount {
                                error_str:
                                    "Parameters cannot appear after variadic binding"
                                        .to_string(),
                            }),
                            Self::get_span(&items[idx]),
                        ));
                    }

                    break;
                }
            }

            params.push(self.expand_pattern(item)?);
            idx += 1;
        }

        Ok(ParamList { params, rest })
    }
}
