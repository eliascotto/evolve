//! High-Level Intermediate Representation (HIR) for the Evolve compiler.
//!
//! HIR is a simplified, compiler-friendly representation that:
//! - Removes metadata (except on Def nodes)
//! - Normalizes syntax
//! - Identifies tail positions for TCO
//! - Supports pattern matching/destructuring
//!
//! The `Lowerer` transforms CST (`Value`) into HIR.

use std::sync::Arc;

use crate::collections::{List, Map, Vector};
use crate::core::special_forms::SpecialFormRegistry;
use crate::core::Metadata;
use crate::error::{error_at, Error, SpannedResult, SyntaxError};
use crate::interner::{self, KeywId, SymId};
use crate::reader::Span;
use crate::value::Value;

//===----------------------------------------------------------------------===//
// HIR - High-Level Intermediate Representation
//===----------------------------------------------------------------------===//

/// High-Level IR node representing a simplified form of the CST.
#[derive(Debug, Clone, PartialEq)]
pub enum HIR {
    /// A literal value (nil, bool, char, int, float, string, keyword)
    Literal {
        span: Span,
        value: Literal,
    },

    /// A symbol reference (variable lookup)
    Var {
        span: Span,
        name: SymId,
    },

    /// A quoted expression - the value is returned as-is
    Quote {
        span: Span,
        value: Box<Value>,
    },

    /// A definition: (def name value)
    Def {
        span: Span,
        name: SymId,
        value: Box<HIR>,
        meta: Option<Metadata>,
    },

    /// A macro definition: (defmacro name [params] body)
    DefMacro {
        span: Span,
        name: SymId,
        params: Vec<Pattern>,
        body: Vec<HIR>,
        meta: Option<Metadata>,
    },

    /// A conditional: (if cond then else?)
    If {
        span: Span,
        condition: Box<HIR>,
        then_branch: Box<HIR>,
        else_branch: Option<Box<HIR>>,
        /// Whether this if is in tail position
        is_tail: bool,
    },

    /// A let binding: (let* [bindings...] body...)
    Let {
        span: Span,
        bindings: Vec<(Pattern, HIR)>,
        body: Vec<HIR>,
        /// Whether this let is in tail position
        is_tail: bool,
    },

    /// A do block: (do forms...)
    Do {
        span: Span,
        forms: Vec<HIR>,
        /// Whether this do is in tail position
        is_tail: bool,
    },

    /// A function definition: (fn* name? [params...] body...)
    Fn {
        span: Span,
        name: Option<SymId>,
        params: Vec<Pattern>,
        body: Vec<HIR>,
    },

    /// A loop: (loop* [bindings...] body...)
    Loop {
        span: Span,
        bindings: Vec<(Pattern, HIR)>,
        body: Vec<HIR>,
    },

    /// A recur: (recur args...)
    Recur {
        span: Span,
        args: Vec<HIR>,
    },

    /// A function call: (f args...)
    Call {
        span: Span,
        callee: Box<HIR>,
        args: Vec<HIR>,
        /// Whether this call is in tail position
        is_tail: bool,
    },

    /// A vector literal: [items...]
    Vector {
        span: Span,
        items: Vec<HIR>,
    },

    /// A map literal: {key value ...}
    Map {
        span: Span,
        entries: Vec<(HIR, HIR)>,
    },

    /// A set literal: #{items...}
    Set {
        span: Span,
        items: Vec<HIR>,
    },

    /// A namespace declaration: (ns name)
    Ns {
        span: Span,
        name: SymId,
    },
}

impl HIR {
    /// Get the span of this HIR node.
    pub fn span(&self) -> &Span {
        match self {
            HIR::Literal { span, .. }
            | HIR::Var { span, .. }
            | HIR::Quote { span, .. }
            | HIR::Def { span, .. }
            | HIR::DefMacro { span, .. }
            | HIR::If { span, .. }
            | HIR::Let { span, .. }
            | HIR::Do { span, .. }
            | HIR::Fn { span, .. }
            | HIR::Loop { span, .. }
            | HIR::Recur { span, .. }
            | HIR::Call { span, .. }
            | HIR::Vector { span, .. }
            | HIR::Map { span, .. }
            | HIR::Set { span, .. }
            | HIR::Ns { span, .. } => span,
        }
    }

    /// Set the tail position flag on this HIR node if applicable.
    pub fn with_tail(self, is_tail: bool) -> Self {
        match self {
            HIR::If { span, condition, then_branch, else_branch, .. } => HIR::If {
                span,
                condition,
                then_branch: Box::new(then_branch.with_tail(is_tail)),
                else_branch: else_branch.map(|e| Box::new(e.with_tail(is_tail))),
                is_tail,
            },
            HIR::Let { span, bindings, body, .. } => {
                let mut body = body;
                if let Some(last) = body.pop() {
                    body.push(last.with_tail(is_tail));
                }
                HIR::Let { span, bindings, body, is_tail }
            }
            HIR::Do { span, forms, .. } => {
                let mut forms = forms;
                if let Some(last) = forms.pop() {
                    forms.push(last.with_tail(is_tail));
                }
                HIR::Do { span, forms, is_tail }
            }
            HIR::Call { span, callee, args, .. } => {
                HIR::Call { span, callee, args, is_tail }
            }
            other => other,
        }
    }
}

//===----------------------------------------------------------------------===//
// Literal
//===----------------------------------------------------------------------===//

/// Literal values that can appear in HIR.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Nil,
    Bool(bool),
    Char(char),
    Int(i64),
    Float(f64),
    String(Arc<str>),
    Keyword(KeywId),
}

//===----------------------------------------------------------------------===//
// Pattern - for destructuring
//===----------------------------------------------------------------------===//

/// Pattern for destructuring in let/fn bindings.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Bind a value to a symbol
    Bind { name: SymId, span: Span },

    /// Ignore the value (underscore `_`)
    Ignore { span: Span },

    /// Destructure a vector: [a b c]
    Vector { patterns: Vec<Pattern>, span: Span },

    /// Destructure a map: {key pattern ...}
    Map { entries: Vec<(KeywId, Pattern)>, span: Span },

    /// Rest pattern for variadic arguments: & rest
    Rest { pattern: Box<Pattern>, span: Span },
}

impl Pattern {
    /// Get the span of this pattern.
    pub fn span(&self) -> &Span {
        match self {
            Pattern::Bind { span, .. }
            | Pattern::Ignore { span }
            | Pattern::Vector { span, .. }
            | Pattern::Map { span, .. }
            | Pattern::Rest { span, .. } => span,
        }
    }

    /// Collect all bound symbol names from this pattern.
    pub fn bound_names(&self) -> Vec<SymId> {
        match self {
            Pattern::Bind { name, .. } => vec![*name],
            Pattern::Ignore { .. } => vec![],
            Pattern::Vector { patterns, .. } => {
                patterns.iter().flat_map(|p| p.bound_names()).collect()
            }
            Pattern::Map { entries, .. } => {
                entries.iter().flat_map(|(_, p)| p.bound_names()).collect()
            }
            Pattern::Rest { pattern, .. } => pattern.bound_names(),
        }
    }
}

//===----------------------------------------------------------------------===//
// Lowerer
//===----------------------------------------------------------------------===//

/// Transforms CST (Value) into HIR.
pub struct Lowerer {
    special_forms: SpecialFormRegistry,
    /// Symbol ID for `_` (underscore/ignore pattern)
    underscore_sym: SymId,
    /// Symbol ID for `&` (rest pattern)
    ampersand_sym: SymId,
}

impl Lowerer {
    /// Create a new Lowerer.
    pub fn new() -> Self {
        Self {
            special_forms: SpecialFormRegistry::new(),
            underscore_sym: interner::intern_sym("_"),
            ampersand_sym: interner::intern_sym("&"),
        }
    }

    /// Lower a Value (CST) to HIR.
    pub fn lower(&self, value: &Value) -> SpannedResult<HIR> {
        self.lower_inner(value, false)
    }

    /// Lower a Value to HIR, with tail position tracking.
    fn lower_inner(&self, value: &Value, is_tail: bool) -> SpannedResult<HIR> {
        match value {
            // Literals
            Value::Nil { span } => Ok(HIR::Literal { span: span.clone(), value: Literal::Nil }),
            Value::Bool { span, value: b } => {
                Ok(HIR::Literal { span: span.clone(), value: Literal::Bool(*b) })
            }
            Value::Char { span, value: c } => {
                Ok(HIR::Literal { span: span.clone(), value: Literal::Char(*c) })
            }
            Value::Int { span, value: n } => {
                Ok(HIR::Literal { span: span.clone(), value: Literal::Int(*n) })
            }
            Value::Float { span, value: f } => {
                Ok(HIR::Literal { span: span.clone(), value: Literal::Float(*f) })
            }
            Value::String { span, value: s } => {
                Ok(HIR::Literal { span: span.clone(), value: Literal::String(s.clone()) })
            }
            Value::Keyword { span, value: k } => {
                Ok(HIR::Literal { span: span.clone(), value: Literal::Keyword(*k) })
            }

            // Symbol - variable reference
            Value::Symbol { span, value: sym } => {
                Ok(HIR::Var { span: span.clone(), name: sym.id() })
            }

            // List - could be a special form or function call
            Value::List { span, value: list, .. } => self.lower_list(span, list, is_tail),

            // Vector literal
            Value::Vector { span, value: vec, .. } => self.lower_vector(span, vec),

            // Map literal
            Value::Map { span, value: map, .. } => self.lower_map(span, map),

            // Set literal
            Value::Set { span, value: set, .. } => self.lower_set(span, set),

            // These shouldn't appear in source code being lowered
            Value::Function { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a function value".to_string(),
                }),
            )),
            Value::Var { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a var value".to_string(),
                }),
            )),
            Value::NativeFunction { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a native function value".to_string(),
                }),
            )),
            Value::SpecialForm { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a special form value".to_string(),
                }),
            )),
            Value::Namespace { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a namespace value".to_string(),
                }),
            )),
            // Runtime values that shouldn't appear in source code
            Value::Atom { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower an atom value".to_string(),
                }),
            )),
            Value::Ref { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a ref value".to_string(),
                }),
            )),
            Value::Agent { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower an agent value".to_string(),
                }),
            )),
            Value::Condition { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a condition value".to_string(),
                }),
            )),
            Value::Restart { span, .. } => Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::InvalidList {
                    reason: "Cannot lower a restart value".to_string(),
                }),
            )),
        }
    }

    /// Lower a list form (special form or function call).
    fn lower_list(
        &self,
        span: &Span,
        list: &Arc<List<Value>>,
        is_tail: bool,
    ) -> SpannedResult<HIR> {
        if list.is_empty() {
            // Empty list is a literal
            return Ok(HIR::Call {
                span: span.clone(),
                callee: Box::new(HIR::Literal { span: span.clone(), value: Literal::Nil }),
                args: vec![],
                is_tail,
            });
        }

        let head = list.head().unwrap();
        let args = list.tail().unwrap_or_else(List::new);

        // Check if head is a symbol that matches a special form
        if let Value::Symbol { value: sym, .. } = head {
            let sym_id = sym.id();

            if sym_id == self.special_forms.s_def {
                return self.lower_def(span, &args);
            }
            if sym_id == self.special_forms.s_defmacro {
                return self.lower_defmacro(span, &args);
            }
            if sym_id == self.special_forms.s_if {
                return self.lower_if(span, &args, is_tail);
            }
            if sym_id == self.special_forms.s_let {
                return self.lower_let(span, &args, is_tail);
            }
            if sym_id == self.special_forms.s_do {
                return self.lower_do(span, &args, is_tail);
            }
            if sym_id == self.special_forms.s_quote {
                return self.lower_quote(span, &args);
            }
            if sym_id == self.special_forms.s_fn {
                return self.lower_fn(span, &args);
            }
            if sym_id == self.special_forms.s_loop {
                return self.lower_loop(span, &args);
            }
            if sym_id == self.special_forms.s_recur {
                return self.lower_recur(span, &args);
            }
            if sym_id == self.special_forms.s_ns {
                return self.lower_ns(span, &args);
            }
        }

        // Regular function call
        let callee = self.lower_inner(head, false)?;
        let lowered_args: Result<Vec<_>, _> =
            args.iter().map(|a| self.lower_inner(a, false)).collect();

        Ok(HIR::Call { span: span.clone(), callee: Box::new(callee), args: lowered_args?, is_tail })
    }

    /// Lower a def form: (def name value?)
    fn lower_def(&self, span: &Span, args: &List<Value>) -> SpannedResult<HIR> {
        if args.is_empty() {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "def requires at least 1 argument".to_string(),
                }),
            ));
        }

        let mut iter = args.iter();
        let name_value = iter.next().unwrap();

        let (name, meta) = match name_value {
            Value::Symbol { value: sym, .. } => (sym.id(), sym.metadata()),
            other => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidSymbol {
                        value: "First argument to def must be a symbol".to_string(),
                    }),
                ));
            }
        };

        // Handle optional docstring and value
        let (docstring, value_expr) = match iter.len() {
            0 => {
                // (def x) - value is nil
                (None, None)
            }
            1 => {
                // (def x value)
                (None, iter.next())
            }
            2 => {
                // (def x "doc" value)
                let potential_doc = iter.next().unwrap();
                let val = iter.next();
                if let Value::String { value: doc, .. } = potential_doc {
                    (Some(doc.clone()), val)
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
                    span.clone(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: format!("def requires at most 3 arguments, got {}", len + 1),
                    }),
                ));
            }
        };

        let value_hir = match value_expr {
            Some(v) => self.lower_inner(v, false)?,
            None => HIR::Literal { span: span.clone(), value: Literal::Nil },
        };

        // Merge docstring into metadata if present
        let final_meta = match (meta, docstring) {
            (Some(mut m), Some(doc)) => {
                let doc_key = Value::Keyword {
                    span: span.clone(),
                    value: interner::intern_kw("doc"),
                };
                let doc_value = Value::String { span: span.clone(), value: doc };
                m.insert(doc_key, doc_value);
                Some(m)
            }
            (None, Some(doc)) => {
                let doc_key = Value::Keyword {
                    span: span.clone(),
                    value: interner::intern_kw("doc"),
                };
                let doc_value = Value::String { span: span.clone(), value: doc };
                Some(Metadata::from([(doc_key, doc_value)]))
            }
            (m, None) => m,
        };

        Ok(HIR::Def { span: span.clone(), name, value: Box::new(value_hir), meta: final_meta })
    }

    /// Lower a defmacro form: (defmacro name [params] body...)
    fn lower_defmacro(&self, span: &Span, args: &List<Value>) -> SpannedResult<HIR> {
        if args.len() < 2 {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "defmacro requires at least 2 arguments".to_string(),
                }),
            ));
        }

        let mut iter = args.iter().peekable();

        // Get macro name
        let (name, meta) = match iter.next() {
            Some(Value::Symbol { value: sym, .. }) => (sym.id(), sym.metadata()),
            Some(other) => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidSymbol {
                        value: "First argument to defmacro must be a symbol".to_string(),
                    }),
                ));
            }
            None => unreachable!(),
        };

        // Skip optional docstring
        if let Some(Value::String { .. }) = iter.peek() {
            iter.next();
        }

        // Skip optional attribute map
        if let Some(Value::Map { .. }) = iter.peek() {
            iter.next();
        }

        // Get parameters
        let params = match iter.next() {
            Some(Value::Vector { value: vec, .. }) => self.lower_params(vec)?,
            Some(other) => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidVector {
                        reason: "defmacro parameters must be a vector".to_string(),
                    }),
                ));
            }
            None => {
                return Err(error_at(
                    span.clone(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "defmacro requires a parameter vector".to_string(),
                    }),
                ));
            }
        };

        // Lower body forms - the last form is in tail position
        let body: Vec<_> = iter.cloned().collect();
        let body_hir = self.lower_body(&body)?;

        Ok(HIR::DefMacro { span: span.clone(), name, params, body: body_hir, meta })
    }

    /// Lower an if form: (if cond then else?)
    fn lower_if(&self, span: &Span, args: &List<Value>, is_tail: bool) -> SpannedResult<HIR> {
        if args.len() < 2 || args.len() > 3 {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "if requires 2 or 3 arguments".to_string(),
                }),
            ));
        }

        let mut iter = args.iter();
        let condition = iter.next().unwrap();
        let then_form = iter.next().unwrap();
        let else_form = iter.next();

        let condition_hir = self.lower_inner(condition, false)?;
        // Branches inherit tail position
        let then_hir = self.lower_inner(then_form, is_tail)?;
        let else_hir = else_form.map(|e| self.lower_inner(e, is_tail)).transpose()?;

        Ok(HIR::If {
            span: span.clone(),
            condition: Box::new(condition_hir),
            then_branch: Box::new(then_hir),
            else_branch: else_hir.map(Box::new),
            is_tail,
        })
    }

    /// Lower a let form: (let* [bindings...] body...)
    fn lower_let(&self, span: &Span, args: &List<Value>, is_tail: bool) -> SpannedResult<HIR> {
        if args.is_empty() {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "let* requires at least 1 argument".to_string(),
                }),
            ));
        }

        let bindings_form = args.head().unwrap();
        let bindings_vec = match bindings_form {
            Value::Vector { value, .. } => value.clone(),
            other => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidVector {
                        reason: "First argument to let* must be a vector".to_string(),
                    }),
                ));
            }
        };

        if bindings_vec.len() % 2 == 1 {
            return Err(error_at(
                bindings_form.span(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "let* binding vector requires an even number of forms".to_string(),
                }),
            ));
        }

        // Lower bindings
        let mut bindings = Vec::new();
        let mut iter = bindings_vec.iter();
        while let (Some(pattern_val), Some(val_expr)) = (iter.next(), iter.next()) {
            let pattern = self.lower_pattern(pattern_val)?;
            let value_hir = self.lower_inner(val_expr, false)?;
            bindings.push((pattern, value_hir));
        }

        // Lower body
        let body_forms: Vec<_> = args.tail().map(|t| t.iter().cloned().collect()).unwrap_or_default();
        let body_hir = self.lower_body_with_tail(&body_forms, is_tail)?;

        Ok(HIR::Let { span: span.clone(), bindings, body: body_hir, is_tail })
    }

    /// Lower a do form: (do forms...)
    fn lower_do(&self, span: &Span, args: &List<Value>, is_tail: bool) -> SpannedResult<HIR> {
        let forms: Vec<_> = args.iter().cloned().collect();
        let forms_hir = self.lower_body_with_tail(&forms, is_tail)?;

        Ok(HIR::Do { span: span.clone(), forms: forms_hir, is_tail })
    }

    /// Lower a quote form: (quote expr)
    fn lower_quote(&self, span: &Span, args: &List<Value>) -> SpannedResult<HIR> {
        if args.len() != 1 {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "quote requires exactly 1 argument".to_string(),
                }),
            ));
        }

        let quoted = args.head().unwrap().clone();
        Ok(HIR::Quote { span: span.clone(), value: Box::new(quoted) })
    }

    /// Lower a fn form: (fn* name? [params...] body...)
    fn lower_fn(&self, span: &Span, args: &List<Value>) -> SpannedResult<HIR> {
        if args.is_empty() {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "fn* requires at least 1 argument".to_string(),
                }),
            ));
        }

        let mut iter = args.iter().peekable();

        // Optional name
        let name = match iter.peek() {
            Some(Value::Symbol { value: sym, .. }) => {
                let id = sym.id();
                iter.next();
                Some(id)
            }
            _ => None,
        };

        // Get parameters
        let params = match iter.next() {
            Some(Value::Vector { value: vec, .. }) => self.lower_params(vec)?,
            Some(other) => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidVector {
                        reason: "fn* parameters must be a vector".to_string(),
                    }),
                ));
            }
            None => {
                return Err(error_at(
                    span.clone(),
                    Error::SyntaxError(SyntaxError::WrongArgumentCount {
                        error_str: "fn* requires a parameter vector".to_string(),
                    }),
                ));
            }
        };

        // Lower body forms - function body is in tail position
        let body_forms: Vec<_> = iter.cloned().collect();
        let body_hir = self.lower_body_with_tail(&body_forms, true)?;

        Ok(HIR::Fn { span: span.clone(), name, params, body: body_hir })
    }

    /// Lower a loop form: (loop* [bindings...] body...)
    fn lower_loop(&self, span: &Span, args: &List<Value>) -> SpannedResult<HIR> {
        if args.is_empty() {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "loop* requires at least 1 argument".to_string(),
                }),
            ));
        }

        let bindings_form = args.head().unwrap();
        let bindings_vec = match bindings_form {
            Value::Vector { value, .. } => value.clone(),
            other => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidVector {
                        reason: "First argument to loop* must be a vector".to_string(),
                    }),
                ));
            }
        };

        if bindings_vec.len() % 2 == 1 {
            return Err(error_at(
                bindings_form.span(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "loop* binding vector requires an even number of forms".to_string(),
                }),
            ));
        }

        // Lower bindings
        let mut bindings = Vec::new();
        let mut iter = bindings_vec.iter();
        while let (Some(pattern_val), Some(val_expr)) = (iter.next(), iter.next()) {
            let pattern = self.lower_pattern(pattern_val)?;
            let value_hir = self.lower_inner(val_expr, false)?;
            bindings.push((pattern, value_hir));
        }

        // Lower body - loop body is in tail position for recur
        let body_forms: Vec<_> = args.tail().map(|t| t.iter().cloned().collect()).unwrap_or_default();
        let body_hir = self.lower_body_with_tail(&body_forms, true)?;

        Ok(HIR::Loop { span: span.clone(), bindings, body: body_hir })
    }

    /// Lower a recur form: (recur args...)
    fn lower_recur(&self, span: &Span, args: &List<Value>) -> SpannedResult<HIR> {
        let args_hir: Result<Vec<_>, _> =
            args.iter().map(|a| self.lower_inner(a, false)).collect();

        Ok(HIR::Recur { span: span.clone(), args: args_hir? })
    }

    /// Lower a ns form: (ns name)
    fn lower_ns(&self, span: &Span, args: &List<Value>) -> SpannedResult<HIR> {
        if args.is_empty() {
            return Err(error_at(
                span.clone(),
                Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: "ns requires at least 1 argument".to_string(),
                }),
            ));
        }

        let name_value = args.head().unwrap();
        let name = match name_value {
            Value::Symbol { value: sym, .. } => sym.id(),
            other => {
                return Err(error_at(
                    other.span(),
                    Error::SyntaxError(SyntaxError::InvalidSymbol {
                        value: "First argument to ns must be a symbol".to_string(),
                    }),
                ));
            }
        };

        Ok(HIR::Ns { span: span.clone(), name })
    }

    /// Lower a vector literal.
    fn lower_vector(&self, span: &Span, vec: &Arc<Vector<Value>>) -> SpannedResult<HIR> {
        let items: Result<Vec<_>, _> =
            vec.iter().map(|v| self.lower_inner(v, false)).collect();

        Ok(HIR::Vector { span: span.clone(), items: items? })
    }

    /// Lower a map literal.
    fn lower_map(&self, span: &Span, map: &Arc<Map<Value, Value>>) -> SpannedResult<HIR> {
        let entries: Result<Vec<_>, _> = map
            .iter()
            .map(|(k, v)| {
                let key_hir = self.lower_inner(k, false)?;
                let val_hir = self.lower_inner(v, false)?;
                Ok((key_hir, val_hir))
            })
            .collect();

        Ok(HIR::Map { span: span.clone(), entries: entries? })
    }

    /// Lower a set literal.
    fn lower_set(
        &self,
        span: &Span,
        set: &Arc<crate::collections::Set<Value>>,
    ) -> SpannedResult<HIR> {
        let items: Result<Vec<_>, _> =
            set.iter().map(|v| self.lower_inner(v, false)).collect();

        Ok(HIR::Set { span: span.clone(), items: items? })
    }

    /// Lower a parameter vector to patterns.
    fn lower_params(&self, params: &Arc<Vector<Value>>) -> SpannedResult<Vec<Pattern>> {
        let mut patterns = Vec::new();
        let mut iter = params.iter().peekable();

        while let Some(param) = iter.next() {
            // Check for rest parameter: & rest
            if let Value::Symbol { value: sym, span, .. } = param {
                if sym.id() == self.ampersand_sym {
                    // Next must be the rest parameter name
                    let rest_param = iter.next().ok_or_else(|| {
                        error_at(
                            span.clone(),
                            Error::SyntaxError(SyntaxError::InvalidSymbol {
                                value: "& must be followed by a symbol".to_string(),
                            }),
                        )
                    })?;
                    let rest_pattern = self.lower_pattern(rest_param)?;
                    patterns.push(Pattern::Rest {
                        pattern: Box::new(rest_pattern),
                        span: span.clone(),
                    });
                    continue;
                }
            }

            patterns.push(self.lower_pattern(param)?);
        }

        Ok(patterns)
    }

    /// Lower a Value to a Pattern (for destructuring).
    fn lower_pattern(&self, value: &Value) -> SpannedResult<Pattern> {
        match value {
            Value::Symbol { value: sym, span } => {
                if sym.id() == self.underscore_sym {
                    Ok(Pattern::Ignore { span: span.clone() })
                } else {
                    Ok(Pattern::Bind { name: sym.id(), span: span.clone() })
                }
            }

            Value::Vector { value: vec, span, .. } => {
                let patterns: Result<Vec<_>, _> =
                    vec.iter().map(|v| self.lower_pattern(v)).collect();
                Ok(Pattern::Vector { patterns: patterns?, span: span.clone() })
            }

            Value::Map { value: map, span, .. } => {
                // Map destructuring expects keyword keys
                let entries: Result<Vec<_>, _> = map
                    .iter()
                    .map(|(k, v)| {
                        let key = match k {
                            Value::Keyword { value: kw, .. } => *kw,
                            other => {
                                return Err(error_at(
                                    other.span(),
                                    Error::SyntaxError(SyntaxError::InvalidMap {
                                        reason: "Map pattern keys must be keywords".to_string(),
                                    }),
                                ));
                            }
                        };
                        let pattern = self.lower_pattern(v)?;
                        Ok((key, pattern))
                    })
                    .collect();
                Ok(Pattern::Map { entries: entries?, span: span.clone() })
            }

            other => Err(error_at(
                other.span(),
                Error::SyntaxError(SyntaxError::InvalidSymbol {
                    value: format!(
                        "Invalid pattern: expected symbol, vector, or map, got {}",
                        other.as_str()
                    ),
                }),
            )),
        }
    }

    /// Lower a body of forms, with the last in tail position.
    fn lower_body(&self, forms: &[Value]) -> SpannedResult<Vec<HIR>> {
        self.lower_body_with_tail(forms, true)
    }

    /// Lower a body of forms, optionally marking the last as tail.
    fn lower_body_with_tail(&self, forms: &[Value], is_tail: bool) -> SpannedResult<Vec<HIR>> {
        if forms.is_empty() {
            return Ok(vec![]);
        }

        let mut result = Vec::with_capacity(forms.len());
        let last_idx = forms.len() - 1;

        for (i, form) in forms.iter().enumerate() {
            let form_is_tail = is_tail && i == last_idx;
            result.push(self.lower_inner(form, form_is_tail)?);
        }

        Ok(result)
    }
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::{Reader, Source};
    use crate::runtime::Runtime;

    fn parse(source: &str) -> Value {
        let runtime = Runtime::new();
        Reader::read(source, Source::REPL, runtime).unwrap()
    }

    fn lower(source: &str) -> HIR {
        let value = parse(source);
        let lowerer = Lowerer::new();
        lowerer.lower(&value).unwrap()
    }

    #[test]
    fn test_lower_literals() {
        assert!(matches!(lower("nil"), HIR::Literal { value: Literal::Nil, .. }));
        assert!(matches!(lower("true"), HIR::Literal { value: Literal::Bool(true), .. }));
        assert!(matches!(lower("false"), HIR::Literal { value: Literal::Bool(false), .. }));
        assert!(matches!(lower("42"), HIR::Literal { value: Literal::Int(42), .. }));
        assert!(matches!(lower("3.14"), HIR::Literal { value: Literal::Float(f), .. } if (f - 3.14).abs() < 0.001));
        assert!(matches!(lower("\"hello\""), HIR::Literal { value: Literal::String(_), .. }));
        assert!(matches!(lower(":keyword"), HIR::Literal { value: Literal::Keyword(_), .. }));
    }

    #[test]
    fn test_lower_symbol() {
        let hir = lower("foo");
        assert!(matches!(hir, HIR::Var { .. }));
    }

    #[test]
    fn test_lower_def() {
        let hir = lower("(def x 42)");
        match hir {
            HIR::Def { name, value, meta, .. } => {
                assert_eq!(interner::sym_to_str(name), "x");
                assert!(matches!(*value, HIR::Literal { value: Literal::Int(42), .. }));
                assert!(meta.is_none());
            }
            _ => panic!("Expected HIR::Def"),
        }
    }

    #[test]
    fn test_lower_def_with_nil() {
        let hir = lower("(def x)");
        match hir {
            HIR::Def { name, value, .. } => {
                assert_eq!(interner::sym_to_str(name), "x");
                assert!(matches!(*value, HIR::Literal { value: Literal::Nil, .. }));
            }
            _ => panic!("Expected HIR::Def"),
        }
    }

    #[test]
    fn test_lower_if() {
        let hir = lower("(if true 1 2)");
        match hir {
            HIR::If { condition, then_branch, else_branch, is_tail, .. } => {
                assert!(matches!(*condition, HIR::Literal { value: Literal::Bool(true), .. }));
                assert!(matches!(*then_branch, HIR::Literal { value: Literal::Int(1), .. }));
                assert!(matches!(*else_branch.unwrap(), HIR::Literal { value: Literal::Int(2), .. }));
                assert!(!is_tail); // Top-level is not tail
            }
            _ => panic!("Expected HIR::If"),
        }
    }

    #[test]
    fn test_lower_if_no_else() {
        let hir = lower("(if true 1)");
        match hir {
            HIR::If { else_branch, .. } => {
                assert!(else_branch.is_none());
            }
            _ => panic!("Expected HIR::If"),
        }
    }

    #[test]
    fn test_lower_let() {
        let hir = lower("(let* [x 1 y 2] x)");
        match hir {
            HIR::Let { bindings, body, .. } => {
                assert_eq!(bindings.len(), 2);
                assert!(matches!(&bindings[0].0, Pattern::Bind { name, .. } if interner::sym_to_str(*name) == "x"));
                assert!(matches!(&bindings[1].0, Pattern::Bind { name, .. } if interner::sym_to_str(*name) == "y"));
                assert_eq!(body.len(), 1);
            }
            _ => panic!("Expected HIR::Let"),
        }
    }

    #[test]
    fn test_lower_do() {
        let hir = lower("(do 1 2 3)");
        match hir {
            HIR::Do { forms, .. } => {
                assert_eq!(forms.len(), 3);
            }
            _ => panic!("Expected HIR::Do"),
        }
    }

    #[test]
    fn test_lower_quote() {
        let hir = lower("(quote (1 2 3))");
        match hir {
            HIR::Quote { value, .. } => {
                assert!(matches!(*value, Value::List { .. }));
            }
            _ => panic!("Expected HIR::Quote"),
        }
    }

    #[test]
    fn test_lower_fn() {
        let hir = lower("(fn* [x y] (+ x y))");
        match hir {
            HIR::Fn { name, params, body, .. } => {
                assert!(name.is_none());
                assert_eq!(params.len(), 2);
                assert_eq!(body.len(), 1);
                // Body should be marked as tail
                assert!(matches!(&body[0], HIR::Call { is_tail: true, .. }));
            }
            _ => panic!("Expected HIR::Fn"),
        }
    }

    #[test]
    fn test_lower_fn_with_name() {
        let hir = lower("(fn* my-fn [x] x)");
        match hir {
            HIR::Fn { name, params, .. } => {
                assert!(name.is_some());
                assert_eq!(interner::sym_to_str(name.unwrap()), "my-fn");
                assert_eq!(params.len(), 1);
            }
            _ => panic!("Expected HIR::Fn"),
        }
    }

    #[test]
    fn test_lower_fn_with_rest_param() {
        let hir = lower("(fn* [x & rest] x)");
        match hir {
            HIR::Fn { params, .. } => {
                assert_eq!(params.len(), 2);
                assert!(matches!(&params[0], Pattern::Bind { .. }));
                assert!(matches!(&params[1], Pattern::Rest { .. }));
            }
            _ => panic!("Expected HIR::Fn"),
        }
    }

    #[test]
    fn test_lower_loop() {
        let hir = lower("(loop* [x 0] (if (> x 10) x (recur (+ x 1))))");
        match hir {
            HIR::Loop { bindings, body, .. } => {
                assert_eq!(bindings.len(), 1);
                assert!(matches!(&bindings[0].0, Pattern::Bind { name, .. } if interner::sym_to_str(*name) == "x"));
                assert_eq!(body.len(), 1);
            }
            _ => panic!("Expected HIR::Loop"),
        }
    }

    #[test]
    fn test_lower_recur() {
        let hir = lower("(recur 1 2 3)");
        match hir {
            HIR::Recur { args, .. } => {
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected HIR::Recur"),
        }
    }

    #[test]
    fn test_lower_call() {
        let hir = lower("(+ 1 2)");
        match hir {
            HIR::Call { callee, args, is_tail, .. } => {
                assert!(matches!(*callee, HIR::Var { .. }));
                assert_eq!(args.len(), 2);
                assert!(!is_tail);
            }
            _ => panic!("Expected HIR::Call"),
        }
    }

    #[test]
    fn test_lower_vector() {
        let hir = lower("[1 2 3]");
        match hir {
            HIR::Vector { items, .. } => {
                assert_eq!(items.len(), 3);
            }
            _ => panic!("Expected HIR::Vector"),
        }
    }

    #[test]
    fn test_lower_map() {
        let hir = lower("{:a 1 :b 2}");
        match hir {
            HIR::Map { entries, .. } => {
                assert_eq!(entries.len(), 2);
            }
            _ => panic!("Expected HIR::Map"),
        }
    }

    #[test]
    fn test_lower_set() {
        let hir = lower("#{1 2 3}");
        match hir {
            HIR::Set { items, .. } => {
                assert_eq!(items.len(), 3);
            }
            _ => panic!("Expected HIR::Set"),
        }
    }

    #[test]
    fn test_lower_ns() {
        let hir = lower("(ns my-namespace)");
        match hir {
            HIR::Ns { name, .. } => {
                assert_eq!(interner::sym_to_str(name), "my-namespace");
            }
            _ => panic!("Expected HIR::Ns"),
        }
    }

    #[test]
    fn test_pattern_ignore() {
        // Note: The reader doesn't support standalone `_` as a symbol.
        // Test Pattern::Ignore by directly lowering a symbol with `_` name.
        let lowerer = Lowerer::new();

        // Create a symbol Value with the underscore SymId directly
        let underscore_sym = crate::core::Symbol::new("_");
        let span = Span::new(0, 1);
        let value = Value::Symbol {
            span: span.clone(),
            value: std::sync::Arc::new(underscore_sym)
        };

        let pattern = lowerer.lower_pattern(&value).unwrap();
        assert!(matches!(pattern, Pattern::Ignore { .. }));
    }

    #[test]
    fn test_pattern_vector_destructure() {
        let hir = lower("(let* [[a b] [1 2]] a)");
        match hir {
            HIR::Let { bindings, .. } => {
                match &bindings[0].0 {
                    Pattern::Vector { patterns, .. } => {
                        assert_eq!(patterns.len(), 2);
                    }
                    _ => panic!("Expected Pattern::Vector"),
                }
            }
            _ => panic!("Expected HIR::Let"),
        }
    }

    #[test]
    fn test_tail_position_in_if() {
        let hir = lower("(fn* [] (if true 1 2))");
        match hir {
            HIR::Fn { body, .. } => {
                match &body[0] {
                    HIR::If { is_tail, .. } => {
                        assert!(*is_tail, "if should be in tail position");
                    }
                    _ => panic!("Expected HIR::If"),
                }
            }
            _ => panic!("Expected HIR::Fn"),
        }
    }

    #[test]
    fn test_tail_position_in_let() {
        let hir = lower("(fn* [] (let* [x 1] (+ x 1)))");
        match hir {
            HIR::Fn { body, .. } => {
                match &body[0] {
                    HIR::Let { body: let_body, is_tail, .. } => {
                        assert!(*is_tail, "let should be in tail position");
                        assert!(matches!(&let_body[0], HIR::Call { is_tail: true, .. }));
                    }
                    _ => panic!("Expected HIR::Let"),
                }
            }
            _ => panic!("Expected HIR::Fn"),
        }
    }

    #[test]
    fn test_nested_tail_position() {
        let hir = lower("(fn* [] (do (println \"hi\") (if true (+ 1 2) (- 1 2))))");
        match hir {
            HIR::Fn { body, .. } => {
                match &body[0] {
                    HIR::Do { forms, is_tail, .. } => {
                        assert!(*is_tail);
                        // First form is not tail
                        assert!(matches!(&forms[0], HIR::Call { is_tail: false, .. }));
                        // Second form (if) is tail
                        match &forms[1] {
                            HIR::If { then_branch, else_branch, is_tail, .. } => {
                                assert!(*is_tail);
                                assert!(matches!(**then_branch, HIR::Call { is_tail: true, .. }));
                                assert!(matches!(*else_branch.as_ref().unwrap().as_ref(), HIR::Call { is_tail: true, .. }));
                            }
                            _ => panic!("Expected HIR::If"),
                        }
                    }
                    _ => panic!("Expected HIR::Do"),
                }
            }
            _ => panic!("Expected HIR::Fn"),
        }
    }

    #[test]
    fn test_bound_names() {
        let pattern = Pattern::Vector {
            patterns: vec![
                Pattern::Bind { name: interner::intern_sym("a"), span: Span::new(0, 1) },
                Pattern::Ignore { span: Span::new(2, 3) },
                Pattern::Bind { name: interner::intern_sym("b"), span: Span::new(4, 5) },
            ],
            span: Span::new(0, 5),
        };

        let names = pattern.bound_names();
        assert_eq!(names.len(), 2);
        assert_eq!(interner::sym_to_str(names[0]), "a");
        assert_eq!(interner::sym_to_str(names[1]), "b");
    }
}
