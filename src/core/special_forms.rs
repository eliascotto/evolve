use crate::interner::{self, SymId};

#[derive(Debug, Clone)]
pub struct SpecialFormRegistry {
    pub s_def: SymId,
    pub s_defmacro: SymId,
    pub s_if: SymId,
    pub s_let: SymId,
    pub s_do: SymId,
    pub s_quote: SymId,
    pub s_fn: SymId,
    pub s_loop: SymId,
    pub s_recur: SymId,
    pub s_ns: SymId,
    // Exception handling
    pub s_try: SymId,
    pub s_catch: SymId,
    pub s_finally: SymId,
    pub s_throw: SymId,
    // STM
    pub s_dosync: SymId,
    // Condition system
    pub s_handler_bind: SymId,
    pub s_handler_case: SymId,
    pub s_restart_case: SymId,
    pub s_signal: SymId,
    pub s_error: SymId,
    pub s_invoke_restart: SymId,
}

impl SpecialFormRegistry {
    pub fn new() -> Self {
        Self {
            s_def: interner::intern_sym("def"),
            s_defmacro: interner::intern_sym("defmacro"),
            s_if: interner::intern_sym("if"),
            s_let: interner::intern_sym("let*"),
            s_do: interner::intern_sym("do"),
            s_quote: interner::intern_sym("quote"),
            s_fn: interner::intern_sym("fn*"),
            s_loop: interner::intern_sym("loop*"),
            s_recur: interner::intern_sym("recur"),
            s_ns: interner::intern_sym("ns"),
            // Exception handling
            s_try: interner::intern_sym("try"),
            s_catch: interner::intern_sym("catch"),
            s_finally: interner::intern_sym("finally"),
            s_throw: interner::intern_sym("throw"),
            // STM
            s_dosync: interner::intern_sym("dosync"),
            // Condition system
            s_handler_bind: interner::intern_sym("handler-bind"),
            s_handler_case: interner::intern_sym("handler-case"),
            s_restart_case: interner::intern_sym("restart-case"),
            s_signal: interner::intern_sym("signal"),
            s_error: interner::intern_sym("error"),
            s_invoke_restart: interner::intern_sym("invoke-restart"),
        }
    }
}
