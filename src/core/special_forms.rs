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
        }
    }
}
