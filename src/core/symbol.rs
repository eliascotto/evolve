use crate::core::Metadata;
use crate::interner::{self, NsId, SymId};

#[derive(Debug, Clone)]
pub struct Symbol {
    /// Identifier of the symbol (its interned name).
    pub id: SymId,
    /// Optional namespace associated with the symbol.
    pub ns: Option<NsId>,
    /// The name of the symbol.
    pub name: Option<String>,
    /// Optional metadata attached to the symbol.
    pub meta: Option<Metadata>,
}

impl Symbol {
    pub fn new(symbol_str: &str) -> Self {
        // Extract the namespace from the symbol string, and create a ns symbol
        let mut ns = None;

        if let Some((before, _)) = symbol_str.split_once('/') {
            if !before.is_empty() {
                ns = Some(interner::intern_ns(before));
            }
        }

        let sym_id = interner::intern_sym(symbol_str);

        Self { id: sym_id, ns, name: Some(symbol_str.to_string()), meta: None }
    }

    pub fn new_with_id(id: SymId, ns: Option<NsId>, meta: Option<Metadata>) -> Self {
        Self { id, ns, name: None, meta }
    }

    pub fn id(&self) -> SymId {
        self.id
    }

    pub fn name(&self) -> String {
        match &self.name {
            Some(name) => name.clone(),
            None => interner::sym_to_str(self.id),
        }
    }

    pub fn namespace(&self) -> String {
        interner::ns_to_str(self.ns.unwrap())
    }

    pub fn set_namespace(&mut self, ns: NsId) {
        self.ns = Some(ns);
    }

    pub fn is_qualified(&self) -> bool {
        self.ns.is_some()
    }

    pub fn metadata(&self) -> Option<Metadata> {
        self.meta.clone()
    }

    pub fn with_meta(&self, meta: Option<Metadata>) -> Self {
        Self { meta, ..self.clone() }
    }
}
