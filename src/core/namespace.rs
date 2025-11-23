use std::sync::Arc;
use std::sync::Mutex;

use crate::collections::Map;
use crate::core::Var;
use crate::interner::{self, NsId, SymId};

#[derive(Debug, Clone)]
pub struct Namespace {
    pub id: NsId,
    pub name: String,
    // Symbol bindings with references to the variables
    pub bindings: Map<SymId, Arc<Var>>,
    // Table of imported libraries aliases, declared via `require`
    pub aliases: Map<SymId, SymId>,
}

impl Namespace {
    pub fn new(id: NsId, name: String) -> Self {
        Self { id, name, bindings: Map::new(), aliases: Map::new() }
    }

    pub fn insert(&self, sym: SymId, var: Arc<Var>) -> Self {
        Self { bindings: self.bindings.insert(sym, var), ..self.clone() }
    }

    pub fn get(&self, sym: SymId) -> Option<&Arc<Var>> {
        self.bindings.get(&sym)
    }

    pub fn id(&self) -> NsId {
        self.id
    }
}

impl PartialEq for Namespace {
    fn eq(&self, other: &Self) -> bool {
        // Two namespaces are equal if they have the same ID
        self.id == other.id
    }
}

#[derive(Debug)]
pub struct NamespaceRegistry {
    registry: Mutex<Map<NsId, Arc<Namespace>>>,
    current: Mutex<NsId>,
}

impl NamespaceRegistry {
    pub fn new() -> Self {
        Self { registry: Mutex::new(Map::new()), current: Mutex::new(NsId(0)) }
    }

    /// Looks up a namespace by its ID.
    pub fn lookup(&self, id: NsId) -> Option<Arc<Namespace>> {
        self.registry.lock().unwrap().get(&id).cloned()
    }

    /// Inserts a namespace by its ID.
    pub fn register(&self, id: NsId, ns: Arc<Namespace>) {
        self.registry.lock().unwrap().insert(id, ns);
    }

    /// Removes a namespace by its ID.
    pub fn unregister(&self, id: NsId) {
        self.registry.lock().unwrap().remove(&id);
    }

    /// Finds a namespace by its name or creates a new one if it doesn't exist.
    pub fn find_or_create(&self, ns_name: &str) -> Arc<Namespace> {
        let id = interner::intern_ns(ns_name);

        if let Some(ns) = self.registry.lock().unwrap().get(&id).cloned() {
            return ns;
        }

        // Lock is released, so we can safely create and register
        // Note: There's a small race condition here if two threads create the same
        // namespace simultaneously, but the second one will just overwrite the first,
        // which is acceptable since namespaces are immutable.
        let ns_arc = Arc::new(Namespace::new(id, ns_name.to_string()));
        let mut registry = self.registry.lock().unwrap();
        let new_map = registry.clone().insert(id, ns_arc.clone());
        *registry = new_map;
        ns_arc
    }

    /// Sets the current namespace.
    pub fn set_current(&self, id: NsId) {
        *self.current.lock().unwrap() = id;
    }

    /// Gets the current namespace.
    pub fn get_current(&self) -> Arc<Namespace> {
        let id = *self.current.lock().unwrap();
        self.lookup(id).unwrap()
    }
}
