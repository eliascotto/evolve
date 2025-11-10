use std::sync::Arc;
use std::sync::Mutex;

use crate::collections::Map;
use crate::interner;
use crate::interner::{NsId, SymId};
use crate::core::Var;
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct Namespace {
    pub id: NsId,
    // Symbol bindings with references to the variables
    pub bindings: Map<SymId, Arc<Var>>,
    // Table of imported libraries aliases, declared via `require`
    pub aliases: Map<SymId, SymId>,
}

impl Namespace {
    pub fn new(id: NsId) -> Self {
        Self { id, bindings: Map::new(), aliases: Map::new() }
    }

    pub fn insert(&self, sym: SymId, var: Arc<Var>) -> Self {
        Self { bindings: self.bindings.insert(sym, var), ..self.clone() }
    }

    pub fn get(&self, sym: SymId) -> Option<&Arc<Var>> {
        self.bindings.get(&sym)
    }
}

static NS_REGISTRY: Lazy<Mutex<Map<NsId, Arc<Namespace>>>> =
    Lazy::new(|| Mutex::new(Map::new()));

/// Looks up a namespace by its ID
pub fn ns_lookup(id: NsId) -> Option<Arc<Namespace>> {
    NS_REGISTRY.lock().unwrap().get(&id).cloned()
}

/// Registers a namespace by its ID
fn ns_register(ns: Arc<Namespace>) -> NsId {
    let id = ns.id;
    NS_REGISTRY.lock().unwrap().insert(id, ns);
    id
}

/// Unregisters a namespace by its ID
pub fn ns_unregister(id: NsId) {
    NS_REGISTRY.lock().unwrap().remove(&id);
}

/// Finds a namespace by its name or creates a new one if it doesn't exist
pub fn ns_find_or_create(ns_name: &'static str) -> Arc<Namespace> {
    let ns_id = interner::intern_ns(ns_name);
    // First, try to get existing namespace (lock is released after this block)
    if let Some(ns) = NS_REGISTRY.lock().unwrap().get(&ns_id).cloned() {
        return ns;
    }
    // Lock is released, so we can safely create and register
    // Note: There's a small race condition here if two threads create the same
    // namespace simultaneously, but the second one will just overwrite the first,
    // which is acceptable since namespaces are immutable.
    let ns = Namespace::new(ns_id);
    let ns_arc = Arc::new(ns);
    ns_register(ns_arc.clone());
    ns_arc
}
