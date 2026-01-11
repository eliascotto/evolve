use once_cell::sync::Lazy;
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
    // Table of imported libraries aliases (alias symbol -> namespace ID)
    pub aliases: Map<SymId, NsId>,
    // Table of referred symbols (local symbol -> foreign Var)
    pub refers: Map<SymId, Arc<Var>>,
}

impl Namespace {
    pub fn new(id: NsId, name: String) -> Self {
        Self {
            id,
            name,
            bindings: Map::new(),
            aliases: Map::new(),
            refers: Map::new(),
        }
    }

    pub fn insert(&self, sym: SymId, var: Arc<Var>) -> Self {
        Self { bindings: self.bindings.insert(sym, var), ..self.clone() }
    }

    pub fn get(&self, sym: SymId) -> Option<&Arc<Var>> {
        // First check local bindings, then referred symbols
        self.bindings.get(&sym).or_else(|| self.refers.get(&sym))
    }

    pub fn id(&self) -> NsId {
        self.id
    }

    /// Adds an alias mapping (e.g., :as fb -> foo.bar namespace)
    pub fn add_alias(&self, alias: SymId, ns_id: NsId) -> Self {
        Self { aliases: self.aliases.insert(alias, ns_id), ..self.clone() }
    }

    /// Gets the namespace ID for an alias
    pub fn get_alias(&self, alias: SymId) -> Option<&NsId> {
        self.aliases.get(&alias)
    }

    /// Adds a referred symbol (e.g., :refer [foo bar])
    pub fn add_refer(&self, sym: SymId, var: Arc<Var>) -> Self {
        Self { refers: self.refers.insert(sym, var), ..self.clone() }
    }

    /// Gets a referred var by symbol
    pub fn get_refer(&self, sym: SymId) -> Option<&Arc<Var>> {
        self.refers.get(&sym)
    }

    /// Returns all public bindings in this namespace
    pub fn public_bindings(&self) -> impl Iterator<Item = (&SymId, &Arc<Var>)> {
        self.bindings.iter().filter(|(_, var)| var.is_public())
    }

    /// Returns all bindings (public and private) in this namespace
    pub fn all_bindings(&self) -> impl Iterator<Item = (&SymId, &Arc<Var>)> {
        self.bindings.iter()
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

    /// Updates a namespace in the registry.
    pub fn update(&self, ns: Arc<Namespace>) {
        let mut registry = self.registry.lock().unwrap();
        let new_map = registry.clone().insert(ns.id, ns);
        *registry = new_map;
    }
}

// Global namespace registry (similar to interner pattern)
static NS_REGISTRY: Lazy<NamespaceRegistry> = Lazy::new(|| {
    let registry = NamespaceRegistry::new();
    // Create and register the default "user" namespace
    let user_ns = registry.find_or_create("user");
    registry.set_current(user_ns.id);
    registry
});

/// Finds a namespace by its name or creates a new one if it doesn't exist.
pub fn find_or_create_ns(ns_name: &str) -> Arc<Namespace> {
    NS_REGISTRY.find_or_create(ns_name)
}

/// Looks up a namespace by its ID.
pub fn lookup_ns(id: NsId) -> Option<Arc<Namespace>> {
    NS_REGISTRY.lookup(id)
}

/// Updates a namespace in the global registry.
pub fn update_ns(ns: Arc<Namespace>) {
    NS_REGISTRY.update(ns);
}

/// Sets the current namespace.
pub fn set_current_ns(id: NsId) {
    NS_REGISTRY.set_current(id);
}

/// Gets the current namespace.
pub fn get_current_ns() -> Arc<Namespace> {
    NS_REGISTRY.get_current()
}
