//! Environment for variable bindings with lexical scoping support.
//!
//! # Design
//!
//! Environments are **immutable** - `set` returns a new `Env`. This enables thread-safe sharing
//! via `Arc`, safe closure capture, and preserved state for debugging.
//!
//! # Performance
//!
//! Uses `Map<SymId, Value>` (HAMT-based persistent map) instead of `FxHashMap` for efficient
//! immutable updates:
//! - **FxHashMap `set` operation**: O(n) - must clone entire map, copying all entries
//! - **Persistent Map `set` operation**: O(log n) - uses structural sharing, only new nodes created
//! - **Trade-off**: Persistent Map `get` operation is O(log n) vs FxHashMap's O(1), but acceptable
//!   for typical sizes (< 1000 bindings)
//!
//! For runtime evaluation with `set!` (mutation), consider mutable environments using
//! `Rc<RefCell<FxHashMap>>` or `Arc<Mutex<FxHashMap>>`.

use std::sync::Arc;

use crate::collections::{Map, Vector};
use crate::core::namespace::update_ns;
use crate::core::Var;
use crate::core::{Namespace, RecurContext};
use crate::interner::SymId;
use crate::value::Value;

/// Environment for variable bindings with lexical scoping support.
#[derive(Debug, Clone)]
pub struct Env {
    // Namespace binding
    pub ns: Arc<Namespace>,
    // Parent environment
    pub parent: Option<Arc<Env>>,
    // Variable bindings
    pub bindings: Map<SymId, Value>,

    //--------------------------------------------------//
    // Used by recur to know what context it's in
    // and what arguments to pass to the recursive call.
    recur_context: Option<RecurContext>,
}

impl Env {
    pub fn new(ns: Arc<Namespace>) -> Self {
        Self { ns, parent: None, bindings: Map::new(), recur_context: None }
    }

    /// Returns the value if found, otherwise looks in the parent environment.
    pub fn get(&self, key: SymId) -> Option<&Value> {
        self.bindings
            .get(&key)
            .or_else(|| self.parent.as_ref().and_then(|p| p.get(key)))
    }

    /// Returns a new environment with the key bound to the value.
    pub fn set(&self, key: SymId, value: Value) -> Self {
        Self {
            ns: self.ns.clone(),
            bindings: self.bindings.insert(key, value),
            parent: self.parent.clone(),
            recur_context: self.recur_context.clone(),
        }
    }

    /// Defines a variable in the namespace by registering a symbol-to-Var binding.
    /// Returns a new environment with the updated namespace bindings.
    /// Also persists the namespace change to the global registry.
    pub fn define_var(&self, sym: SymId, var: Arc<Var>) -> Self {
        let new_ns = Arc::new(self.ns.insert(sym, var));
        // Persist the namespace change to the global registry
        // so that other environments using this namespace can see the new binding
        update_ns(new_ns.clone());
        Self {
            ns: new_ns,
            bindings: self.bindings.clone(),
            parent: self.parent.clone(),
            recur_context: self.recur_context.clone(),
        }
    }

    /// Creates a new environment with the same namespace and bindings, but a new parent environment.
    pub fn create_child(&self) -> Self {
        Self {
            ns: self.ns.clone(),
            bindings: self.bindings.clone(),
            parent: Some(Arc::new(self.clone())),
            recur_context: self.recur_context.clone(),
        }
    }

    /// Creates a new environment with the same namespace, but replaces the local
    /// bindings with the supplied vector while keeping the parent pointer.
    ///
    /// The trampoline evaluator relies on this helper to materialise lexical
    /// scopes without mutating the parent `Env` instance, allowing tail-position
    /// evaluation to swap environments cheaply.
    ///
    /// `bindings` should be a vector of pairs: [sym1 val1 sym2 val2 ...]
    /// where sym1, sym2, etc. are symbols and val1, val2, etc. are values.
    pub fn create_child_with_bindings(&self, bindings: Arc<Vector<Value>>) -> Self {
        let mut new_bindings = Map::new();

        // Process bindings in pairs: [sym1 val1 sym2 val2 ...]
        let mut iter = bindings.iter();
        while let (Some(param), Some(arg)) = (iter.next(), iter.next()) {
            match param {
                Value::Symbol { value, .. } => {
                    new_bindings = new_bindings.insert(value.id(), arg.clone());
                }
                _ => {
                    // Invalid binding - param must be a symbol
                    // This should be caught earlier, but handle gracefully
                    continue;
                }
            }
        }

        Self {
            ns: self.ns.clone(),
            bindings: new_bindings,
            parent: Some(Arc::new(self.clone())),
            recur_context: self.recur_context.clone(),
        }
    }

    /// Creates a new environment setting the recursion context.
    pub fn with_recur_context(mut self, context: RecurContext) -> Self {
        self.recur_context = Some(context);
        self
    }

    /// Returns the recursion context if set, otherwise None.
    pub fn get_recur_context(&self) -> Option<&RecurContext> {
        self.recur_context.as_ref()
    }
}
