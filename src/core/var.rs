use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::interner::{KeywId, SymId};
use crate::core::Namespace;
use crate::value::Value;

#[derive(Debug)]
pub struct Var {
    // Symbol identifier
    pub symbol: SymId,
    // Namespace binding
    pub ns: Arc<Namespace>,
    // Value binding
    pub value: Option<Arc<RwLock<Value>>>,

    pub meta: Option<BTreeMap<KeywId, Value>>,
    // Dynamic flag
    dynamic: AtomicBool,
    // Revision counter
    rev: AtomicU64,
}

impl Var {
    pub fn new(symbol: SymId, ns: Arc<Namespace>) -> Self {
        Self {
            symbol,
            ns,
            value: None,
            meta: None,
            dynamic: AtomicBool::new(false),
            rev: AtomicU64::new(0),
        }
    }

    pub fn new_with_value(
        symbol: SymId,
        ns: Arc<Namespace>,
        value: Value,
        meta: Option<BTreeMap<KeywId, Value>>,
    ) -> Self {
        Self {
            symbol,
            ns,
            value: Some(Arc::new(RwLock::new(value))),
            meta: meta,
            dynamic: AtomicBool::new(false),
            rev: AtomicU64::new(0),
        }
    }

    pub fn is_bound(&self) -> bool {
        self.value.is_some()
    }

    pub fn set_meta(&self, meta: Option<BTreeMap<KeywId, Value>>) -> Self {
        Self { meta: meta, ..self.clone() }
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        Self {
            symbol: self.symbol,
            ns: self.ns.clone(),
            value: self.value.clone(),
            meta: self.meta.clone(),
            dynamic: AtomicBool::new(self.dynamic.load(Ordering::Relaxed)),
            rev: AtomicU64::new(self.rev.load(Ordering::Relaxed)),
        }
    }
}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.symbol.hash(state);
        self.ns.id.hash(state);
        self.dynamic.load(Ordering::Relaxed).hash(state);
        self.rev.load(Ordering::Relaxed).hash(state);
    }
}
