use rustc_hash::FxHashMap;

use crate::interner::SymId;
use crate::value::Value;

#[derive(Debug, Clone)]
pub struct Env {
    pub parent: Option<Box<Env>>,
    pub bindings: FxHashMap<SymId, Value>,
}

impl Env {
    pub fn new() -> Self {
        Self { parent: None, bindings: FxHashMap::default() }
    }

    pub fn get(&self, key: SymId) -> Option<&Value> {
        self.bindings
            .get(&key)
            .or_else(|| self.parent.as_ref().and_then(|p| p.get(key)))
    }

    pub fn set(&mut self, key: SymId, value: Value) {
        self.bindings.insert(key, value);
    }

    pub fn find(&self, key: SymId) -> Option<&Value> {
        if self.bindings.contains_key(&key) {
            Some(self.bindings.get(&key).unwrap())
        } else {
            self.parent.as_ref().and_then(|p| p.find(key))
        }
    }
}
