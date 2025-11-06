use rustc_hash::FxHashMap;

use crate::value::Value;

pub struct Env {
  pub parent: Option<Box<Env>>,
  pub bindings: FxHashMap<String, Value>,
}

impl Env {
  pub fn new() -> Self {
    Self { parent: None, bindings: FxHashMap::default() }
  }

  pub fn get(&self, key: &str) -> Option<&Value> {
    self.bindings.get(key)
  }

  pub fn set(&mut self, key: &str, value: Value) {
    self.bindings.insert(key.to_string(), value);
  }
}
