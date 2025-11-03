use std::collections::HashMap;

use crate::value::Value;

const NAMESPACE_SEPARATOR: char = '/';

pub struct Symbol {
    pub name: String,
    pub ns: Option<String>,
    pub meta: HashMap<Value, Value>,
}

impl Symbol {
    pub fn new(name: String) -> Self {
        let mut ns = None;
        if name.contains(NAMESPACE_SEPARATOR) {
            ns = Some(name.split(NAMESPACE_SEPARATOR).next().unwrap().to_string());
        }
        Self { name, ns, meta: HashMap::new() }
    }
}
