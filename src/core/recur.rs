use std::sync::Arc;

use crate::collections::{List, Vector};
use crate::value::Value;

#[derive(Debug, Clone)]
pub enum RecurContext {
    Loop {
        bindings: Arc<Vector<Value>>, // Symbols for loop bindings
        body: Arc<List<Value>>,        // Body to re-evaluate on recur
    },
    Function {
        params: Arc<Vector<Value>>, // Function parameters vector
        body: Arc<List<Value>>,      // Function body for recur
    },
}
