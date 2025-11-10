use std::sync::Arc;

use crate::collections::Vector;
use crate::value::Value;

#[derive(Debug, Clone)]
pub enum RecurContext {
    Loop {
        bindings: Arc<Vector<Value>>, // Symbols for loop bindings
    },
    Function {
        params: Arc<Vector<Value>>, // Function parameters vector
    },
}
