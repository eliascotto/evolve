//! Condition System for restartable exceptions.
//!
//! This implements a Common Lisp-style condition system with conditions,
//! handlers, and restarts. Conditions are more flexible than traditional
//! exceptions because they separate the decision of *what* to do (handler)
//! from *how* to do it (restart).

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::collections::Vector;
use crate::interner::{self, SymId};
use crate::value::Value;

thread_local! {
    static HANDLER_STACK: RefCell<Vec<HandlerFrame>> = const { RefCell::new(Vec::new()) };
    static RESTART_STACK: RefCell<Vec<RestartFrame>> = const { RefCell::new(Vec::new()) };
}

/// A condition (restartable exception).
#[derive(Debug, Clone)]
pub struct Condition {
    name: SymId,
    data: Arc<Vector<Value>>,
}

impl Condition {
    /// Create a new condition with the given name and data.
    pub fn new(name: SymId, data: Vec<Value>) -> Self {
        Self { name, data: Arc::new(Vector::from_iter(data)) }
    }

    /// Get the condition name.
    pub fn name(&self) -> String {
        interner::sym_to_str(self.name)
    }

    /// Get the condition name as SymId.
    pub fn name_sym(&self) -> SymId {
        self.name
    }

    /// Get the condition data.
    pub fn data(&self) -> &Vector<Value> {
        &self.data
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#<Condition {}>", self.name())
    }
}

/// A restart point in the condition system.
#[derive(Clone)]
pub struct Restart {
    name: SymId,
    handler: Arc<dyn Fn(&[Value]) -> Value + Send + Sync>,
}

impl Restart {
    /// Create a new restart with the given name and handler.
    pub fn new<F>(name: SymId, handler: F) -> Self
    where
        F: Fn(&[Value]) -> Value + Send + Sync + 'static,
    {
        Self { name, handler: Arc::new(handler) }
    }

    /// Get the restart name.
    pub fn name(&self) -> String {
        interner::sym_to_str(self.name)
    }

    /// Get the restart name as SymId.
    pub fn name_sym(&self) -> SymId {
        self.name
    }

    /// Invoke the restart with the given arguments.
    pub fn invoke(&self, args: &[Value]) -> Value {
        (self.handler)(args)
    }
}

impl fmt::Debug for Restart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Restart").field("name", &self.name()).finish()
    }
}

impl fmt::Display for Restart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#<Restart {}>", self.name())
    }
}

/// A handler binding for a condition type.
#[derive(Clone)]
pub struct Handler {
    condition_type: SymId,
    handler_fn: Arc<dyn Fn(&Condition) -> HandlerResult + Send + Sync>,
}

impl Handler {
    /// Create a new handler for the given condition type.
    pub fn new<F>(condition_type: SymId, handler_fn: F) -> Self
    where
        F: Fn(&Condition) -> HandlerResult + Send + Sync + 'static,
    {
        Self { condition_type, handler_fn: Arc::new(handler_fn) }
    }

    /// Check if this handler matches the given condition.
    pub fn matches(&self, condition: &Condition) -> bool {
        self.condition_type == condition.name
    }

    /// Call the handler with the given condition.
    pub fn call(&self, condition: &Condition) -> HandlerResult {
        (self.handler_fn)(condition)
    }
}

impl fmt::Debug for Handler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handler")
            .field("condition_type", &interner::sym_to_str(self.condition_type))
            .finish()
    }
}

/// The result of calling a handler.
#[derive(Debug, Clone)]
pub enum HandlerResult {
    /// The handler handled the condition and returned a value.
    Handled(Value),
    /// The handler declined to handle the condition.
    Decline,
    /// The handler wants to invoke a restart.
    InvokeRestart(SymId, Vec<Value>),
}

/// A frame on the handler stack.
#[derive(Debug)]
struct HandlerFrame {
    handlers: Vec<Handler>,
}

impl HandlerFrame {
    fn new(handlers: Vec<Handler>) -> Self {
        Self { handlers }
    }
}

/// A frame on the restart stack.
#[derive(Debug)]
struct RestartFrame {
    restarts: HashMap<SymId, Arc<Restart>>,
}

impl RestartFrame {
    fn new(restarts: Vec<Arc<Restart>>) -> Self {
        let mut map = HashMap::new();
        for restart in restarts {
            map.insert(restart.name, restart);
        }
        Self { restarts: map }
    }
}

/// Condition system errors.
#[derive(Debug, Clone)]
pub enum ConditionError {
    /// An unhandled condition was signaled.
    Unhandled(Arc<Condition>),
    /// The requested restart was not found.
    RestartNotFound(String),
}

impl fmt::Display for ConditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConditionError::Unhandled(c) => {
                write!(f, "Unhandled condition: {}", c.name())
            }
            ConditionError::RestartNotFound(name) => {
                write!(f, "Restart not found: {}", name)
            }
        }
    }
}

/// Push a handler frame onto the stack.
pub fn push_handlers(handlers: Vec<Handler>) {
    HANDLER_STACK.with(|stack| {
        stack.borrow_mut().push(HandlerFrame::new(handlers));
    });
}

/// Pop a handler frame from the stack.
pub fn pop_handlers() {
    HANDLER_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

/// Push a restart frame onto the stack.
pub fn push_restarts(restarts: Vec<Arc<Restart>>) {
    RESTART_STACK.with(|stack| {
        stack.borrow_mut().push(RestartFrame::new(restarts));
    });
}

/// Pop a restart frame from the stack.
pub fn pop_restarts() {
    RESTART_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

/// Signal a condition.
///
/// This searches the handler stack for a matching handler. If found, the handler
/// is called and may handle the condition, decline, or invoke a restart.
///
/// If no handler handles the condition, returns `Ok(None)`.
pub fn signal(condition: &Condition) -> Result<Option<Value>, ConditionError> {
    HANDLER_STACK.with(|stack| {
        let stack = stack.borrow();
        // Search handlers from most recent to oldest
        for frame in stack.iter().rev() {
            for handler in &frame.handlers {
                if handler.matches(condition) {
                    match handler.call(condition) {
                        HandlerResult::Handled(value) => return Ok(Some(value)),
                        HandlerResult::Decline => continue,
                        HandlerResult::InvokeRestart(restart_name, args) => {
                            drop(stack);
                            return invoke_restart_internal(restart_name, &args);
                        }
                    }
                }
            }
        }
        Ok(None)
    })
}

/// Signal an error condition.
///
/// Like `signal`, but if no handler handles the condition, returns an error
/// instead of None.
pub fn error(condition: &Condition) -> Result<Value, ConditionError> {
    match signal(condition)? {
        Some(value) => Ok(value),
        None => Err(ConditionError::Unhandled(Arc::new(condition.clone()))),
    }
}

/// Find a restart by name.
pub fn find_restart(name: SymId) -> Option<Arc<Restart>> {
    RESTART_STACK.with(|stack| {
        let stack = stack.borrow();
        for frame in stack.iter().rev() {
            if let Some(restart) = frame.restarts.get(&name) {
                return Some(restart.clone());
            }
        }
        None
    })
}

/// Invoke a restart by name.
pub fn invoke_restart(name: SymId, args: &[Value]) -> Result<Value, ConditionError> {
    match invoke_restart_internal(name, args)? {
        Some(value) => Ok(value),
        None => Err(ConditionError::RestartNotFound(interner::sym_to_str(name))),
    }
}

fn invoke_restart_internal(name: SymId, args: &[Value]) -> Result<Option<Value>, ConditionError> {
    if let Some(restart) = find_restart(name) {
        Ok(Some(restart.invoke(args)))
    } else {
        Err(ConditionError::RestartNotFound(interner::sym_to_str(name)))
    }
}

/// Get all available restarts.
pub fn available_restarts() -> Vec<Arc<Restart>> {
    RESTART_STACK.with(|stack| {
        let stack = stack.borrow();
        let mut restarts = Vec::new();
        for frame in stack.iter().rev() {
            for (_, restart) in &frame.restarts {
                restarts.push(restart.clone());
            }
        }
        restarts
    })
}

/// Execute a function with handlers bound.
pub fn with_handlers<F, T>(handlers: Vec<Handler>, f: F) -> T
where
    F: FnOnce() -> T,
{
    push_handlers(handlers);
    let result = f();
    pop_handlers();
    result
}

/// Execute a function with restarts available.
pub fn with_restarts<F, T>(restarts: Vec<Arc<Restart>>, f: F) -> T
where
    F: FnOnce() -> T,
{
    push_restarts(restarts);
    let result = f();
    pop_restarts();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic_span;

    fn int_val(n: i64) -> Value {
        Value::Int { span: synthetic_span!(), value: n }
    }

    fn nil_val() -> Value {
        Value::Nil { span: synthetic_span!() }
    }

    #[test]
    fn test_condition_new() {
        let name = interner::intern_sym("test-error");
        let condition = Condition::new(name, vec![int_val(42)]);
        assert_eq!(condition.name(), "test-error");
        assert_eq!(condition.data().len(), 1);
    }

    #[test]
    fn test_signal_no_handler() {
        let name = interner::intern_sym("test-error");
        let condition = Condition::new(name, vec![]);
        let result = signal(&condition).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_signal_with_handler() {
        let name = interner::intern_sym("test-error");
        let condition = Condition::new(name, vec![]);

        let handler = Handler::new(name, |_| HandlerResult::Handled(int_val(99)));

        let result = with_handlers(vec![handler], || signal(&condition));
        assert!(result.is_ok());
        match result.unwrap() {
            Some(Value::Int { value, .. }) => assert_eq!(value, 99),
            _ => panic!("Expected Int(99)"),
        }
    }

    #[test]
    fn test_error_unhandled() {
        let name = interner::intern_sym("test-error");
        let condition = Condition::new(name, vec![]);
        let result = error(&condition);
        assert!(matches!(result, Err(ConditionError::Unhandled(_))));
    }

    #[test]
    fn test_restart() {
        let restart_name = interner::intern_sym("use-value");
        let restart =
            Arc::new(Restart::new(restart_name, |args| args.first().cloned().unwrap_or(nil_val())));

        let result = with_restarts(vec![restart], || invoke_restart(restart_name, &[int_val(42)]));
        assert!(result.is_ok());
        match result.unwrap() {
            Value::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int(42)"),
        }
    }

    #[test]
    fn test_handler_invokes_restart() {
        let error_name = interner::intern_sym("divide-error");
        let restart_name = interner::intern_sym("return-zero");

        let restart = Arc::new(Restart::new(restart_name, |_| int_val(0)));

        let handler =
            Handler::new(error_name, move |_| HandlerResult::InvokeRestart(restart_name, vec![]));

        let condition = Condition::new(error_name, vec![]);

        let result = with_restarts(vec![restart], || {
            with_handlers(vec![handler], || signal(&condition))
        });

        assert!(result.is_ok());
        match result.unwrap() {
            Some(Value::Int { value, .. }) => assert_eq!(value, 0),
            _ => panic!("Expected Int(0)"),
        }
    }
}
