//! Thread-safe mutable references with atomic compare-and-swap updates.
//!
//! Atoms provide a way to manage shared, synchronous, independent state.
//! They are thread-safe and can be used from multiple threads concurrently.

use std::fmt;
use std::sync::RwLock;

use crate::value::Value;

/// An Atom is a thread-safe mutable reference to a Value.
///
/// Atoms use a RwLock internally to provide thread-safe access to the
/// underlying value. Updates are atomic and can be done either by
/// directly setting a new value (`reset!`) or by applying a function
/// to the current value (`swap!`).
#[derive(Debug)]
pub struct Atom {
    value: RwLock<Value>,
}

impl Atom {
    /// Create a new Atom with the given initial value.
    pub fn new(value: Value) -> Self {
        Self { value: RwLock::new(value) }
    }

    /// Get the current value of the Atom.
    ///
    /// This acquires a read lock and clones the value.
    pub fn deref(&self) -> Value {
        self.value.read().unwrap().clone()
    }

    /// Atomically set the Atom's value to the given value.
    ///
    /// Returns the new value.
    pub fn reset(&self, new_value: Value) -> Value {
        let mut guard = self.value.write().unwrap();
        *guard = new_value.clone();
        new_value
    }

    /// Atomically update the Atom's value by applying a function.
    ///
    /// The function receives the current value and returns the new value.
    /// This is done under a write lock, so the function should be fast
    /// and should not block.
    ///
    /// Returns the new value.
    pub fn swap<F>(&self, f: F) -> Value
    where
        F: FnOnce(&Value) -> Value,
    {
        let mut guard = self.value.write().unwrap();
        let new_value = f(&*guard);
        *guard = new_value.clone();
        new_value
    }

    /// Compare and swap: atomically set the value to `new_value` if the
    /// current value equals `expected`.
    ///
    /// Returns true if the swap was successful, false otherwise.
    pub fn compare_and_set(&self, expected: &Value, new_value: Value) -> bool {
        let mut guard = self.value.write().unwrap();
        if &*guard == expected {
            *guard = new_value;
            true
        } else {
            false
        }
    }
}

impl Clone for Atom {
    fn clone(&self) -> Self {
        Self { value: RwLock::new(self.deref()) }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#<Atom {:?}>", self.deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic_span;

    fn int_val(n: i64) -> Value {
        Value::Int { span: synthetic_span!(), value: n }
    }

    #[test]
    fn test_atom_new_and_deref() {
        let atom = Atom::new(int_val(42));
        match atom.deref() {
            Value::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_atom_reset() {
        let atom = Atom::new(int_val(42));
        let new_val = atom.reset(int_val(100));
        match new_val {
            Value::Int { value, .. } => assert_eq!(value, 100),
            _ => panic!("Expected Int"),
        }
        match atom.deref() {
            Value::Int { value, .. } => assert_eq!(value, 100),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_atom_swap() {
        let atom = Atom::new(int_val(42));
        let new_val = atom.swap(|v| match v {
            Value::Int { value, span } => {
                Value::Int { span: span.clone(), value: value + 1 }
            }
            _ => v.clone(),
        });
        match new_val {
            Value::Int { value, .. } => assert_eq!(value, 43),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_atom_compare_and_set_success() {
        let atom = Atom::new(int_val(42));
        let expected = int_val(42);
        let new_val = int_val(100);
        assert!(atom.compare_and_set(&expected, new_val));
        match atom.deref() {
            Value::Int { value, .. } => assert_eq!(value, 100),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_atom_compare_and_set_failure() {
        let atom = Atom::new(int_val(42));
        let expected = int_val(99); // wrong expected value
        let new_val = int_val(100);
        assert!(!atom.compare_and_set(&expected, new_val));
        match atom.deref() {
            Value::Int { value, .. } => assert_eq!(value, 42), // unchanged
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_atom_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let atom = Arc::new(Atom::new(int_val(0)));
        let mut handles = vec![];

        for _ in 0..10 {
            let atom_clone = Arc::clone(&atom);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    atom_clone.swap(|v| match v {
                        Value::Int { value, span } => {
                            Value::Int { span: span.clone(), value: value + 1 }
                        }
                        _ => v.clone(),
                    });
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        match atom.deref() {
            Value::Int { value, .. } => assert_eq!(value, 1000),
            _ => panic!("Expected Int"),
        }
    }
}
