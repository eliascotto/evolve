//! Software Transactional Memory (STM) for coordinated multi-reference updates.
//!
//! STM provides a way to perform coordinated, synchronous updates to multiple
//! Refs within a transaction. Transactions are optimistic and will retry if
//! conflicts are detected.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, RwLock};

use crate::value::Value;

/// Global transaction counter for generating unique transaction IDs.
static TRANSACTION_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static CURRENT_TRANSACTION: RefCell<Option<TransactionContext>> = const { RefCell::new(None) };
}

/// A Ref is an STM reference - a transactional mutable cell.
///
/// Refs can only be modified within a transaction (dosync block).
/// Outside of transactions, refs can be read but not modified.
#[derive(Debug)]
pub struct Ref {
    value: RwLock<Value>,
    version: AtomicU64,
}

impl Ref {
    /// Create a new Ref with the given initial value.
    pub fn new(value: Value) -> Self {
        Self { value: RwLock::new(value), version: AtomicU64::new(0) }
    }

    /// Get the current value of the Ref.
    ///
    /// This can be called from inside or outside a transaction.
    /// Inside a transaction, it returns the transaction-local value if one exists,
    /// and records the read for conflict detection.
    pub fn deref(&self) -> Value {
        // Check if we're in a transaction and have a local value
        if let Some(local_value) = self.get_transaction_local() {
            return local_value;
        }

        // If in a transaction, record this read
        CURRENT_TRANSACTION.with(|tx| {
            if let Some(ctx) = tx.borrow().as_ref() {
                let ptr = self as *const Ref as usize;
                let mut reads = ctx.reads.borrow_mut();
                if !reads.contains_key(&ptr) {
                    reads.insert(ptr, self.committed_version());
                }
            }
        });

        self.value.read().unwrap().clone()
    }

    /// Get the current version number.
    pub fn version(&self) -> u64 {
        self.version.load(AtomicOrdering::SeqCst)
    }

    /// Internal: Get the committed value, ignoring transaction-local state.
    fn committed_value(&self) -> Value {
        self.value.read().unwrap().clone()
    }

    /// Internal: Get the committed version.
    fn committed_version(&self) -> u64 {
        self.version.load(AtomicOrdering::SeqCst)
    }

    /// Internal: Check if there's a transaction-local value for this ref.
    fn get_transaction_local(&self) -> Option<Value> {
        CURRENT_TRANSACTION.with(|tx| {
            if let Some(ctx) = tx.borrow().as_ref() {
                let ptr = self as *const Ref as usize;
                ctx.writes.borrow().get(&ptr).cloned()
            } else {
                None
            }
        })
    }

    /// Alter the Ref's value within a transaction.
    ///
    /// The function receives the current value and returns the new value.
    /// This change is not committed until the transaction completes successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if called outside of a transaction.
    pub fn alter<F>(&self, f: F) -> Result<Value, StmError>
    where
        F: FnOnce(&Value) -> Value,
    {
        CURRENT_TRANSACTION.with(|tx| {
            let ctx = tx.borrow();
            if ctx.is_none() {
                return Err(StmError::NotInTransaction);
            }
            let ctx = ctx.as_ref().unwrap();

            // Get the current value (might be from a previous alter in this tx)
            let current = self.deref();
            let new_value = f(&current);

            // Record the read if we haven't already
            let ptr = self as *const Ref as usize;
            {
                let mut reads = ctx.reads.borrow_mut();
                if !reads.contains_key(&ptr) {
                    reads.insert(ptr, self.committed_version());
                }
            }

            // Record the write
            {
                let mut writes = ctx.writes.borrow_mut();
                writes.insert(ptr, new_value.clone());
            }

            Ok(new_value)
        })
    }

    /// Set the Ref's value within a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if called outside of a transaction.
    pub fn ref_set(&self, value: Value) -> Result<Value, StmError> {
        self.alter(|_| value.clone())
    }

    /// Internal: Commit a value to this ref (called at transaction commit time).
    fn commit(&self, value: Value) {
        let mut guard = self.value.write().unwrap();
        *guard = value;
        self.version.fetch_add(1, AtomicOrdering::SeqCst);
    }
}

impl Clone for Ref {
    fn clone(&self) -> Self {
        Self {
            value: RwLock::new(self.committed_value()),
            version: AtomicU64::new(self.committed_version()),
        }
    }
}

impl fmt::Display for Ref {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#<Ref {:?}>", self.deref())
    }
}

/// Transaction context for tracking reads and writes.
#[derive(Debug)]
struct TransactionContext {
    #[allow(dead_code)]
    id: u64,
    reads: RefCell<HashMap<usize, u64>>,  // Ref ptr -> version at read time
    writes: RefCell<HashMap<usize, Value>>, // Ref ptr -> new value
}

impl TransactionContext {
    fn new() -> Self {
        Self {
            id: TRANSACTION_COUNTER.fetch_add(1, AtomicOrdering::SeqCst),
            reads: RefCell::new(HashMap::new()),
            writes: RefCell::new(HashMap::new()),
        }
    }
}

/// STM errors.
#[derive(Debug, Clone)]
pub enum StmError {
    /// Operation requires a transaction but none is active.
    NotInTransaction,
    /// Transaction detected a conflict and should retry.
    Conflict,
    /// Maximum retry count exceeded.
    MaxRetriesExceeded,
}

impl fmt::Display for StmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StmError::NotInTransaction => {
                write!(f, "Operation requires a transaction (use dosync)")
            }
            StmError::Conflict => write!(f, "Transaction conflict detected"),
            StmError::MaxRetriesExceeded => {
                write!(f, "Transaction exceeded maximum retry count")
            }
        }
    }
}

/// Maximum number of transaction retries before giving up.
const MAX_RETRIES: usize = 10000;

/// Global lock for STM commit serialization.
static COMMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Execute a function within a transaction.
///
/// The function will be retried if conflicts are detected with other
/// concurrent transactions. All Ref modifications within the function
/// are atomic - they either all succeed or none do.
///
/// # Arguments
///
/// * `f` - The function to execute within the transaction.
/// * `refs` - A slice of refs that are involved in the transaction (for validation).
///
/// # Returns
///
/// The return value of the function, or an error if the transaction could not complete.
pub fn dosync<F, T>(f: F, refs: &[Arc<Ref>]) -> Result<T, StmError>
where
    F: Fn() -> T,
{
    for _attempt in 0..MAX_RETRIES {
        // Start a new transaction
        let ctx = TransactionContext::new();
        CURRENT_TRANSACTION.with(|tx| {
            *tx.borrow_mut() = Some(ctx);
        });

        // Execute the transaction body
        let result = f();

        // Acquire the commit lock to serialize commits
        let _commit_guard = COMMIT_LOCK.lock().unwrap();

        // Try to commit
        let commit_result = CURRENT_TRANSACTION.with(|tx| {
            let ctx = tx.borrow();
            let ctx = ctx.as_ref().unwrap();

            // Validate: check that all read refs still have the same version
            let reads = ctx.reads.borrow();
            for (ptr, version) in reads.iter() {
                // Find the ref in our refs slice
                for r in refs {
                    if Arc::as_ptr(r) as usize == *ptr {
                        if r.committed_version() != *version {
                            return Err(StmError::Conflict);
                        }
                        break;
                    }
                }
            }

            // Commit all writes
            let writes = ctx.writes.borrow();
            for (ptr, value) in writes.iter() {
                for r in refs {
                    if Arc::as_ptr(r) as usize == *ptr {
                        r.commit(value.clone());
                        break;
                    }
                }
            }

            Ok(())
        });

        // Clear transaction context
        CURRENT_TRANSACTION.with(|tx| {
            *tx.borrow_mut() = None;
        });

        // Drop commit guard before checking result
        drop(_commit_guard);

        match commit_result {
            Ok(()) => return Ok(result),
            Err(StmError::Conflict) => continue,
            Err(e) => return Err(e),
        }
    }

    Err(StmError::MaxRetriesExceeded)
}

/// Check if we're currently in a transaction.
pub fn in_transaction() -> bool {
    CURRENT_TRANSACTION.with(|tx| tx.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic_span;
    use std::thread;

    fn int_val(n: i64) -> Value {
        Value::Int { span: synthetic_span!(), value: n }
    }

    #[test]
    fn test_ref_new_and_deref() {
        let r = Ref::new(int_val(42));
        match r.deref() {
            Value::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_alter_outside_transaction_fails() {
        let r = Ref::new(int_val(42));
        let result = r.alter(|v| match v {
            Value::Int { value, span } => Value::Int { span: span.clone(), value: value + 1 },
            _ => v.clone(),
        });
        assert!(matches!(result, Err(StmError::NotInTransaction)));
    }

    #[test]
    fn test_simple_transaction() {
        let r = Arc::new(Ref::new(int_val(0)));
        let refs = vec![r.clone()];

        let result = dosync(
            || {
                r.alter(|v| match v {
                    Value::Int { value, span } => {
                        Value::Int { span: span.clone(), value: value + 10 }
                    }
                    _ => v.clone(),
                })
                .unwrap();
                42
            },
            &refs,
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        match r.deref() {
            Value::Int { value, .. } => assert_eq!(value, 10),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_multiple_alters_in_transaction() {
        let r = Arc::new(Ref::new(int_val(0)));
        let refs = vec![r.clone()];

        dosync(
            || {
                r.alter(|v| match v {
                    Value::Int { value, span } => {
                        Value::Int { span: span.clone(), value: value + 5 }
                    }
                    _ => v.clone(),
                })
                .unwrap();
                r.alter(|v| match v {
                    Value::Int { value, span } => {
                        Value::Int { span: span.clone(), value: value + 5 }
                    }
                    _ => v.clone(),
                })
                .unwrap();
            },
            &refs,
        )
        .unwrap();

        match r.deref() {
            Value::Int { value, .. } => assert_eq!(value, 10),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_concurrent_transactions() {
        let r = Arc::new(Ref::new(int_val(0)));
        let refs = vec![r.clone()];
        let mut handles = vec![];

        for _ in 0..10 {
            let r_clone = r.clone();
            let refs_clone = refs.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    dosync(
                        || {
                            r_clone
                                .alter(|v| match v {
                                    Value::Int { value, span } => {
                                        Value::Int { span: span.clone(), value: value + 1 }
                                    }
                                    _ => v.clone(),
                                })
                                .unwrap();
                        },
                        &refs_clone,
                    )
                    .unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        match r.deref() {
            Value::Int { value, .. } => assert_eq!(value, 1000),
            _ => panic!("Expected Int"),
        }
    }
}
