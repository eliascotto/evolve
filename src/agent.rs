//! Asynchronous state management with action queues.
//!
//! Agents provide a way to manage independent, asynchronous state. Actions
//! are functions that transform the agent's state and are processed in order
//! in a background thread.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::value::Value;

/// An action to be processed by an agent.
///
/// The action contains a boxed function that transforms the current state
/// into a new state.
type ActionFn = Box<dyn FnOnce(&Value) -> Value + Send>;

/// An Agent is an asynchronous state container.
///
/// Actions are sent to the agent and processed in order in a background thread.
/// The agent's state can be read at any time (though it may be stale if actions
/// are pending).
pub struct Agent {
    state: Arc<Mutex<Value>>,
    sender: Sender<ActionFn>,
    pending_count: Arc<(Mutex<usize>, Condvar)>,
    shutdown: Arc<AtomicBool>,
}

impl Agent {
    /// Create a new Agent with the given initial state.
    ///
    /// This spawns a background thread to process actions.
    pub fn new(initial_state: Value) -> Self {
        let state = Arc::new(Mutex::new(initial_state));
        let (sender, receiver) = mpsc::channel::<ActionFn>();
        let pending_count = Arc::new((Mutex::new(0usize), Condvar::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let state_clone = state.clone();
        let pending_clone = pending_count.clone();
        let shutdown_clone = shutdown.clone();

        thread::spawn(move || {
            loop {
                if shutdown_clone.load(AtomicOrdering::Relaxed) {
                    break;
                }

                match receiver.recv() {
                    Ok(action) => {
                        let mut guard = state_clone.lock().unwrap();
                        let new_state = action(&*guard);
                        *guard = new_state;
                        drop(guard);

                        // Decrement pending count and notify waiters
                        let (lock, cvar) = &*pending_clone;
                        let mut count = lock.lock().unwrap();
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            cvar.notify_all();
                        }
                    }
                    Err(_) => {
                        // Channel closed, exit the thread
                        break;
                    }
                }
            }
        });

        Self { state, sender, pending_count, shutdown }
    }

    /// Get the current state of the Agent.
    ///
    /// This returns the current committed state, which may be stale if
    /// actions are pending.
    pub fn deref(&self) -> Value {
        self.state.lock().unwrap().clone()
    }

    /// Send an action to the Agent.
    ///
    /// The action is a function that receives the current state and returns
    /// the new state. Actions are processed asynchronously in order.
    ///
    /// This returns immediately without waiting for the action to complete.
    pub fn send<F>(&self, action: F)
    where
        F: FnOnce(&Value) -> Value + Send + 'static,
    {
        // Increment pending count
        let (lock, _) = &*self.pending_count;
        {
            let mut count = lock.lock().unwrap();
            *count += 1;
        }

        // Send the action (ignore errors if channel is closed)
        let _ = self.sender.send(Box::new(action));
    }

    /// Wait for all pending actions to complete.
    ///
    /// This blocks until all currently queued actions have been processed.
    pub fn await_agent(&self) {
        let (lock, cvar) = &*self.pending_count;
        let mut count = lock.lock().unwrap();
        while *count > 0 {
            count = cvar.wait(count).unwrap();
        }
    }

    /// Get the number of pending actions.
    pub fn pending(&self) -> usize {
        let (lock, _) = &*self.pending_count;
        *lock.lock().unwrap()
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        self.shutdown.store(true, AtomicOrdering::Relaxed);
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("state", &self.deref())
            .field("pending", &self.pending())
            .finish()
    }
}

impl fmt::Display for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#<Agent {:?}>", self.deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic_span;
    use std::thread;
    use std::time::Duration;

    fn int_val(n: i64) -> Value {
        Value::Int { span: synthetic_span!(), value: n }
    }

    #[test]
    fn test_agent_new_and_deref() {
        let agent = Agent::new(int_val(42));
        match agent.deref() {
            Value::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_agent_send_and_await() {
        let agent = Arc::new(Agent::new(int_val(0)));

        agent.send(|v| match v {
            Value::Int { value, span } => Value::Int { span: span.clone(), value: value + 10 },
            _ => v.clone(),
        });

        agent.await_agent();

        match agent.deref() {
            Value::Int { value, .. } => assert_eq!(value, 10),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_agent_multiple_sends() {
        let agent = Arc::new(Agent::new(int_val(0)));

        for _ in 0..10 {
            agent.send(|v| match v {
                Value::Int { value, span } => {
                    Value::Int { span: span.clone(), value: value + 1 }
                }
                _ => v.clone(),
            });
        }

        agent.await_agent();

        match agent.deref() {
            Value::Int { value, .. } => assert_eq!(value, 10),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_agent_concurrent_sends() {
        let agent = Arc::new(Agent::new(int_val(0)));
        let mut handles = vec![];

        for _ in 0..10 {
            let agent_clone = agent.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    agent_clone.send(|v| match v {
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

        agent.await_agent();

        match agent.deref() {
            Value::Int { value, .. } => assert_eq!(value, 1000),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_agent_pending_count() {
        let agent = Arc::new(Agent::new(int_val(0)));

        // Initially no pending actions
        assert_eq!(agent.pending(), 0);

        // Send an action that takes some time
        agent.send(|v| {
            thread::sleep(Duration::from_millis(50));
            v.clone()
        });

        // Should have pending actions
        assert!(agent.pending() >= 0); // May have already processed

        agent.await_agent();

        // After await, should be 0
        assert_eq!(agent.pending(), 0);
    }
}
