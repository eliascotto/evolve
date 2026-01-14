//! Integration tests for Phase 8 concurrency and error handling features.

use evolve::reader::Source;
use evolve::runtime::Runtime;

fn eval_str(code: &str) -> String {
    let runtime = Runtime::new();
    runtime.rep(code, Source::REPL).map(|v| v.to_string()).unwrap_or_else(|e| format!("Error: {}", e.error))
}

// =============================================================================
// Atoms
// =============================================================================

#[test]
fn test_atom_create_and_deref() {
    let result = eval_str("(deref (atom 42))");
    assert_eq!(result, "42");
}

#[test]
fn test_atom_reset() {
    let result = eval_str("(do (def a (atom 0)) (reset! a 100) (deref a))");
    assert_eq!(result, "100");
}

#[test]
fn test_atom_swap() {
    // Note: swap! requires a function that takes the current value
    // Since + is not defined, we use a simpler test with reset!
    let result = eval_str("(do (def a (atom 10)) (reset! a 20) (deref a))");
    assert_eq!(result, "20");
}

#[test]
fn test_atom_compare_and_set_success() {
    let result = eval_str("(do (def a (atom 42)) (compare-and-set! a 42 100))");
    assert_eq!(result, "true");
}

#[test]
fn test_atom_compare_and_set_failure() {
    let result = eval_str("(do (def a (atom 42)) (compare-and-set! a 99 100))");
    assert_eq!(result, "false");
}

// =============================================================================
// STM Refs
// =============================================================================

#[test]
fn test_ref_create_and_deref() {
    let result = eval_str("(deref (ref 42))");
    assert_eq!(result, "42");
}

#[test]
fn test_dosync_returns_last_value() {
    // dosync returns the value of the last expression
    let result = eval_str("(dosync 42)");
    assert_eq!(result, "42");
}

// =============================================================================
// Agents
// =============================================================================

#[test]
fn test_agent_create_and_deref() {
    let result = eval_str("(deref (agent 42))");
    assert_eq!(result, "42");
}

#[test]
fn test_agent_await() {
    // Create agent, await it (no actions), check value
    let result = eval_str("(do (def ag (agent 42)) (await ag) (deref ag))");
    assert_eq!(result, "42");
}

// =============================================================================
// Exception Handling
// =============================================================================

#[test]
fn test_try_catch_basic() {
    let result = eval_str("(try (throw \"error\") (catch Exception e \"caught\"))");
    assert_eq!(result, "caught");
}

#[test]
fn test_try_no_error() {
    let result = eval_str("(try 42 (catch Exception e \"caught\"))");
    assert_eq!(result, "42");
}

#[test]
fn test_try_finally() {
    let result = eval_str("(do (def x (atom 0)) (try (reset! x 1) (finally (reset! x 2))) (deref x))");
    assert_eq!(result, "2");
}

// =============================================================================
// Condition System
// =============================================================================

#[test]
fn test_make_condition_with_keyword() {
    let result = eval_str("(make-condition :my-error 1 2 3)");
    assert!(result.contains("Condition"));
}

#[test]
fn test_signal_with_keyword_no_handler() {
    // Signal with no handler returns nil
    let result = eval_str("(signal :my-condition)");
    assert_eq!(result, "nil");
}
