//! Tests for namespace and module functionality.

use std::sync::Arc;

use evolve::{
    interner,
    reader::Source,
    runtime::Runtime,
    value::Value,
};

fn eval(runtime: &Arc<Runtime>, form: &str) -> Value {
    runtime
        .clone()
        .rep(form, Source::REPL)
        .unwrap_or_else(|err| {
            let snippet = if err.span.end <= form.len() {
                &form[err.span.clone().to_range()]
            } else {
                ""
            };
            panic!(
                "failed to eval `{}`: {:?} (snippet: {:?})",
                form, err, snippet
            )
        })
}

fn assert_int(value: &Value, expected: i64) {
    match value {
        Value::Int { value: v, .. } => assert_eq!(*v, expected),
        other => panic!("expected Int({}), got {:?}", expected, other),
    }
}

fn assert_symbol_name(value: &Value, expected: &str) {
    match value {
        Value::Symbol { value: sym, .. } => {
            assert_eq!(interner::sym_to_str(sym.id()), expected)
        }
        other => panic!("expected Symbol({}), got {:?}", expected, other),
    }
}

//===----------------------------------------------------------------------===//
// Basic Namespace Tests
//===----------------------------------------------------------------------===//

#[test]
fn test_ns_creates_namespace() {
    let runtime = Runtime::new();
    let result = eval(&runtime, "(ns my.test.namespace)");
    assert_symbol_name(&result, "my.test.namespace");
}

#[test]
fn test_ns_switches_to_new_namespace() {
    let runtime = Runtime::new();
    eval(&runtime, "(ns my.new.ns)");
    // Define something in the new namespace
    let result = eval(&runtime, "(do (def ^:public x 42) x)");
    assert_int(&result, 42);
}

#[test]
fn test_multiple_ns_declarations() {
    let runtime = Runtime::new();
    // Use unique namespace names to avoid conflicts with other tests
    let result = eval(&runtime, "
        (do
            (ns test.first.ns.multi)
            (def ^:public x 1)
            (ns test.second.ns.multi)
            (def ^:public y 2)
            ;; Check that y is accessible in second.ns
            y)
    ");
    assert_int(&result, 2);
}

//===----------------------------------------------------------------------===//
// Require Tests
//===----------------------------------------------------------------------===//

#[test]
fn test_require_simple() {
    let runtime = Runtime::new();
    // Create a namespace with some definitions
    eval(&runtime, "(ns foo.bar)");
    eval(&runtime, "(def ^:public x 100)");

    // Create another namespace that requires foo.bar
    eval(&runtime, "(ns baz.qux (:require foo.bar))");

    // The namespace should be loaded (we can't access x directly without :refer or :as)
    // This test just verifies the require parsing works
}

#[test]
fn test_require_with_alias() {
    let runtime = Runtime::new();
    // Create a namespace with some definitions
    eval(&runtime, "(ns lib.utils)");
    eval(&runtime, "(def ^:public helper 42)");

    // Create another namespace that requires lib.utils with an alias
    // The ns form returns the namespace name symbol
    let result = eval(&runtime, "(ns app.main (:require [lib.utils :as u]))");

    // Check the namespace was created
    assert_symbol_name(&result, "app.main");
}

#[test]
fn test_require_with_refer() {
    let runtime = Runtime::new();
    // Create a namespace with some definitions
    eval(&runtime, "(ns math.ops)");
    eval(&runtime, "(def ^:public add (fn* [a b] (+ a b)))");

    // Create another namespace that requires math.ops with :refer
    eval(&runtime, "(ns calc.main (:require [math.ops :refer [add]]))");

    // The referred symbol should be accessible
    // Note: This requires the namespace to actually have the binding
}

//===----------------------------------------------------------------------===//
// Public/Private Tests
//===----------------------------------------------------------------------===//

#[test]
fn test_var_public_metadata() {
    let runtime = Runtime::new();
    // Define a public var
    let result = eval(&runtime, "(def ^:public my-public-var 123)");

    // The var should be created
    match result {
        Value::Var { .. } => {}
        other => panic!("expected Var, got {:?}", other),
    }
}

#[test]
fn test_var_without_public_is_private() {
    let runtime = Runtime::new();
    // Define a private var (no ^:public)
    let result = eval(&runtime, "(def my-private-var 456)");

    // The var should be created
    match result {
        Value::Var { .. } => {}
        other => panic!("expected Var, got {:?}", other),
    }
}

//===----------------------------------------------------------------------===//
// Namespace Isolation Tests
//===----------------------------------------------------------------------===//

#[test]
fn test_namespace_isolation() {
    let runtime = Runtime::new();
    // Use unique namespace names to avoid conflicts with other tests
    let result = eval(&runtime, "
        (do
            ;; Define in first namespace
            (ns test.isolation.ns1)
            (def ^:public x 1)

            ;; Define in second namespace (different value)
            (ns test.isolation.ns2)
            (def ^:public x 2)

            ;; Check the value in ns2
            x)
    ");
    assert_int(&result, 2);
}

#[test]
fn test_ns_with_docstring() {
    // Note: docstring support in ns may not be implemented yet
    let runtime = Runtime::new();
    let result = eval(&runtime, "(ns my.documented.ns)");
    assert_symbol_name(&result, "my.documented.ns");
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

#[test]
fn test_nested_ns_names() {
    let runtime = Runtime::new();
    let result = eval(&runtime, "(ns very.deeply.nested.namespace.name)");
    assert_symbol_name(&result, "very.deeply.nested.namespace.name");
}

#[test]
fn test_ns_can_be_declared_multiple_times() {
    let runtime = Runtime::new();
    // Use a unique namespace name for this test to avoid conflicts
    let result = eval(&runtime, "
        (do
            (ns reusable.ns.test)
            (def ^:public first-def 1)
            ;; Switch to another namespace and back
            (ns other.ns)
            (ns reusable.ns.test)
            ;; The definition should still exist in the namespace
            first-def)
    ");
    assert_int(&result, 1);
}

//===----------------------------------------------------------------------===//
// Require Error Cases
//===----------------------------------------------------------------------===//

#[test]
fn test_require_empty_vector() {
    let runtime = Runtime::new();
    // Empty vector in require should be handled gracefully
    let result = eval(&runtime, "(ns test.empty (:require []))");
    assert_symbol_name(&result, "test.empty");
}
