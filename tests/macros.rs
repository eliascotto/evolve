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

fn assert_bool(value: &Value, expected: bool) {
    match value {
        Value::Bool { value: v, .. } => assert_eq!(*v, expected),
        other => panic!("expected Bool({}), got {:?}", expected, other),
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

fn list_items<'a>(value: &'a Value) -> Vec<&'a Value> {
    match value {
        Value::List { value: list, .. } => list.iter().collect(),
        other => panic!("expected List, got {:?}", other),
    }
}

fn vector_items<'a>(value: &'a Value) -> Vec<&'a Value> {
    match value {
        Value::Vector { value: vector, .. } => vector.iter().collect(),
        other => panic!("expected Vector, got {:?}", other),
    }
}

#[test]
fn defmacro_defines_callable_macro() {
    let runtime = Runtime::new();
    let result = eval(
        &runtime,
        "
        (do
            (defmacro unless [pred body] `(if ~pred nil ~body))
            [(unless false 42) (unless true 42)])
        ",
    );

    let items = vector_items(&result);
    assert_int(items[0], 42);
    assert!(matches!(items[1], Value::Nil { .. }));
}

#[test]
fn macroexpand1_expands_one_level_only() {
    let runtime = Runtime::new();
    let expanded = eval(
        &runtime,
        "
        (do
            (defmacro twice [x] `(do ~x ~x))
            (macroexpand1 '(twice 1)))
        ",
    );
    let items = list_items(&expanded);
    assert_eq!(items.len(), 3);
    assert_symbol_name(items[0], "do");
    assert_int(items[1], 1);
    assert_int(items[2], 1);
}

#[test]
fn macroexpand_fully_expands_nested_macros() {
    let runtime = Runtime::new();
    let expanded = eval(
        &runtime,
        "
        (do
            (defmacro twice [x] `(do ~x ~x))
            (defmacro thrice [x] `(do (twice ~x) ~x))
            (macroexpand '(thrice 1)))
        ",
    );
    let items = list_items(&expanded);
    assert_eq!(items.len(), 3);
    assert_symbol_name(items[0], "do");

    let nested = list_items(items[1]);
    assert_symbol_name(nested[0], "do");
    assert_int(nested[1], 1);
    assert_int(nested[2], 1);
    assert_int(items[2], 1);
}

#[test]
fn quasiquote_unquote_and_splice_form_lists() {
    let runtime = Runtime::new();
    let result = eval(
        &runtime,
        "
        (do
            (def b 2)
            (def c '(3 4))
            `(a ~b ~@c))
        ",
    );
    let items = list_items(&result);
    // (a 2 3 4)
    assert_eq!(items.len(), 4);
    assert_symbol_name(items[0], "a");
    assert_int(items[1], 2);
    assert_int(items[2], 3);
    assert_int(items[3], 4);
}

#[test]
fn macro_expansion_occurs_before_evaluation() {
    let runtime = Runtime::new();
    let result = eval(
        &runtime,
        "
        (do
            (defmacro short-circuit [a b] `(if ~a ~a ~b))
            (short-circuit true should-never-run))
        ",
    );
    assert_bool(&result, true);
}

#[test]
fn nested_macros_expand_and_evaluate() {
    let runtime = Runtime::new();
    let result = eval(
        &runtime,
        "
        (do
            (defmacro wrap [expr] `(list ~expr))
            (defmacro double-wrap [expr] `(wrap (wrap ~expr)))
            (double-wrap 1))
        ",
    );
    let outer = list_items(&result);
    assert_eq!(outer.len(), 1);

    let inner = list_items(outer[0]);
    assert_eq!(inner.len(), 1);
    assert_int(inner[0], 1);
}

