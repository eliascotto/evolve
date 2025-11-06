use std::fmt::Write;

use crate::interner;
use crate::value::Value;

/// Pretty-prints an AST (Value) with indentation and proper formatting.
/// This function displays the AST in a more readable, multi-line format.
pub fn pretty_print_ast(value: &Value) -> String {
    pretty_print_ast_with_indent(value, 0)
}

fn pretty_print_ast_with_indent(value: &Value, indent: usize) -> String {
    let indent_str = "  ".repeat(indent);
    let mut result = String::new();

    match value {
        Value::Nil { span: _ } => write!(result, "Nil:nil").unwrap(),
        Value::Bool { span: _, value: b } => write!(result, "Bool:{}", b).unwrap(),
        Value::Int { span: _, value: i } => write!(result, "Int:{}", i).unwrap(),
        Value::Float { span: _, value: f } => write!(result, "Float:{}", f).unwrap(),
        Value::Char { span: _, value: c } => write!(result, "Char:\\{}", c).unwrap(),
        Value::String { span: _, value: s } => {
            // Escape quotes in strings
            let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
            write!(result, "String:\"{}\"", escaped).unwrap();
        }
        Value::Symbol { span: _, value: s, meta: _ } => {
            write!(result, "Symbol:{}", interner::sym_to_str(*s)).unwrap()
        }
        Value::Keyword { span: _, value: k } => {
            write!(result, "Keyword:{}", interner::kw_to_str(*k)).unwrap()
        }
        Value::List { span: _, value: items, meta: _ } => {
            write!(result, "List:").unwrap();
            write!(result, "(").unwrap();
            if items.is_empty() {
                write!(result, ")").unwrap();
            } else {
                write!(result, "\n").unwrap();
                for (i, item) in items.iter().enumerate() {
                    write!(result, "{}{}", indent_str, "  ").unwrap();
                    write!(result, "{}", pretty_print_ast_with_indent(item, indent + 1)).unwrap();
                    if i < items.len() - 1 {
                        write!(result, "\n").unwrap();
                    }
                }
                write!(result, "\n{}", indent_str).unwrap();
                write!(result, ")").unwrap();
            }
        }
        Value::Vector { span: _, value: items, meta: _ } => {
            write!(result, "Vector:").unwrap();
            write!(result, "[").unwrap();
            if items.is_empty() {
                write!(result, "]").unwrap();
            } else {
                write!(result, "\n").unwrap();
                for (i, item) in items.iter().enumerate() {
                    write!(result, "{}{}", indent_str, "  ").unwrap();
                    write!(result, "{}", pretty_print_ast_with_indent(item, indent + 1)).unwrap();
                    if i < items.len() - 1 {
                        write!(result, "\n").unwrap();
                    }
                }
                write!(result, "\n{}", indent_str).unwrap();
                write!(result, "]").unwrap();
            }
        }
        Value::Map { span: _, value: map, meta: _ } => {
            write!(result, "Map:").unwrap();
            write!(result, "{{").unwrap();
            if map.is_empty() {
                write!(result, "}}").unwrap();
            } else {
                write!(result, "\n").unwrap();
                let entries: Vec<_> = map.iter().collect();
                for (i, (k, v)) in entries.iter().enumerate() {
                    write!(result, "{}{}", indent_str, "  ").unwrap();
                    write!(result, "{} ", pretty_print_ast_with_indent(k, indent + 1)).unwrap();
                    write!(result, "{}", pretty_print_ast_with_indent(v, indent + 1)).unwrap();
                    if i < entries.len() - 1 {
                        write!(result, "\n").unwrap();
                    }
                }
                write!(result, "\n{}", indent_str).unwrap();
                write!(result, "}}").unwrap();
            }
        }
        Value::Set { span: _, value: set, meta: _ } => {
            write!(result, "Set:").unwrap();
            write!(result, "{{").unwrap();
            if set.is_empty() {
                write!(result, "}}").unwrap();
            } else {
                write!(result, "\n").unwrap();
                let items: Vec<_> = set.iter().collect();
                for (i, item) in items.iter().enumerate() {
                    write!(result, "{}{}", indent_str, "  ").unwrap();
                    write!(result, "{}", pretty_print_ast_with_indent(item, indent + 1)).unwrap();
                    if i < items.len() - 1 {
                        write!(result, "\n").unwrap();
                    }
                }
                write!(result, "\n{}", indent_str).unwrap();
                write!(result, "}}").unwrap();
            }
        }
        Value::SpecialForm { span: _ } => write!(result, "SpecialForm").unwrap(),
        Value::Namespace { span: _, value: ns } => {
            write!(result, "Namespace:{}", interner::ns_to_str(*ns)).unwrap()
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;
    use logos::Span;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn test_pretty_print_nil() {
        let value = Value::Nil { span: Span { start: 0, end: 0 } };
        assert_eq!(pretty_print_ast(&value), "Nil:nil");
    }

    #[test]
    fn test_pretty_print_bool() {
        assert_eq!(
            pretty_print_ast(&Value::Bool { span: Span { start: 0, end: 0 }, value: true }),
            "Bool:true"
        );
        assert_eq!(
            pretty_print_ast(&Value::Bool { span: Span { start: 0, end: 0 }, value: false }),
            "Bool:false"
        );
    }

    #[test]
    fn test_pretty_print_int() {
        assert_eq!(
            pretty_print_ast(&Value::Int { span: Span { start: 0, end: 0 }, value: 42 }),
            "Int:42"
        );
        assert_eq!(
            pretty_print_ast(&Value::Int { span: Span { start: 0, end: 0 }, value: -100 }),
            "Int:-100"
        );
        assert_eq!(
            pretty_print_ast(&Value::Int { span: Span { start: 0, end: 0 }, value: 0 }),
            "Int:0"
        );
    }

    #[test]
    fn test_pretty_print_float() {
        assert_eq!(
            pretty_print_ast(&Value::Float { span: Span { start: 0, end: 0 }, value: 3.14 }),
            "Float:3.14"
        );
        assert_eq!(
            pretty_print_ast(&Value::Float { span: Span { start: 0, end: 0 }, value: -42.5 }),
            "Float:-42.5"
        );
        assert_eq!(
            pretty_print_ast(&Value::Float { span: Span { start: 0, end: 0 }, value: 0.0 }),
            "Float:0"
        );
    }

    #[test]
    fn test_pretty_print_char() {
        assert_eq!(
            pretty_print_ast(&Value::Char { span: Span { start: 0, end: 0 }, value: 'a' }),
            "Char:\\a"
        );
        assert_eq!(
            pretty_print_ast(&Value::Char { span: Span { start: 0, end: 0 }, value: ' ' }),
            "Char:\\ "
        );
        assert_eq!(
            pretty_print_ast(&Value::Char { span: Span { start: 0, end: 0 }, value: '\n' }),
            "Char:\\\n"
        );
    }

    #[test]
    fn test_pretty_print_string() {
        assert_eq!(
            pretty_print_ast(&Value::String {
                span: Span { start: 0, end: 0 },
                value: "hello".to_string()
            }),
            "String:\"hello\""
        );
        assert_eq!(
            pretty_print_ast(&Value::String {
                span: Span { start: 0, end: 0 },
                value: "".to_string()
            }),
            "String:\"\""
        );
        assert_eq!(
            pretty_print_ast(&Value::String {
                span: Span { start: 0, end: 0 },
                value: "test \"quote\"".to_string()
            }),
            "String:\"test \\\"quote\\\"\""
        );
        assert_eq!(
            pretty_print_ast(&Value::String {
                span: Span { start: 0, end: 0 },
                value: "back\\slash".to_string()
            }),
            "String:\"back\\\\slash\""
        );
    }

    #[test]
    fn test_pretty_print_symbol() {
        assert_eq!(
            pretty_print_ast(&Value::Symbol {
                span: Span { start: 0, end: 0 },
                value: interner::intern_sym("foo"),
                meta: None,
            }),
            "Symbol:foo"
        );
        assert_eq!(
            pretty_print_ast(&Value::Symbol {
                span: Span { start: 0, end: 0 },
                value: interner::intern_sym("my-symbol"),
                meta: None,
            }),
            "Symbol:my-symbol"
        );
    }

    #[test]
    fn test_pretty_print_keyword() {
        assert_eq!(
            pretty_print_ast(&Value::Keyword {
                span: Span { start: 0, end: 0 },
                value: interner::intern_kw(":foo")
            }),
            "Keyword:foo"
        );
        assert_eq!(
            pretty_print_ast(&Value::Keyword {
                span: Span { start: 0, end: 0 },
                value: interner::intern_kw(":bar"),
            }),
            "Keyword:bar"
        );
    }

    #[test]
    fn test_pretty_print_empty_list() {
        let value = Value::List { span: Span { start: 0, end: 0 }, value: vec![], meta: None };
        assert_eq!(pretty_print_ast(&value), "List:()");
    }

    #[test]
    fn test_pretty_print_list() {
        let value = Value::List {
            span: Span { start: 0, end: 0 },
            value: vec![
                Value::Int { span: Span { start: 0, end: 0 }, value: 1 },
                Value::Int { span: Span { start: 0, end: 0 }, value: 2 },
                Value::Int { span: Span { start: 0, end: 0 }, value: 3 },
            ],
            meta: None,
        };
        let expected = "List:(\n  Int:1\n  Int:2\n  Int:3\n)";
        assert_eq!(pretty_print_ast(&value), expected);
    }

    #[test]
    fn test_pretty_print_nested_list() {
        let value = Value::List {
            span: Span { start: 0, end: 0 },
            value: vec![
                Value::Int { span: Span { start: 0, end: 0 }, value: 1 },
                Value::List {
                    span: Span { start: 0, end: 0 },
                    value: vec![
                        Value::Int { span: Span { start: 0, end: 0 }, value: 2 },
                        Value::Int { span: Span { start: 0, end: 0 }, value: 3 },
                    ],
                    meta: None,
                },
            ],
            meta: None,
        };
        let expected = "List:(\n  Int:1\n  List:(\n    Int:2\n    Int:3\n  )\n)";
        assert_eq!(pretty_print_ast(&value), expected);
    }

    #[test]
    fn test_pretty_print_empty_vector() {
        let value = Value::Vector { span: Span { start: 0, end: 0 }, value: vec![], meta: None };
        assert_eq!(pretty_print_ast(&value), "Vector:[]");
    }

    #[test]
    fn test_pretty_print_vector() {
        let value = Value::Vector {
            span: Span { start: 0, end: 0 },
            value: vec![
                Value::String { span: Span { start: 0, end: 0 }, value: "hello".to_string() },
                Value::String { span: Span { start: 0, end: 0 }, value: "world".to_string() },
            ],
            meta: None,
        };
        let expected = "Vector:[\n  String:\"hello\"\n  String:\"world\"\n]";
        assert_eq!(pretty_print_ast(&value), expected);
    }

    #[test]
    fn test_pretty_print_empty_map() {
        let value =
            Value::Map { span: Span { start: 0, end: 0 }, value: BTreeMap::new(), meta: None };
        assert_eq!(pretty_print_ast(&value), "Map:{}");
    }

    #[test]
    fn test_pretty_print_map() {
        let mut map = BTreeMap::new();
        map.insert(
            Value::String { span: Span { start: 0, end: 0 }, value: "key1".to_string() },
            Value::Int { span: Span { start: 0, end: 0 }, value: 1 },
        );
        map.insert(
            Value::String { span: Span { start: 0, end: 0 }, value: "key2".to_string() },
            Value::Int { span: Span { start: 0, end: 0 }, value: 2 },
        );
        let value = Value::Map { span: Span { start: 0, end: 0 }, value: map, meta: None };
        let output = pretty_print_ast(&value);
        // Since BTreeMap is ordered, we check that it contains the expected elements
        assert!(output.starts_with("Map:{\n"));
        assert!(output.contains("String:\"key1\""));
        assert!(output.contains("String:\"key2\""));
        assert!(output.contains("Int:1"));
        assert!(output.contains("Int:2"));
        assert!(output.ends_with("\n}"));
    }

    #[test]
    fn test_pretty_print_empty_set() {
        let value =
            Value::Set { span: Span { start: 0, end: 0 }, value: BTreeSet::new(), meta: None };
        assert_eq!(pretty_print_ast(&value), "Set:{}");
    }

    #[test]
    fn test_pretty_print_set() {
        let mut set = BTreeSet::new();
        set.insert(Value::Int { span: Span { start: 0, end: 0 }, value: 1 });
        set.insert(Value::Int { span: Span { start: 0, end: 0 }, value: 2 });
        set.insert(Value::Int { span: Span { start: 0, end: 0 }, value: 3 });
        let value = Value::Set { span: Span { start: 0, end: 0 }, value: set, meta: None };
        let output = pretty_print_ast(&value);
        // Since BTreeSet is ordered, we check that it contains the expected elements
        assert!(output.starts_with("Set:{\n"));
        assert!(output.contains("Int:1"));
        assert!(output.contains("Int:2"));
        assert!(output.contains("Int:3"));
        assert!(output.ends_with("\n}"));
    }

    #[test]
    fn test_pretty_print_complex_nested() {
        let value = Value::List {
            span: Span { start: 0, end: 0 },
            value: vec![
                Value::Symbol {
                    span: Span { start: 0, end: 0 },
                    value: interner::intern_sym("foo"),
                    meta: None,
                },
                Value::List {
                    span: Span { start: 0, end: 0 },
                    value: vec![
                        Value::Int { span: Span { start: 0, end: 0 }, value: 1 },
                        Value::Vector {
                            span: Span { start: 0, end: 0 },
                            value: vec![
                                Value::String {
                                    span: Span { start: 0, end: 0 },
                                    value: "hello".to_string(),
                                },
                                Value::Bool { span: Span { start: 0, end: 0 }, value: true },
                            ],
                            meta: None,
                        },
                    ],
                    meta: None,
                },
            ],
            meta: None,
        };
        let expected = "List:(\n  Symbol:foo\n  List:(\n    Int:1\n    Vector:[\n      String:\"hello\"\n      Bool:true\n    ]\n  )\n)";
        assert_eq!(pretty_print_ast(&value), expected);
    }
}
