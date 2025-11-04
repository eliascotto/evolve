use crate::value::Value;
use std::fmt::Write;

/// Pretty-prints an AST (Value) with indentation and proper formatting.
/// This function displays the AST in a more readable, multi-line format.
pub fn pretty_print_ast(value: &Value) -> String {
    pretty_print_ast_with_indent(value, 0)
}

fn pretty_print_ast_with_indent(value: &Value, indent: usize) -> String {
    let indent_str = "  ".repeat(indent);
    let mut result = String::new();

    match value {
        Value::Nil => write!(result, "Nil:nil").unwrap(),
        Value::Bool(b) => write!(result, "Bool:{}", b).unwrap(),
        Value::Int(i) => write!(result, "Int:{}", i).unwrap(),
        Value::Float(f) => write!(result, "Float:{}", f).unwrap(),
        Value::Char(c) => write!(result, "Char:\\{}", c).unwrap(),
        Value::String(s) => {
            // Escape quotes in strings
            let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
            write!(result, "String:\"{}\"", escaped).unwrap();
        }
        Value::Symbol(s) => write!(result, "Symbol:{}", s).unwrap(),
        Value::Keyword(k) => write!(result, "Keyword:{}", k).unwrap(),
        Value::List(items) => {
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
        Value::Vector(items) => {
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
        Value::Map(map) => {
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
        Value::Set(set) => {
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
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn test_pretty_print_nil() {
        let value = Value::Nil;
        assert_eq!(pretty_print_ast(&value), "Nil:nil");
    }

    #[test]
    fn test_pretty_print_bool() {
        assert_eq!(pretty_print_ast(&Value::Bool(true)), "Bool:true");
        assert_eq!(pretty_print_ast(&Value::Bool(false)), "Bool:false");
    }

    #[test]
    fn test_pretty_print_int() {
        assert_eq!(pretty_print_ast(&Value::Int(42)), "Int:42");
        assert_eq!(pretty_print_ast(&Value::Int(-100)), "Int:-100");
        assert_eq!(pretty_print_ast(&Value::Int(0)), "Int:0");
    }

    #[test]
    fn test_pretty_print_float() {
        assert_eq!(pretty_print_ast(&Value::Float(3.14)), "Float:3.14");
        assert_eq!(pretty_print_ast(&Value::Float(-42.5)), "Float:-42.5");
        assert_eq!(pretty_print_ast(&Value::Float(0.0)), "Float:0");
    }

    #[test]
    fn test_pretty_print_char() {
        assert_eq!(pretty_print_ast(&Value::Char('a')), "Char:\\a");
        assert_eq!(pretty_print_ast(&Value::Char(' ')), "Char:\\ ");
        assert_eq!(pretty_print_ast(&Value::Char('\n')), "Char:\\\n");
    }

    #[test]
    fn test_pretty_print_string() {
        assert_eq!(pretty_print_ast(&Value::String("hello".to_string())), "String:\"hello\"");
        assert_eq!(pretty_print_ast(&Value::String("".to_string())), "String:\"\"");
        assert_eq!(pretty_print_ast(&Value::String("test \"quote\"".to_string())), "String:\"test \\\"quote\\\"\"");
        assert_eq!(pretty_print_ast(&Value::String("back\\slash".to_string())), "String:\"back\\\\slash\"");
    }

    #[test]
    fn test_pretty_print_symbol() {
        assert_eq!(pretty_print_ast(&Value::Symbol("foo".to_string())), "Symbol:foo");
        assert_eq!(pretty_print_ast(&Value::Symbol("my-symbol".to_string())), "Symbol:my-symbol");
    }

    #[test]
    fn test_pretty_print_keyword() {
        assert_eq!(pretty_print_ast(&Value::Keyword(":foo".to_string())), "Keyword::foo");
        assert_eq!(pretty_print_ast(&Value::Keyword(":bar".to_string())), "Keyword::bar");
    }

    #[test]
    fn test_pretty_print_empty_list() {
        let value = Value::List(vec![]);
        assert_eq!(pretty_print_ast(&value), "List:()");
    }

    #[test]
    fn test_pretty_print_list() {
        let value = Value::List(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
        ]);
        let expected = "List:(\n  Int:1\n  Int:2\n  Int:3\n)";
        assert_eq!(pretty_print_ast(&value), expected);
    }

    #[test]
    fn test_pretty_print_nested_list() {
        let value = Value::List(vec![
            Value::Int(1),
            Value::List(vec![
                Value::Int(2),
                Value::Int(3),
            ]),
        ]);
        let expected = "List:(\n  Int:1\n  List:(\n    Int:2\n    Int:3\n  )\n)";
        assert_eq!(pretty_print_ast(&value), expected);
    }

    #[test]
    fn test_pretty_print_empty_vector() {
        let value = Value::Vector(vec![]);
        assert_eq!(pretty_print_ast(&value), "Vector:[]");
    }

    #[test]
    fn test_pretty_print_vector() {
        let value = Value::Vector(vec![
            Value::String("hello".to_string()),
            Value::String("world".to_string()),
        ]);
        let expected = "Vector:[\n  String:\"hello\"\n  String:\"world\"\n]";
        assert_eq!(pretty_print_ast(&value), expected);
    }

    #[test]
    fn test_pretty_print_empty_map() {
        let value = Value::Map(BTreeMap::new());
        assert_eq!(pretty_print_ast(&value), "Map:{}");
    }

    #[test]
    fn test_pretty_print_map() {
        let mut map = BTreeMap::new();
        map.insert(Value::String("key1".to_string()), Value::Int(1));
        map.insert(Value::String("key2".to_string()), Value::Int(2));
        let value = Value::Map(map);
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
        let value = Value::Set(BTreeSet::new());
        assert_eq!(pretty_print_ast(&value), "Set:{}");
    }

    #[test]
    fn test_pretty_print_set() {
        let mut set = BTreeSet::new();
        set.insert(Value::Int(1));
        set.insert(Value::Int(2));
        set.insert(Value::Int(3));
        let value = Value::Set(set);
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
        let value = Value::List(vec![
            Value::Symbol("foo".to_string()),
            Value::List(vec![
                Value::Int(1),
                Value::Vector(vec![
                    Value::String("hello".to_string()),
                    Value::Bool(true),
                ]),
            ]),
        ]);
        let expected = "List:(\n  Symbol:foo\n  List:(\n    Int:1\n    Vector:[\n      String:\"hello\"\n      Bool:true\n    ]\n  )\n)";
        assert_eq!(pretty_print_ast(&value), expected);
    }
}

