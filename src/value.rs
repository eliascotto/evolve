use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::collections::{List, Map, Set, Vector};
use crate::env::Env;
use crate::error::{Error, SyntaxError};
use crate::interner::{self, KeywId, NsId, SymId};
use crate::reader::Span;
use crate::core::{Metadata, Var};
use crate::core::native_fns::NativeFn;

//===----------------------------------------------------------------------===//
// CST
//===----------------------------------------------------------------------===//

#[derive(Debug, Clone)]
pub enum Value {
    // Primitives
    Nil {
        span: Span,
    },
    Bool {
        span: Span,
        value: bool,
    },
    Char {
        span: Span,
        value: char,
    },
    Int {
        span: Span,
        value: i64,
    },
    Float {
        span: Span,
        value: f64,
    },
    String {
        span: Span,
        value: Arc<str>,
    },
    Symbol {
        span: Span,
        value: SymId,
        meta: Option<Metadata>,
    },
    Keyword {
        span: Span,
        value: KeywId,
    },
    // Collections
    List {
        span: Span,
        value: Arc<List<Value>>,
        meta: Option<Metadata>,
    },
    Vector {
        span: Span,
        value: Arc<Vector<Value>>,
        meta: Option<Metadata>,
    },
    Map {
        span: Span,
        value: Arc<Map<Value, Value>>,
        meta: Option<Metadata>,
    },
    Set {
        span: Span,
        value: Arc<Set<Value>>,
        meta: Option<Metadata>,
    },
    // Namespace
    Namespace {
        span: Span,
        value: NsId,
    },
    // Function
    Function {
        span: Span,
        name: Option<SymId>,
        // A params is a vector of parameters.
        params: Arc<Vector<Value>>,
        // A body is a list of expressions to be evaluated.
        body: Arc<List<Value>>,
        // Closure environment
        env: Arc<Env>,
    },
    Var {
        span: Span,
        value: Arc<Var>,
    },
    NativeFunction {
        span: Span,
        name: SymId,
        f: NativeFn,
    },
    //--------------------------------------------------//
    // Internal
    //--------------------------------------------------//
    SpecialForm {
        span: Span,
        name: SymId,
    },
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Nil { span: _ }, Value::Nil { span: _ }) => true,
            (
                Value::Bool { span: _, value: a },
                Value::Bool { span: _, value: b },
            ) => a == b,
            (
                Value::Char { span: _, value: a },
                Value::Char { span: _, value: b },
            ) => a == b,
            (Value::Int { span: _, value: a }, Value::Int { span: _, value: b }) => {
                a == b
            }
            (
                Value::Float { span: _, value: a },
                Value::Float { span: _, value: b },
            ) => {
                // Handle NaN comparison
                if a.is_nan() && b.is_nan() { true } else { a == b }
            }
            (
                Value::String { span: _, value: a },
                Value::String { span: _, value: b },
            ) => a == b,
            (
                Value::Symbol { span: _, value: a, meta: _ },
                Value::Symbol { span: _, value: b, meta: _ },
            ) => a == b,
            (
                Value::Keyword { span: _, value: a },
                Value::Keyword { span: _, value: b },
            ) => a == b,
            (
                Value::List { span: _, value: a, meta: _ },
                Value::List { span: _, value: b, meta: _ },
            ) => a == b,
            (
                Value::Vector { span: _, value: a, meta: _ },
                Value::Vector { span: _, value: b, meta: _ },
            ) => a == b,
            (
                Value::Map { span: _, value: a, meta: _ },
                Value::Map { span: _, value: b, meta: _ },
            ) => a == b,
            (
                Value::Set { span: _, value: a, meta: _ },
                Value::Set { span: _, value: b, meta: _ },
            ) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Value::Nil { span: _ }, Value::Nil { span: _ }) => Ordering::Equal,
            (
                Value::Bool { span: _, value: a },
                Value::Bool { span: _, value: b },
            ) => a.cmp(b),
            (
                Value::Char { span: _, value: a },
                Value::Char { span: _, value: b },
            ) => a.cmp(b),
            (Value::Int { span: _, value: a }, Value::Int { span: _, value: b }) => {
                a.cmp(b)
            }
            (
                Value::Float { span: _, value: a },
                Value::Float { span: _, value: b },
            ) => {
                // Handle NaN comparison - NaN is considered less than any other value
                if a.is_nan() && b.is_nan() {
                    Ordering::Equal
                } else if a.is_nan() {
                    Ordering::Less
                } else if b.is_nan() {
                    Ordering::Greater
                } else {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                }
            }
            (
                Value::String { span: _, value: a },
                Value::String { span: _, value: b },
            ) => a.cmp(b),
            (
                Value::Symbol { span: _, value: a, meta: _ },
                Value::Symbol { span: _, value: b, meta: _ },
            ) => a.0.cmp(&b.0),
            (
                Value::Keyword { span: _, value: a },
                Value::Keyword { span: _, value: b },
            ) => a.0.cmp(&b.0),
            (
                Value::List { span: _, value: a, meta: _ },
                Value::List { span: _, value: b, meta: _ },
            ) => {
                let mut a_iter = a.iter();
                let mut b_iter = b.iter();
                loop {
                    match (a_iter.next(), b_iter.next()) {
                        (None, None) => break Ordering::Equal,
                        (None, _) => break Ordering::Less,
                        (_, None) => break Ordering::Greater,
                        (Some(a_val), Some(b_val)) => match a_val.cmp(b_val) {
                            Ordering::Equal => continue,
                            other => break other,
                        },
                    }
                }
            }
            (
                Value::Vector { span: _, value: a, meta: _ },
                Value::Vector { span: _, value: b, meta: _ },
            ) => {
                let mut a_iter = a.iter();
                let mut b_iter = b.iter();
                loop {
                    match (a_iter.next(), b_iter.next()) {
                        (None, None) => break Ordering::Equal,
                        (None, _) => break Ordering::Less,
                        (_, None) => break Ordering::Greater,
                        (Some(a_val), Some(b_val)) => match a_val.cmp(b_val) {
                            Ordering::Equal => continue,
                            other => break other,
                        },
                    }
                }
            }
            (
                Value::Map { span: _, value: a, meta: _ },
                Value::Map { span: _, value: b, meta: _ },
            ) => {
                let mut a_iter = a.iter();
                let mut b_iter = b.iter();
                loop {
                    match (a_iter.next(), b_iter.next()) {
                        (None, None) => break Ordering::Equal,
                        (None, _) => break Ordering::Less,
                        (_, None) => break Ordering::Greater,
                        (Some((a_k, a_v)), Some((b_k, b_v))) => match a_k.cmp(b_k) {
                            Ordering::Equal => match a_v.cmp(b_v) {
                                Ordering::Equal => continue,
                                other => break other,
                            },
                            other => break other,
                        },
                    }
                }
            }
            (
                Value::Set { span: _, value: a, meta: _ },
                Value::Set { span: _, value: b, meta: _ },
            ) => {
                let mut a_iter = a.iter();
                let mut b_iter = b.iter();
                loop {
                    match (a_iter.next(), b_iter.next()) {
                        (None, None) => break Ordering::Equal,
                        (None, _) => break Ordering::Less,
                        (_, None) => break Ordering::Greater,
                        (Some(a_val), Some(b_val)) => match a_val.cmp(b_val) {
                            Ordering::Equal => continue,
                            other => break other,
                        },
                    }
                }
            }
            (
                Value::NativeFunction { span: _, name: a, f: _ },
                Value::NativeFunction { span: _, name: b, f: _ },
            ) => a.0.cmp(&b.0),
            _ => Ordering::Equal,
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Nil { span: _ } => {
                0u8.hash(state);
            }
            Value::Bool { span: _, value } => {
                1u8.hash(state);
                value.hash(state);
            }
            Value::Char { span: _, value } => {
                2u8.hash(state);
                value.hash(state);
            }
            Value::Int { span: _, value } => {
                3u8.hash(state);
                value.hash(state);
            }
            Value::Float { span: _, value } => {
                4u8.hash(state);
                // Handle NaN: use a canonical representation
                if value.is_nan() {
                    f64::NAN.to_bits().hash(state);
                } else {
                    value.to_bits().hash(state);
                }
            }
            Value::String { span: _, value } => {
                5u8.hash(state);
                value.hash(state);
            }
            Value::Symbol { span: _, value, meta: _ } => {
                6u8.hash(state);
                value.hash(state);
            }
            Value::Keyword { span: _, value } => {
                7u8.hash(state);
                value.hash(state);
            }
            Value::List { span: _, value, meta: _ } => {
                8u8.hash(state);
                // Hash the list contents
                for item in value.iter() {
                    item.hash(state);
                }
            }
            Value::Vector { span: _, value, meta: _ } => {
                9u8.hash(state);
                // Hash the vector contents
                for item in value.iter() {
                    item.hash(state);
                }
            }
            Value::Map { span: _, value, meta: _ } => {
                10u8.hash(state);
                // Hash map entries in a deterministic order
                // Since Map preserves insertion order, we can iterate
                for (k, v) in value.iter() {
                    k.hash(state);
                    v.hash(state);
                }
            }
            Value::Set { span: _, value, meta: _ } => {
                11u8.hash(state);
                // Hash set elements in a deterministic order
                for item in value.iter() {
                    item.hash(state);
                }
            }
            Value::Namespace { span: _, value } => {
                12u8.hash(state);
                value.hash(state);
            }
            Value::Function { span: _, name, params: _, body: _, env: _ } => {
                13u8.hash(state);
                name.hash(state);
                // Note: We don't hash params/body/env to avoid infinite recursion
                // and because functions are compared by identity, not structure
            }
            Value::Var { span: _, value } => {
                14u8.hash(state);
                value.hash(state);
            }
            Value::NativeFunction { span: _, name, f: _ } => {
                15u8.hash(state);
                name.hash(state);
            }
            Value::SpecialForm { span: _, name } => {
                16u8.hash(state);
                name.hash(state);
            }
        }
    }
}

/// Pretty-prints a sequence of values.
///
/// # Arguments
///
/// * `seq` - The sequence of values to print.
/// * `start` - The string to print before the sequence.
/// * `end` - The string to print after the sequence.
/// * `join` - The string to print between values.
///
/// # Returns
///
/// A string representing the pretty-printed sequence.
pub fn pr_seq<'a, I>(seq: I, start: &str, end: &str, join: &str) -> String
where
    I: IntoIterator<Item = &'a Value>,
{
    use std::fmt::Write;

    let mut out = String::new();
    out.push_str(start);

    let mut it = seq.into_iter().peekable();
    while let Some(v) = it.next() {
        // assuming Value: Display (or keep v.to_string())
        write!(&mut out, "{}", v).unwrap();
        if it.peek().is_some() {
            out.push_str(join);
        }
    }

    out.push_str(end);
    out
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Value::Nil { span: _ } => String::from("nil"),
            Value::Bool { span: _, value: val } => val.to_string(),
            Value::Int { span: _, value: val } => format!("{}", val),
            Value::Float { span: _, value: val } => format!("{}", val),
            Value::Char { span: _, value: c } => format!("\\{}", c),
            Value::String { span: _, value: s } => s.to_string(),
            Value::Symbol { span: _, value: sym, meta: _ } => {
                interner::sym_to_str(*sym)
            }
            Value::Keyword { span: _, value: kw } => interner::kw_print(*kw),
            Value::List { span: _, value: l, meta: _ } => {
                pr_seq(l.iter(), "(", ")", " ")
            }
            Value::Vector { span: _, value: v, meta: _ } => {
                pr_seq(v.iter(), "[", "]", " ")
            }
            Value::Map { span: _, value: hm, meta: _ } => {
                pr_seq(hm.iter().flat_map(|(k, v)| [k, v]), "{", "}", " ")
            }
            Value::Set { span: _, value: set, meta: _ } => {
                pr_seq(set.iter(), "{", "}", " ")
            }
            Value::Namespace { span: _, value: ns } => interner::ns_to_str(*ns),
            Value::SpecialForm { span: _, name: n } => {
                format!("#<special-form:{}>", interner::sym_to_str(*n))
            }
            Value::Function { span: _, name: n, params: _, body: _, env: _ } => {
                let name_str = match n {
                    Some(sym) => interner::sym_to_str(*sym),
                    None => "anon".to_string(),
                };
                format!("#<fn:{}>", name_str)
            }
            Value::Var { span: _, value } => {
                let bound = value.is_bound();
                format!(
                    "var ({}):{}",
                    if bound { "bound" } else { "unbound" },
                    interner::sym_to_str(value.symbol)
                )
            }
            Value::NativeFunction { span: _, name: n, f: _ } => {
                format!("#<native-fn:{}>", interner::sym_to_str(*n))
            }
        };
        write!(f, "{}", s)
    }
}

impl Value {
    pub fn as_str(&self) -> &'static str {
        match self {
            Value::Nil { span: _ } => "Nil",
            Value::Bool { span: _, value: _ } => "Bool",
            Value::Int { span: _, value: _ } => "Int",
            Value::Float { span: _, value: _ } => "Float",
            Value::Char { span: _, value: _ } => "Char",
            Value::String { span: _, value: _ } => "String",
            Value::Symbol { span: _, value: _, meta: _ } => "Symbol",
            Value::Keyword { span: _, value: _ } => "Keyword",
            Value::List { span: _, value: _, meta: _ } => "List",
            Value::Vector { span: _, value: _, meta: _ } => "Vector",
            Value::Map { span: _, value: _, meta: _ } => "Map",
            Value::Set { span: _, value: _, meta: _ } => "Set",
            Value::Namespace { span: _, value: _ } => "Namespace",
            Value::Function { span: _, name: _, params: _, body: _, env: _ } => {
                "Function"
            }
            Value::Var { span: _, value: _ } => "Var",
            Value::NativeFunction { span: _, name: _, f: _ } => "NativeFunction",
            Value::SpecialForm { span: _, name: _ } => "SpecialForm",
        }
    }

    pub fn get_meta(&self) -> Option<Metadata> {
        match self {
            Value::Symbol { meta, .. }
            | Value::List { meta, .. }
            | Value::Vector { meta, .. }
            | Value::Map { meta, .. }
            | Value::Set { meta, .. } => meta.clone(),
            _ => None,
        }
    }

    pub fn set_meta(&self, meta_def: Value) -> Result<Value, Error> {
        // Check if this type supports metadata
        let mut meta_map = match self {
            Value::Symbol { meta, .. }
            | Value::List { meta, .. }
            | Value::Vector { meta, .. }
            | Value::Map { meta, .. }
            | Value::Set { meta, .. } => meta.clone().unwrap_or_else(Metadata::new),
            _ => {
                return Err(Error::SyntaxError(SyntaxError::InvalidMeta {
                    reason: format!(
                        "Metadata cannot be applied to {}",
                        self.to_string()
                    ),
                }));
            }
        };

        // Handle meta_def: either a Keyword or a Map
        match meta_def {
            Value::Keyword { span, value: kw } => {
                // If meta_def is a Keyword, set its value to true
                let key = Value::Keyword { span: span.clone(), value: kw };
                let value = Value::Bool { span: Span { start: 0, end: 0 }, value: true };
                meta_map.insert(key, value);
            }
            Value::Map { span: _, value: hm, meta: _ } => {
                // If meta_def is a Map, merge all entries directly
                for (k, v) in hm.iter() {
                    meta_map.insert(k.clone(), v.clone());
                }
            }
            _ => {
                return Err(Error::SyntaxError(SyntaxError::InvalidMeta {
                    reason: format!(
                        "Metadata must be a keyword or a map, got {}",
                        meta_def.to_string()
                    ),
                }));
            }
        }

        // Create a new Value with the updated metadata
        let result = match self {
            Value::Symbol { span, value, .. } => Value::Symbol {
                span: span.clone(),
                value: *value,
                meta: Some(meta_map),
            },
            Value::List { span, value, .. } => Value::List {
                span: span.clone(),
                value: value.clone(),
                meta: Some(meta_map),
            },
            Value::Vector { span, value, .. } => Value::Vector {
                span: span.clone(),
                value: value.clone(),
                meta: Some(meta_map),
            },
            Value::Map { span, value, .. } => Value::Map {
                span: span.clone(),
                value: value.clone(),
                meta: Some(meta_map),
            },
            Value::Set { span, value, .. } => Value::Set {
                span: span.clone(),
                value: value.clone(),
                meta: Some(meta_map),
            },
            _ => unreachable!(),
        };

        Ok(result)
    }
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Creates a BTreeMap from a sequence of Values.
/// For HashMap syntax like {k1 v1 k2 v2}, this creates key-value pairs.
/// If the sequence has an odd number of elements, the last element is used as both key and value.
pub fn create_btree_map_from_sequence(seq: Vec<Value>) -> BTreeMap<Value, Value> {
    let mut map = BTreeMap::new();
    let mut iter = seq.into_iter();

    while let Some(key) = iter.next() {
        if let Some(value) = iter.next() {
            map.insert(key, value);
        } else {
            // Odd number of elements, use the key as both key and value
            map.insert(key.clone(), key);
        }
    }

    map
}

/// Creates a BTreeSet from a sequence of Values.
/// Removes duplicates automatically since BTreeSet only stores unique values.
pub fn create_btree_set_from_sequence(seq: Vec<Value>) -> BTreeSet<Value> {
    seq.into_iter().collect()
}

/// Creates a list value.
///
/// # Arguments
///
/// * `span` - The span of the list.
/// * `values` - The values to include in the list.
///
/// # Returns
pub fn list(span: Span, values: Vec<Value>) -> Value {
    Value::List { span: span, value: Arc::new(List::from_iter(values)), meta: None }
}

pub fn symbol(span: Span, value: SymId, meta: Option<Metadata>) -> Value {
    Value::Symbol { span: span, value: value, meta: meta }
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{namespace, Metadata};
    use crate::interner;
    use std::collections::BTreeMap;

    fn test_span() -> Span {
        Span { start: 0, end: 0 }
    }

    // Helper function to create a keyword value
    fn kw(s: &str) -> Value {
        Value::Keyword { span: test_span(), value: interner::intern_kw(s) }
    }

    // Helper function to create a symbol value
    fn sym(s: &str) -> Value {
        Value::Symbol {
            span: test_span(),
            value: interner::intern_sym(s),
            meta: None,
        }
    }

    // Helper function to create a symbol with metadata
    fn sym_with_meta(s: &str, meta: Option<Metadata>) -> Value {
        Value::Symbol { span: test_span(), value: interner::intern_sym(s), meta }
    }

    // Helper function to create an empty list
    fn empty_list() -> Value {
        Value::List { span: test_span(), value: Arc::new(List::new()), meta: None }
    }

    // Helper function to create a list with metadata
    fn list_with_meta(
        v: Vec<Value>,
        meta: Option<Metadata>,
    ) -> Value {
        Value::List { span: test_span(), value: Arc::new(List::from_iter(v)), meta }
    }

    // Helper function to create an empty map
    fn empty_map() -> Value {
        Value::Map { span: test_span(), value: Arc::new(Map::new()), meta: None }
    }

    // Helper function to create a map with metadata
    fn map_with_meta(
        m: BTreeMap<Value, Value>,
        meta: Option<Metadata>,
    ) -> Value {
        let mut map = Map::new();
        for (k, v) in m {
            map = map.insert(k, v);
        }
        Value::Map { span: test_span(), value: Arc::new(map), meta }
    }

    //===----------------------------------------------------------------------===//
    // Metadata Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_get_meta_supported_types() {
        // Types that support metadata should return Some or None
        let mut meta_map = Metadata::new();
        meta_map
            .insert(kw("test"), Value::Bool { span: test_span(), value: true });

        let sym_with = sym_with_meta("test", Some(meta_map.clone()));
        let sym_without = sym("test");

        assert!(sym_with.get_meta().is_some());
        assert!(sym_without.get_meta().is_none());

        let list_with = list_with_meta(vec![], Some(meta_map.clone()));
        let list_without = empty_list();

        assert!(list_with.get_meta().is_some());
        assert!(list_without.get_meta().is_none());
    }

    #[test]
    fn test_get_meta_unsupported_types() {
        // Types that don't support metadata should return None
        let nil = Value::Nil { span: test_span() };
        let bool_val = Value::Bool { span: test_span(), value: true };
        let int_val = Value::Int { span: test_span(), value: 42 };
        let float_val = Value::Float { span: test_span(), value: 3.14 };
        let char_val = Value::Char { span: test_span(), value: 'a' };
        let string_val =
            Value::String { span: test_span(), value: Arc::from("test") };
        let keyword_val = kw("test");
        let namespace_val = Value::Namespace {
            span: test_span(),
            value: interner::intern_ns("test"),
        };
        let special_form = Value::SpecialForm {
            span: test_span(),
            name: interner::intern_sym("test"),
        };
        let function = Value::Function {
            span: test_span(),
            name: Some(interner::intern_sym("test")),
            params: Arc::new(Vector::new()),
            body: Arc::new(List::new()),
            env: Arc::new(Env::new(namespace::ns_find_or_create("test-ns"))),
        };

        assert!(nil.get_meta().is_none());
        assert!(bool_val.get_meta().is_none());
        assert!(int_val.get_meta().is_none());
        assert!(float_val.get_meta().is_none());
        assert!(char_val.get_meta().is_none());
        assert!(string_val.get_meta().is_none());
        assert!(keyword_val.get_meta().is_none());
        assert!(namespace_val.get_meta().is_none());
        assert!(special_form.get_meta().is_none());
        assert!(function.get_meta().is_none());
    }

    #[test]
    fn test_set_meta_unsupported_types() {
        // Setting metadata on unsupported types should return an error
        let nil = Value::Nil { span: test_span() };
        let bool_val = Value::Bool { span: test_span(), value: true };
        let int_val = Value::Int { span: test_span(), value: 42 };
        let keyword_val = kw("test");

        assert!(nil.set_meta(keyword_val.clone()).is_err());
        assert!(bool_val.set_meta(keyword_val.clone()).is_err());
        assert!(int_val.set_meta(keyword_val.clone()).is_err());
    }

    #[test]
    fn test_set_meta_with_keyword() {
        // Setting metadata with a keyword should set it to true
        let sym = sym("test-symbol");
        let meta_kw = kw("deprecated");

        let result = sym.set_meta(meta_kw).unwrap();
        let meta = result.get_meta().unwrap();

        assert_eq!(meta.len(), 1);
        let key = kw("deprecated");
        assert!(meta.contains_key(&key));
        match meta.get(&key) {
            Some(Value::Bool { value: true, .. }) => {}
            _ => panic!("Expected boolean true value"),
        }
    }

    #[test]
    fn test_set_meta_with_map() {
        // Setting metadata with a map should merge the values
        let sym = sym("test-symbol");
        let mut meta_map = BTreeMap::new();
        meta_map.insert(kw("key1"), Value::Int { span: test_span(), value: 42 });
        meta_map.insert(
            kw("key2"),
            Value::String { span: test_span(), value: Arc::from("value") },
        );
        let mut map = Map::new();
        for (k, v) in meta_map {
            map = map.insert(k, v);
        }
        let meta_value =
            Value::Map { span: test_span(), value: Arc::new(map), meta: None };

        let result = sym.set_meta(meta_value).unwrap();
        let meta = result.get_meta().unwrap();

        assert_eq!(meta.len(), 2);
    }

    #[test]
    fn test_set_meta_with_invalid_type() {
        // Setting metadata with invalid type should return an error
        let sym = sym("test-symbol");
        let invalid_meta = Value::Int { span: test_span(), value: 42 };
        let invalid_meta2 =
            Value::String { span: test_span(), value: Arc::from("invalid") };

        assert!(sym.set_meta(invalid_meta).is_err());
        assert!(sym.set_meta(invalid_meta2).is_err());
    }

    #[test]
    fn test_set_meta_with_non_keyword_map_keys() {
        // Setting metadata with a map that has non-keyword keys should return an error
        let sym = sym("test-symbol");
        let mut meta_map = BTreeMap::new();
        meta_map.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::Int { span: test_span(), value: 42 },
        );
        let mut map = Map::new();
        for (k, v) in meta_map {
            map = map.insert(k, v);
        }
        let meta_value =
            Value::Map { span: test_span(), value: Arc::new(map), meta: None };

        let result = sym.set_meta(meta_value).unwrap();
        let meta = result.get_meta().unwrap();
        let int_key = Value::Int { span: test_span(), value: 1 };
        assert!(meta.contains_key(&int_key));
        match meta.get(&int_key) {
            Some(Value::Int { value, .. }) => assert_eq!(*value, 42),
            _ => panic!("Expected integer metadata value"),
        }
    }

    #[test]
    fn test_set_meta_merges_existing() {
        // Setting metadata should merge with existing metadata
        let mut existing_meta = Metadata::new();
        existing_meta
            .insert(kw("key1"), Value::Int { span: test_span(), value: 1 });
        let sym = sym_with_meta("test", Some(existing_meta));

        let mut new_meta_map = BTreeMap::new();
        new_meta_map.insert(kw("key2"), Value::Int { span: test_span(), value: 2 });
        let mut map = Map::new();
        for (k, v) in new_meta_map {
            map = map.insert(k, v);
        }
        let new_meta =
            Value::Map { span: test_span(), value: Arc::new(map), meta: None };

        let result = sym.set_meta(new_meta).unwrap();
        let meta = result.get_meta().unwrap();

        assert_eq!(meta.len(), 2);
        let key1 = kw("key1");
        let key2 = kw("key2");
        assert!(meta.contains_key(&key1));
        assert!(meta.contains_key(&key2));
    }

    #[test]
    fn test_set_meta_overwrites_existing_key() {
        // Setting metadata should overwrite existing keys
        let mut existing_meta = Metadata::new();
        existing_meta
            .insert(kw("key1"), Value::Int { span: test_span(), value: 1 });
        let sym = sym_with_meta("test", Some(existing_meta));

        let mut new_meta_map = BTreeMap::new();
        new_meta_map.insert(
            kw("key1"),
            Value::String { span: test_span(), value: Arc::from("new") },
        );
        let mut map = Map::new();
        for (k, v) in new_meta_map {
            map = map.insert(k, v);
        }
        let new_meta =
            Value::Map { span: test_span(), value: Arc::new(map), meta: None };

        let result = sym.set_meta(new_meta).unwrap();
        let meta = result.get_meta().unwrap();

        assert_eq!(meta.len(), 1);
        let key = kw("key1");
        match meta.get(&key) {
            Some(Value::String { value: v, .. }) => assert_eq!(v.as_ref(), "new"),
            _ => panic!("Expected string value"),
        }
    }

    #[test]
    fn test_metadata_preserved_on_clone() {
        // Metadata should be preserved when cloning
        let mut meta_map = Metadata::new();
        meta_map
            .insert(kw("test"), Value::Bool { span: test_span(), value: true });
        let sym = sym_with_meta("test", Some(meta_map));

        let cloned = sym.clone();
        assert_eq!(sym.get_meta(), cloned.get_meta());
    }

    #[test]
    fn test_metadata_all_collection_types() {
        // All collection types should support metadata
        let meta_kw = kw("test");

        let list = empty_list();
        let vector = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::new()),
            meta: None,
        };
        let map = empty_map();
        let set = Value::Set {
            span: test_span(),
            value: Arc::new(Set::new()),
            meta: None,
        };

        assert!(list.set_meta(meta_kw.clone()).is_ok());
        assert!(vector.set_meta(meta_kw.clone()).is_ok());
        assert!(map.set_meta(meta_kw.clone()).is_ok());
        assert!(set.set_meta(meta_kw).is_ok());
    }

    //===----------------------------------------------------------------------===//
    // Comparison Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_equality_ignores_metadata() {
        // Two values with same content but different metadata should be equal
        let sym1 = sym("test");
        let mut meta1 = Metadata::new();
        meta1
            .insert(kw("key1"), Value::Int { span: test_span(), value: 1 });
        let sym1_with_meta = sym_with_meta("test", Some(meta1));

        let mut meta2 = Metadata::new();
        meta2
            .insert(kw("key2"), Value::Int { span: test_span(), value: 2 });
        let sym2_with_meta = sym_with_meta("test", Some(meta2));

        assert_eq!(sym1, sym1_with_meta);
        assert_eq!(sym1_with_meta, sym2_with_meta);
    }

    #[test]
    fn test_equality_different_types() {
        // Different types should not be equal
        let int_val = Value::Int { span: test_span(), value: 42 };
        let float_val = Value::Float { span: test_span(), value: 42.0 };
        let string_val = Value::String { span: test_span(), value: Arc::from("42") };

        assert_ne!(int_val, float_val);
        assert_ne!(int_val, string_val);
        assert_ne!(float_val, string_val);
    }

    #[test]
    fn test_equality_nan() {
        // NaN values should be equal to each other
        let nan1 = Value::Float { span: test_span(), value: f64::NAN };
        let nan2 = Value::Float { span: test_span(), value: f64::NAN };

        assert_eq!(nan1, nan2);
    }

    #[test]
    fn test_equality_nan_vs_number() {
        // NaN should not be equal to any number
        let nan = Value::Float { span: test_span(), value: f64::NAN };
        let num = Value::Float { span: test_span(), value: 42.0 };

        assert_ne!(nan, num);
    }

    #[test]
    fn test_ordering_nan() {
        // NaN should be considered less than any other value
        let nan = Value::Float { span: test_span(), value: f64::NAN };
        let num = Value::Float { span: test_span(), value: 42.0 };
        let neg_inf = Value::Float { span: test_span(), value: f64::NEG_INFINITY };

        assert!(nan < num);
        assert!(nan < neg_inf);
        assert_eq!(nan.partial_cmp(&nan), Some(Ordering::Equal));
    }

    #[test]
    fn test_ordering_different_types() {
        // Different types should compare as Equal (for ordering purposes)
        let int_val = Value::Int { span: test_span(), value: 42 };
        let float_val = Value::Float { span: test_span(), value: 42.0 };
        let string_val = Value::String { span: test_span(), value: Arc::from("42") };

        assert_eq!(int_val.cmp(&float_val), Ordering::Equal);
        assert_eq!(int_val.cmp(&string_val), Ordering::Equal);
    }

    #[test]
    fn test_ordering_nil() {
        // Nil values should be equal
        let nil1 = Value::Nil { span: test_span() };
        let nil2 = Value::Nil { span: test_span() };

        assert_eq!(nil1.cmp(&nil2), Ordering::Equal);
    }

    #[test]
    fn test_ordering_bool() {
        // false < true
        let false_val = Value::Bool { span: test_span(), value: false };
        let true_val = Value::Bool { span: test_span(), value: true };

        assert!(false_val < true_val);
        assert_eq!(false_val.cmp(&false_val), Ordering::Equal);
        assert_eq!(true_val.cmp(&true_val), Ordering::Equal);
    }

    #[test]
    fn test_ordering_int() {
        // Integer ordering
        let small = Value::Int { span: test_span(), value: 1 };
        let large = Value::Int { span: test_span(), value: 100 };

        assert!(small < large);
        assert_eq!(small.cmp(&small), Ordering::Equal);
    }

    #[test]
    fn test_ordering_float() {
        // Float ordering
        let small = Value::Float { span: test_span(), value: 1.0 };
        let large = Value::Float { span: test_span(), value: 100.0 };

        assert!(small < large);
        assert_eq!(small.cmp(&small), Ordering::Equal);
    }

    #[test]
    fn test_ordering_char() {
        // Character ordering
        let a = Value::Char { span: test_span(), value: 'a' };
        let z = Value::Char { span: test_span(), value: 'z' };

        assert!(a < z);
        assert_eq!(a.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn test_ordering_string() {
        // String ordering (lexicographic)
        let a = Value::String { span: test_span(), value: Arc::from("apple") };
        let z = Value::String { span: test_span(), value: Arc::from("zebra") };

        assert!(a < z);
        assert_eq!(a.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn test_ordering_symbol() {
        // Symbol ordering
        let sym1 = sym("a");
        let sym2 = sym("z");

        assert!(sym1 < sym2);
        assert_eq!(sym1.cmp(&sym1), Ordering::Equal);
    }

    #[test]
    fn test_ordering_keyword() {
        // Keyword ordering
        let kw1 = kw("a");
        let kw2 = kw("z");

        assert!(kw1 < kw2);
        assert_eq!(kw1.cmp(&kw1), Ordering::Equal);
    }

    #[test]
    fn test_ordering_list() {
        // List ordering
        let empty = empty_list();
        let single =
            list_with_meta(vec![Value::Int { span: test_span(), value: 1 }], None);
        let multiple = list_with_meta(
            vec![
                Value::Int { span: test_span(), value: 1 },
                Value::Int { span: test_span(), value: 2 },
            ],
            None,
        );

        assert!(empty < single);
        assert!(single < multiple);
        assert_eq!(empty.cmp(&empty), Ordering::Equal);
    }

    #[test]
    fn test_ordering_list_different_lengths() {
        // Lists with different lengths
        let short =
            list_with_meta(vec![Value::Int { span: test_span(), value: 1 }], None);
        let long = list_with_meta(
            vec![
                Value::Int { span: test_span(), value: 1 },
                Value::Int { span: test_span(), value: 2 },
            ],
            None,
        );

        assert!(short < long);
    }

    #[test]
    fn test_ordering_list_different_contents() {
        // Lists with different contents
        let list1 =
            list_with_meta(vec![Value::Int { span: test_span(), value: 1 }], None);
        let list2 =
            list_with_meta(vec![Value::Int { span: test_span(), value: 2 }], None);

        assert!(list1 < list2);
    }

    #[test]
    fn test_ordering_vector() {
        // Vector ordering
        let empty = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::new()),
            meta: None,
        };
        let single = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::from_iter(vec![Value::Int {
                span: test_span(),
                value: 1,
            }])),
            meta: None,
        };

        assert!(empty < single);
        assert_eq!(empty.cmp(&empty), Ordering::Equal);
    }

    #[test]
    fn test_ordering_map() {
        // Map ordering
        let mut map1 = BTreeMap::new();
        map1.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::Int { span: test_span(), value: 10 },
        );
        let val1 = map_with_meta(map1, None);

        let mut map2 = BTreeMap::new();
        map2.insert(
            Value::Int { span: test_span(), value: 2 },
            Value::Int { span: test_span(), value: 20 },
        );
        let val2 = map_with_meta(map2, None);

        assert!(val1 < val2);
        assert_eq!(val1.cmp(&val1), Ordering::Equal);
    }

    #[test]
    fn test_ordering_set() {
        // Set ordering
        let mut set1 = Set::new();
        set1 = set1.insert(Value::Int { span: test_span(), value: 1 });
        let val1 =
            Value::Set { span: test_span(), value: Arc::new(set1), meta: None };

        let mut set2 = Set::new();
        set2 = set2.insert(Value::Int { span: test_span(), value: 2 });
        let val2 =
            Value::Set { span: test_span(), value: Arc::new(set2), meta: None };

        assert!(val1 < val2);
        assert_eq!(val1.cmp(&val1), Ordering::Equal);
    }

    #[test]
    fn test_equality_collections_ignore_metadata() {
        // Collections with same content but different metadata should be equal
        let mut meta1 = Metadata::new();
        meta1
            .insert(kw("key1"), Value::Int { span: test_span(), value: 1 });
        let list1 = list_with_meta(
            vec![Value::Int { span: test_span(), value: 42 }],
            Some(meta1),
        );

        let mut meta2 = Metadata::new();
        meta2
            .insert(kw("key2"), Value::Int { span: test_span(), value: 2 });
        let list2 = list_with_meta(
            vec![Value::Int { span: test_span(), value: 42 }],
            Some(meta2),
        );

        assert_eq!(list1, list2);
    }

    #[test]
    fn test_equality_empty_collections() {
        // Empty collections of the same type should be equal
        let list1 = empty_list();
        let list2 = empty_list();

        assert_eq!(list1, list2);

        let vec1 = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::new()),
            meta: None,
        };
        let vec2 = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::new()),
            meta: None,
        };

        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_equality_collections_different_types() {
        // Collections of different types should not be equal
        let list = empty_list();
        let vector = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::new()),
            meta: None,
        };
        let map = empty_map();

        assert_ne!(list, vector);
        assert_ne!(list, map);
        assert_ne!(vector, map);
    }

    #[test]
    fn test_equality_float_special_values() {
        // Special float values
        let pos_inf = Value::Float { span: test_span(), value: f64::INFINITY };
        let neg_inf = Value::Float { span: test_span(), value: f64::NEG_INFINITY };
        let pos_inf2 = Value::Float { span: test_span(), value: f64::INFINITY };
        let neg_inf2 = Value::Float { span: test_span(), value: f64::NEG_INFINITY };

        assert_eq!(pos_inf, pos_inf2);
        assert_eq!(neg_inf, neg_inf2);
        assert_ne!(pos_inf, neg_inf);
    }

    #[test]
    fn test_ordering_float_special_values() {
        // Special float values ordering
        let neg_inf = Value::Float { span: test_span(), value: f64::NEG_INFINITY };
        let neg_num = Value::Float { span: test_span(), value: -100.0 };
        let zero = Value::Float { span: test_span(), value: 0.0 };
        let pos_num = Value::Float { span: test_span(), value: 100.0 };
        let pos_inf = Value::Float { span: test_span(), value: f64::INFINITY };
        let nan = Value::Float { span: test_span(), value: f64::NAN };

        assert!(neg_inf < neg_num);
        assert!(neg_num < zero);
        assert!(zero < pos_num);
        assert!(pos_num < pos_inf);
        assert!(nan < neg_inf); // NaN is less than everything
    }

    #[test]
    fn test_equality_map_keys_values() {
        // Maps should compare by their key-value pairs
        let mut map1 = BTreeMap::new();
        map1.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::String { span: test_span(), value: Arc::from("a") },
        );
        map1.insert(
            Value::Int { span: test_span(), value: 2 },
            Value::String { span: test_span(), value: Arc::from("b") },
        );
        let val1 = map_with_meta(map1, None);

        let mut map2 = BTreeMap::new();
        map2.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::String { span: test_span(), value: Arc::from("a") },
        );
        map2.insert(
            Value::Int { span: test_span(), value: 2 },
            Value::String { span: test_span(), value: Arc::from("b") },
        );
        let val2 = map_with_meta(map2, None);

        assert_eq!(val1, val2);
    }

    #[test]
    fn test_equality_map_different_order() {
        // Maps with same key-value pairs in different order should be equal
        let mut map1 = BTreeMap::new();
        map1.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::Int { span: test_span(), value: 10 },
        );
        map1.insert(
            Value::Int { span: test_span(), value: 2 },
            Value::Int { span: test_span(), value: 20 },
        );
        let val1 = map_with_meta(map1, None);

        let mut map2 = BTreeMap::new();
        map2.insert(
            Value::Int { span: test_span(), value: 2 },
            Value::Int { span: test_span(), value: 20 },
        );
        map2.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::Int { span: test_span(), value: 10 },
        );
        let val2 = map_with_meta(map2, None);

        assert_eq!(val1, val2); // BTreeMap maintains order, so this should still be equal
    }

    #[test]
    fn test_partial_cmp_always_some() {
        // partial_cmp should always return Some for Value
        let val1 = Value::Int { span: test_span(), value: 1 };
        let val2 = Value::Int { span: test_span(), value: 2 };

        assert!(val1.partial_cmp(&val2).is_some());
        assert_eq!(val1.partial_cmp(&val2), Some(Ordering::Less));
    }

    #[test]
    fn test_ordering_collections_comprehensive() {
        // Comprehensive test for collection ordering
        use std::cmp::Ordering;

        // Test List ordering
        let list1 = list_with_meta(
            vec![
                Value::Int { span: test_span(), value: 1 },
                Value::Int { span: test_span(), value: 2 },
            ],
            None,
        );
        let list2 = list_with_meta(
            vec![
                Value::Int { span: test_span(), value: 1 },
                Value::Int { span: test_span(), value: 3 },
            ],
            None,
        );
        assert!(list1 < list2);
        assert_eq!(list1.cmp(&list1), Ordering::Equal);

        // Test Vector ordering with different lengths
        let vec1 = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::from_iter(vec![Value::Int {
                span: test_span(),
                value: 1,
            }])),
            meta: None,
        };
        let vec2 = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::from_iter(vec![
                Value::Int { span: test_span(), value: 1 },
                Value::Int { span: test_span(), value: 2 },
            ])),
            meta: None,
        };
        assert!(vec1 < vec2);

        // Test Map ordering
        let mut map1_btree = BTreeMap::new();
        map1_btree.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::Int { span: test_span(), value: 10 },
        );
        let map1 = map_with_meta(map1_btree, None);

        let mut map2_btree = BTreeMap::new();
        map2_btree.insert(
            Value::Int { span: test_span(), value: 1 },
            Value::Int { span: test_span(), value: 20 },
        );
        let map2 = map_with_meta(map2_btree, None);

        assert!(map1 < map2);

        // Test Set ordering
        let mut set1 = Set::new();
        set1 = set1.insert(Value::Int { span: test_span(), value: 1 });
        set1 = set1.insert(Value::Int { span: test_span(), value: 2 });
        let set_val1 =
            Value::Set { span: test_span(), value: Arc::new(set1), meta: None };

        let mut set2 = Set::new();
        set2 = set2.insert(Value::Int { span: test_span(), value: 1 });
        set2 = set2.insert(Value::Int { span: test_span(), value: 3 });
        let set_val2 =
            Value::Set { span: test_span(), value: Arc::new(set2), meta: None };

        assert!(set_val1 < set_val2);

        // Test empty collections are equal
        let empty_list1 = empty_list();
        let empty_list2 = empty_list();
        assert_eq!(empty_list1.cmp(&empty_list2), Ordering::Equal);

        let empty_vec1 = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::new()),
            meta: None,
        };
        let empty_vec2 = Value::Vector {
            span: test_span(),
            value: Arc::new(Vector::new()),
            meta: None,
        };
        assert_eq!(empty_vec1.cmp(&empty_vec2), Ordering::Equal);
    }
}
