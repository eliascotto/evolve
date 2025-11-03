use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone)]
pub enum Value {
    Nil,
    Bool(bool),

    Char(char),
    Int(i64),
    Float(f64),

    String(String),

    Symbol(String),
    // Keywords are used to identify specific values in a collection
    // They are NOT case-sensitive
    Keyword(String),

    // Collections
    List(Vec<Value>),
    Vector(Vec<Value>),
    Map(BTreeMap<Value, Value>),
    Set(BTreeSet<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => {
                // Handle NaN comparison
                if a.is_nan() && b.is_nan() { true } else { a == b }
            }
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::Keyword(a), Value::Keyword(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Vector(a), Value::Vector(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => a == b,
            (Value::Set(a), Value::Set(b)) => a == b,
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
            (Value::Nil, Value::Nil) => Ordering::Equal,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Char(a), Value::Char(b)) => a.cmp(b),
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => {
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
            (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::Symbol(a), Value::Symbol(b)) => a.cmp(b),
            (Value::Keyword(a), Value::Keyword(b)) => a.cmp(b),
            (Value::List(a), Value::List(b)) => a.cmp(b),
            (Value::Vector(a), Value::Vector(b)) => a.cmp(b),
            (Value::Map(a), Value::Map(b)) => a.cmp(b),
            (Value::Set(a), Value::Set(b)) => a.cmp(b),
            // Different types are ordered by their discriminant
            (a, b) => {
                let a_disc = std::mem::discriminant(a);
                let b_disc = std::mem::discriminant(b);
                if a_disc == b_disc {
                    Ordering::Equal
                } else {
                    // Use a simple ordering based on discriminant values
                    // This provides a consistent ordering for different types
                    format!("{:?}", a_disc).cmp(&format!("{:?}", b_disc))
                }
            }
        }
    }
}

pub fn pr_seq(seq: &Vec<Value>, start: &str, end: &str, join: &str) -> String {
    let strs: Vec<String> = seq.iter().map(|x| x.to_string()).collect();
    format!("{}{}{}", start, strs.join(join), end)
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Value::Nil => String::from("nil"),
            Value::Bool(val) => val.to_string(),
            Value::Int(val) => format!("{}", val),
            Value::Float(val) => format!("{}", val),
            Value::Char(c) => format!("\\{}", c),
            Value::String(s) => s.clone(),
            Value::Symbol(sym) => sym.to_string(),
            Value::Keyword(kw) => kw.to_string(),
            Value::List(l) => pr_seq(&l, "(", ")", " "),
            Value::Vector(v) => pr_seq(&v, "[", "]", " "),
            Value::Map(hm) => {
                let l = hm.iter().flat_map(|(k, v)| vec![k.clone(), v.clone()]).collect();
                pr_seq(&l, "{", "}", " ")
            }
            Value::Set(set) => {
                let strs: Vec<String> = set.iter().map(|x| x.to_string()).collect();
                format!("{{{}}}", strs.join(" "))
            } // Value::Func(f, _) => format!("#<fn {:?}>", f),
              // Value::Macro(f, _) => format!("#<macro {:?}>", f),
              // Value::Lambda { ast, params, .. } => {
              //     let mut overloads = vec![];
              //     for idx in 0..params.len() {
              //         overloads.push(format!(
              //             "({} {})",
              //             params[idx].to_string(),
              //             ast[idx].to_string()
              //         ))
              //     }
              //     format!("(fn {})", overloads.join("\n"))
              // }
              // Value::Namespace(ns) => ns.borrow().to_string(),
              // Value::Atom(a) => format!("#atom {{:val {}}}", a.borrow().to_string()),
              // Value::Error(err) => format_error(err.clone()),
              // Value::Recur(l) => format!("#<recur {:?}>", l),
        };
        write!(f, "{}", s)
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

//===----------------------------------------------------------------------===//
// Macros
//===----------------------------------------------------------------------===//

#[macro_export]
macro_rules! list {
    () => (
        Value::List(vec![])
    );
    ($($args:expr),*) => {{
        let v: Vec<Value> = vec![$($args),*];
        Value::List(v)
    }};
}

#[macro_export]
macro_rules! vector {
    () => (
        Value::Vector(vec![])
    );
    ($($args:expr),*) => {{
        let v: Vec<Value> = vec![$($args),*];
        Value::Vector(v)
    }};
}

#[macro_export]
macro_rules! set {
    () => {
        Value::Set(BTreeSet::new())
    };
    ($($args:expr),*) => {{
        let s: BTreeSet<Value> = BTreeSet::from([$($args),*]);
        Value::Set(s)
    }};
}

