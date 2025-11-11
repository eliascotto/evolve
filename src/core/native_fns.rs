use std::convert::TryFrom;
use std::sync::Arc;
use std::collections::HashMap;

use crate::env::Env;
use crate::error::Error;
use crate::reader::Span;
use crate::value::Value;
use crate::interner::{self, SymId};

//===----------------------------------------------------------------------===//
// Native Functions
//===----------------------------------------------------------------------===//

pub type NativeFn = fn(&[Value], &mut Env) -> Result<Value, Error>;

pub struct NativeRegistry {
    fns: HashMap<SymId, NativeFn>,
}

impl NativeRegistry {
    pub fn new() -> Self {
        let mut fns = HashMap::new();
        fns.insert(interner::intern_sym("concat"), concat as NativeFn);
        fns.insert(interner::intern_sym("conj"), conj as NativeFn);
        fns.insert(interner::intern_sym("assoc"), assoc as NativeFn);
        fns.insert(interner::intern_sym("dissoc"), dissoc as NativeFn);
        fns.insert(interner::intern_sym("get"), get as NativeFn);
        fns.insert(interner::intern_sym("count"), count as NativeFn);
        Self { fns }
    }

    pub fn resolve(&self, sym: SymId) -> Option<NativeFn> {
        self.fns.get(&sym).copied()
    }
}

/// `(concat & strings)` — concatenates zero or more string values into a single
/// `String` value.
///
/// # Errors
///
/// Returns a `TypeError` if any argument is not a string.
pub fn concat(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    let mut result = String::new();
    for value in args {
        match value {
            Value::String { value: s, .. } => result.push_str(s.as_ref()),
            other => return Err(type_error("String", other)),
        }
    }

    Ok(Value::String { span: synthetic_span(), value: Arc::<str>::from(result) })
}

/// `(conj coll & items)` — inserts the provided items into the given
/// collection, returning an updated collection of the same type.
///
/// * Lists are prepended (like Clojure).
/// * Vectors append items.
/// * Sets insert items and preserve insertion order uniqueness.
/// * Maps accept entries as `[k v]`, `(k v)` or whole map values.
///
/// # Errors
///
/// * Returns a `RuntimeError` when called without any arguments.
/// * Returns a `TypeError` if the collection type or entry shape is invalid.
pub fn conj(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    let (first, rest) = match args.split_first() {
        Some(split) => split,
        None => {
            return Err(Error::RuntimeError(
                "conj requires at least one argument".to_string(),
            ));
        }
    };

    if rest.is_empty() {
        return Ok(first.clone());
    }

    match first {
        Value::List { span, value: list, meta } => {
            let mut new_list = list.as_ref().clone();
            for item in rest {
                new_list = new_list.prepend(item.clone());
            }
            Ok(Value::List {
                span: span.clone(),
                value: Arc::new(new_list),
                meta: meta.clone(),
            })
        }
        Value::Vector { span, value: vector, meta } => {
            let mut new_vector = vector.as_ref().clone();
            for item in rest {
                new_vector = new_vector.push_back(item.clone());
            }
            Ok(Value::Vector {
                span: span.clone(),
                value: Arc::new(new_vector),
                meta: meta.clone(),
            })
        }
        Value::Set { span, value: set, meta } => {
            let mut new_set = set.as_ref().clone();
            for item in rest {
                new_set = new_set.insert(item.clone());
            }
            Ok(Value::Set {
                span: span.clone(),
                value: Arc::new(new_set),
                meta: meta.clone(),
            })
        }
        Value::Map { span, value: map, meta } => {
            let mut new_map = map.as_ref().clone();
            for item in rest {
                match item {
                    Value::Vector { value: pair, .. }
                        if pair.as_ref().len() == 2 =>
                    {
                        let pair_vec = pair.as_ref();
                        let key =
                            pair_vec.get(0).expect("length checked above").clone();
                        let value =
                            pair_vec.get(1).expect("length checked above").clone();
                        new_map = new_map.insert(key, value);
                    }
                    Value::List { value: pair, .. } if pair.as_ref().len() == 2 => {
                        let mut iter = pair.as_ref().iter();
                        let key = iter.next().unwrap().clone();
                        let value = iter.next().unwrap().clone();
                        new_map = new_map.insert(key, value);
                    }
                    Value::Map { value: other_map, .. } => {
                        for (key, value) in other_map.as_ref().iter() {
                            new_map = new_map.insert(key.clone(), value.clone());
                        }
                    }
                    other => {
                        return Err(Error::TypeError {
                            expected: "map entry (vector/list of length 2) or map"
                                .to_string(),
                            actual: other.as_str().to_string(),
                        });
                    }
                }
            }
            Ok(Value::Map {
                span: span.clone(),
                value: Arc::new(new_map),
                meta: meta.clone(),
            })
        }
        other => Err(type_error("Collection", other)),
    }
}

/// `(assoc map k v & kvs)` — returns an updated map with the provided key/value
/// bindings applied.
///
/// # Errors
///
/// * Returns a `RuntimeError` if fewer than three arguments are supplied or key/value
///   pairs are incomplete.
/// * Returns a `TypeError` if the first argument is not a map.
pub fn assoc(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() < 3 {
        return Err(Error::RuntimeError(
            "assoc requires a map and at least one key/value pair".to_string(),
        ));
    }

    if args.len() % 2 == 0 {
        return Err(Error::RuntimeError(
            "assoc expects key/value pairs after the map argument".to_string(),
        ));
    }

    let (first, rest) = args.split_first().expect("length checked above");
    let (span, map, meta) = match first {
        Value::Map { span, value, meta } => (span, value, meta),
        other => return Err(type_error("Map", other)),
    };

    let mut new_map = map.as_ref().clone();
    let mut iter = rest.iter();
    while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
        new_map = new_map.insert(key.clone(), value.clone());
    }

    Ok(Value::Map {
        span: span.clone(),
        value: Arc::new(new_map),
        meta: meta.clone(),
    })
}

/// `(dissoc map & keys)` — returns a new map with the specified keys removed.
///
/// # Errors
///
/// * Returns a `RuntimeError` if fewer than two arguments are supplied.
/// * Returns a `TypeError` if the first argument is not a map.
pub fn dissoc(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() < 2 {
        return Err(Error::RuntimeError(
            "dissoc requires a map and at least one key".to_string(),
        ));
    }

    let (first, keys) = args.split_first().expect("length checked above");
    let (span, map, meta) = match first {
        Value::Map { span, value, meta } => (span, value, meta),
        other => return Err(type_error("Map", other)),
    };

    if keys.is_empty() {
        return Ok(first.clone());
    }

    let mut new_map = map.as_ref().clone();
    for key in keys {
        new_map = new_map.remove(key);
    }

    Ok(Value::Map {
        span: span.clone(),
        value: Arc::new(new_map),
        meta: meta.clone(),
    })
}

/// `(get coll key default?)` — looks up `key` in a map or index in a vector,
/// returning the associated value or an optional default (or `nil`).
///
/// # Errors
///
/// * Returns a `RuntimeError` if called with fewer than two or more than three arguments.
/// * Returns a `TypeError` if the collection is not a map/vector, or the key is not an integer for vectors.
pub fn get(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() < 2 {
        return Err(Error::RuntimeError(
            "get requires at least a collection and a key/index".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(Error::RuntimeError(
            "get expects at most a collection, a key/index, and an optional default"
                .to_string(),
        ));
    }

    let collection = &args[0];
    let key = &args[1];
    let default = args.get(2).cloned();

    match collection {
        Value::Map { value: map, .. } => {
            if let Some(value) = map.get(key).cloned() {
                Ok(value)
            } else {
                Ok(default.unwrap_or_else(|| Value::Nil { span: synthetic_span() }))
            }
        }
        Value::Vector { value: vector, .. } => {
            let index = match key {
                Value::Int { value: idx, .. } => *idx,
                other => return Err(type_error("Int", other)),
            };
            if index < 0 {
                return Ok(
                    default.unwrap_or_else(|| Value::Nil { span: synthetic_span() })
                );
            }

            let index = usize::try_from(index).map_err(|_| {
                Error::RuntimeError(
                    "index is too large for this platform".to_string(),
                )
            })?;

            if let Some(value) = vector.get(index).cloned() {
                Ok(value)
            } else {
                Ok(default.unwrap_or_else(|| Value::Nil { span: synthetic_span() }))
            }
        }
        other => Err(type_error("Map or Vector", other)),
    }
}

/// `(count coll)` — returns the number of elements in the collection (including
/// strings).
///
/// Nil counts as zero. Strings count code points, not bytes.
///
/// # Errors
///
/// * Returns a `RuntimeError` if no arguments are provided.
/// * Returns a `TypeError` if the value is not a supported collection.
/// * Returns a `RuntimeError` if the resulting length does not fit in `i64`.
pub fn count(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.is_empty() {
        return Err(Error::RuntimeError(
            "count requires a collection argument".to_string(),
        ));
    }

    let len = match &args[0] {
        Value::Nil { .. } => 0usize,
        Value::List { value: list, .. } => list.len(),
        Value::Vector { value: vector, .. } => vector.len(),
        Value::Map { value: map, .. } => map.len(),
        Value::Set { value: set, .. } => set.len(),
        Value::String { value: s, .. } => s.chars().count(),
        other => return Err(type_error("Collection", other)),
    };

    let int_value = i64::try_from(len).map_err(|_| {
        Error::RuntimeError("collection length exceeds supported range".to_string())
    })?;

    Ok(Value::Int { span: synthetic_span(), value: int_value })
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//
fn synthetic_span() -> Span {
    Span { start: 0, end: 0 }
}

fn type_error(expected: &str, actual: &Value) -> Error {
    Error::TypeError {
        expected: expected.to_string(),
        actual: actual.as_str().to_string(),
    }
}
