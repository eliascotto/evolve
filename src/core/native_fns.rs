use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

use crate::agent::Agent;
use crate::atom::Atom;
use crate::collections::{List, Map, Set, Vector};
use crate::condition::Condition;
use crate::env::Env;
use crate::error::{Error, SyntaxError};
use crate::eval::{Evaluator, make_do_value};
use crate::interner::{self, SymId};
use crate::stm::Ref;
use crate::value::Value;
use crate::synthetic_span;

//===----------------------------------------------------------------------===//
// Native Functions
//===----------------------------------------------------------------------===//

pub type NativeFn = fn(&[Value], &mut Env) -> Result<Value, Error>;

#[derive(Debug, Clone)]
pub struct NativeRegistry {
    fns: HashMap<SymId, NativeFn>,
}

impl NativeRegistry {
    pub fn new() -> Self {
        let mut fns = HashMap::new();
        fns.insert(interner::intern_sym("concat"), concat as NativeFn);
        fns.insert(interner::intern_sym("seq"), seq as NativeFn);
        fns.insert(interner::intern_sym("conj"), conj as NativeFn);
        fns.insert(interner::intern_sym("assoc"), assoc as NativeFn);
        fns.insert(interner::intern_sym("dissoc"), dissoc as NativeFn);
        fns.insert(interner::intern_sym("get"), get as NativeFn);
        fns.insert(interner::intern_sym("count"), count as NativeFn);
        fns.insert(interner::intern_sym("list"), list as NativeFn);
        fns.insert(interner::intern_sym("macroexpand1"), macroexpand1 as NativeFn);
        fns.insert(interner::intern_sym("macroexpand"), macroexpand as NativeFn);

        // Atom functions
        fns.insert(interner::intern_sym("atom"), atom as NativeFn);
        fns.insert(interner::intern_sym("deref"), deref as NativeFn);
        fns.insert(interner::intern_sym("reset!"), reset_bang as NativeFn);
        fns.insert(interner::intern_sym("swap!"), swap_bang as NativeFn);
        fns.insert(interner::intern_sym("compare-and-set!"), compare_and_set_bang as NativeFn);

        // STM functions
        fns.insert(interner::intern_sym("ref"), ref_new as NativeFn);
        fns.insert(interner::intern_sym("ref-set"), ref_set as NativeFn);
        fns.insert(interner::intern_sym("alter"), alter as NativeFn);

        // Agent functions
        fns.insert(interner::intern_sym("agent"), agent as NativeFn);
        fns.insert(interner::intern_sym("send"), send as NativeFn);
        fns.insert(interner::intern_sym("send-off"), send as NativeFn); // alias
        fns.insert(interner::intern_sym("await"), await_agent as NativeFn);

        // Condition system functions
        fns.insert(interner::intern_sym("make-condition"), make_condition as NativeFn);

        Self { fns }
    }

    pub fn resolve(&self, sym: SymId) -> Option<NativeFn> {
        self.fns.get(&sym).copied()
    }

    /// Returns the `SymId` for the native function registered under `name`, if any.
    ///
    /// This interns `name` via the global symbol interner and only returns `Some`
    /// when that symbol corresponds to a registered native function.
    pub fn sym_for_name(&self, name: &str) -> Option<SymId> {
        let sym = interner::intern_sym(name);
        if self.fns.contains_key(&sym) { Some(sym) } else { None }
    }
}

/// `(list & items)` — constructs a persistent list containing `items` in order.
///
/// Mirrors Clojure's `list`, returning a new list every time (even when empty).
pub fn list(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    Ok(list_from_iter(args.iter().cloned()))
}

/// `(seq coll)` — returns a sequence view over `coll`, like Clojure's `seq`.
///
/// Returns `nil` when called with `nil` or an empty collection. Works with lists,
/// vectors, sets, maps (yielding `[k v]` entry vectors), and strings (yielding
/// chars). Non-seqable values raise a type error.
pub fn seq(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err(Error::RuntimeError(
            "seq requires exactly one argument: a collection".to_string(),
        ));
    }

    let coll = &args[0];
    match coll {
        Value::Nil { .. } => Ok(Value::Nil { span: synthetic_span!() }),
        Value::List { value: list, .. } => {
            if list.is_empty() {
                Ok(Value::Nil { span: synthetic_span!() })
            } else {
                Ok(coll.clone())
            }
        }
        Value::Vector { value: vector, .. } => {
            if vector.is_empty() {
                Ok(Value::Nil { span: synthetic_span!() })
            } else {
                Ok(list_from_iter(vector.iter().cloned()))
            }
        }
        Value::Set { value: set, .. } => {
            if set.is_empty() {
                Ok(Value::Nil { span: synthetic_span!() })
            } else {
                Ok(list_from_iter(set.iter().cloned()))
            }
        }
        Value::Map { value: map, .. } => {
            if map.is_empty() {
                Ok(Value::Nil { span: synthetic_span!() })
            } else {
                let entries =
                    map.iter().map(|(k, v)| map_entry_value(k.clone(), v.clone()));
                Ok(list_from_iter(entries))
            }
        }
        Value::String { value: s, .. } => {
            if s.is_empty() {
                Ok(Value::Nil { span: synthetic_span!() })
            } else {
                let chars = s
                    .chars()
                    .map(|ch| Value::Char { span: synthetic_span!(), value: ch });
                Ok(list_from_iter(chars))
            }
        }
        other => Err(type_error("Seqable", other)),
    }
}

/// `(concat & colls)` — concatenates zero or more seqable values into a list.
///
/// Accepts lists, vectors, sets, or nil. Each collection's elements are appended
/// in order to produce a new list.
pub fn concat(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    let mut items: Vec<Value> = Vec::new();
    for value in args {
        match value {
            Value::Nil { .. } => {}
            Value::List { value: list, .. } => {
                items.extend(list.iter().cloned());
            }
            Value::Vector { value: vector, .. } => {
                items.extend(vector.iter().cloned());
            }
            Value::Set { value: set, .. } => {
                items.extend(set.iter().cloned());
            }
            other => return Err(type_error("Seqable", other)),
        }
    }

    Ok(Value::List {
        span: synthetic_span!(),
        value: Arc::new(List::from_iter(items)),
        meta: None,
    })
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
                Ok(default.unwrap_or_else(|| Value::Nil { span: synthetic_span!() }))
            }
        }
        Value::Vector { value: vector, .. } => {
            let index = match key {
                Value::Int { value: idx, .. } => *idx,
                other => return Err(type_error("Int", other)),
            };
            if index < 0 {
                return Ok(
                    default.unwrap_or_else(|| Value::Nil { span: synthetic_span!() })
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
                Ok(default.unwrap_or_else(|| Value::Nil { span: synthetic_span!() }))
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

    Ok(Value::Int { span: synthetic_span!(), value: int_value })
}

/// `(macroexpand1 form)` — expands the first macro in the form.
pub fn macroexpand1(args: &[Value], env: &mut Env) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err(Error::RuntimeError(
            "macroexpand1 requires a single argument: a form to macroexpand"
                .to_string(),
        ));
    }

    // Code similar to expand_macro in eval.rs
    match &args[0] {
        Value::List { value: list, .. } => {
            if list.is_empty() {
                return Ok(args[0].clone());
            }

            let head = list.head().unwrap().clone();
            let macro_args = list.tail().unwrap_or_else(List::new);

            let maybe_macro_var = match &head {
                Value::Var { value: var, .. } => Some(var.clone()),
                Value::Symbol { value: sym, .. } => env.ns.get(sym.id()).cloned(),
                _ => return Ok(head.clone()),
            };

            if let Some(var) = maybe_macro_var {
                if var.is_macro() {
                    let macro_name = interner::sym_to_str(var.symbol);
                    let storage = var.value.as_ref().ok_or_else(|| {
                        Error::RuntimeError(format!(
                            "Macro '{}' is unbound",
                            macro_name
                        ))
                    })?;

                    let macro_value = storage.read().map_err(|_| {
                        Error::RuntimeError(format!(
                            "Failed to read macro binding for '{}'",
                            macro_name
                        ))
                    })?;

                    let macro_fn = macro_value.clone();
                    drop(macro_value);

                    let macro_span = macro_fn.span();
                    let (params, body, macro_env) = match macro_fn {
                        Value::Function { params, body, env: fn_env, .. } => {
                            (params, body, fn_env)
                        }
                        other => {
                            return Err(Error::TypeError {
                                expected: "function".to_string(),
                                actual: other.as_str().to_string(),
                            });
                        }
                    };

                    let provided_args: Vec<Value> =
                        macro_args.iter().cloned().collect();
                    if params.len() != provided_args.len() {
                        return Err(Error::SyntaxError(
                            SyntaxError::WrongArgumentCount {
                                error_str: format!(
                                    "Wrong number of arguments to macro. Expected {}, got {}",
                                    params.len(),
                                    provided_args.len()
                                ),
                            },
                        ));
                    }

                    let mut binding_pairs =
                        Vec::with_capacity(provided_args.len() * 2);
                    for (param, arg_form) in params.iter().zip(provided_args.iter())
                    {
                        binding_pairs.push(param.clone());
                        binding_pairs.push(arg_form.clone());
                    }
                    let bindings = Arc::new(Vector::from_iter(binding_pairs));
                    let mut macro_scope_env =
                        macro_env.create_child_with_bindings(bindings);

                    // Create a new evaluator to evaluate the macro body.
                    let evaluator = Evaluator::new();
                    // Create a do value for the macro body.
                    let body_form =
                        make_do_value(&evaluator, body.clone(), macro_span.clone());
                    // Evaluate the macro body.
                    let expansion_form = evaluator
                        .eval(&body_form, &mut macro_scope_env)
                        .map_err(|eval_err| {
                            Error::RuntimeError(format!(
                                "macroexpand evaluation failed at {:?}: {}",
                                eval_err.span, eval_err.error
                            ))
                        })?;

                    // Return the expanded form.
                    Ok(expansion_form)
                } else {
                    return Ok(args[0].clone());
                }
            } else {
                return Ok(args[0].clone());
            }
        }
        _ => return Ok(args[0].clone()),
    }
}

/// `(macroexpand form)` — expands all macros in the form.
pub fn macroexpand(args: &[Value], env: &mut Env) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err(Error::RuntimeError(
            "macroexpand requires a single argument: a form to macroexpand"
                .to_string(),
        ));
    }

    macroexpand_all(&args[0], env)
}

fn macroexpand_all(form: &Value, env: &mut Env) -> Result<Value, Error> {
    let mut expanded = form.clone();
    loop {
        let next = macroexpand1(&[expanded.clone()], env)?;
        if next == expanded {
            break;
        }
        expanded = next;
    }

    match expanded {
        Value::List { span, value, meta } => {
            let mut items = Vec::with_capacity(value.len());
            for item in value.iter() {
                items.push(macroexpand_all(item, env)?);
            }
            Ok(Value::List {
                span,
                value: Arc::new(List::from_iter(items)),
                meta,
            })
        }
        Value::Vector { span, value, meta } => {
            let mut items = Vec::with_capacity(value.len());
            for item in value.iter() {
                items.push(macroexpand_all(item, env)?);
            }
            Ok(Value::Vector {
                span,
                value: Arc::new(Vector::from_iter(items)),
                meta,
            })
        }
        Value::Map { span, value, meta } => {
            let mut entries = Vec::with_capacity(value.len());
            for (key, val) in value.iter() {
                let expanded_key = macroexpand_all(key, env)?;
                let expanded_val = macroexpand_all(val, env)?;
                entries.push((expanded_key, expanded_val));
            }
            Ok(Value::Map {
                span,
                value: Arc::new(Map::from_iter(entries)),
                meta,
            })
        }
        Value::Set { span, value, meta } => {
            let mut items = Vec::with_capacity(value.len());
            for item in value.iter() {
                items.push(macroexpand_all(item, env)?);
            }
            Ok(Value::Set {
                span,
                value: Arc::new(Set::from_iter(items)),
                meta,
            })
        }
        other => Ok(other),
    }
}

//===----------------------------------------------------------------------===//
// Atoms
//===----------------------------------------------------------------------===//

/// `(atom value)` — creates a new atom with the given initial value.
pub fn atom(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err(Error::RuntimeError(
            "atom requires exactly one argument: the initial value".to_string(),
        ));
    }
    Ok(Value::Atom {
        span: synthetic_span!(),
        value: Arc::new(Atom::new(args[0].clone())),
    })
}

/// `(deref ref)` — dereferences an atom, ref, or agent to get its current value.
pub fn deref(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err(Error::RuntimeError(
            "deref requires exactly one argument".to_string(),
        ));
    }
    match &args[0] {
        Value::Atom { value, .. } => Ok(value.deref()),
        Value::Ref { value, .. } => Ok(value.deref()),
        Value::Agent { value, .. } => Ok(value.deref()),
        Value::Var { value, .. } => {
            if let Some(storage) = &value.value {
                Ok(storage.read().unwrap().clone())
            } else {
                Err(Error::RuntimeError("Var is unbound".to_string()))
            }
        }
        other => Err(type_error("Atom, Ref, Agent, or Var", other)),
    }
}

/// `(reset! atom value)` — atomically sets the atom's value to the new value.
pub fn reset_bang(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err(Error::RuntimeError(
            "reset! requires exactly two arguments: atom and new value".to_string(),
        ));
    }
    match &args[0] {
        Value::Atom { value: atom, .. } => Ok(atom.reset(args[1].clone())),
        other => Err(type_error("Atom", other)),
    }
}

/// `(swap! atom f & args)` — atomically updates the atom's value by applying f.
///
/// The function receives the current value followed by any additional args,
/// and returns the new value.
pub fn swap_bang(args: &[Value], env: &mut Env) -> Result<Value, Error> {
    if args.len() < 2 {
        return Err(Error::RuntimeError(
            "swap! requires at least two arguments: atom and function".to_string(),
        ));
    }

    let atom = match &args[0] {
        Value::Atom { value, .. } => value.clone(),
        other => return Err(type_error("Atom", other)),
    };

    let func = &args[1];
    let extra_args: Vec<Value> = args[2..].to_vec();

    let new_value = atom.swap(|current| {
        let mut call_args = vec![current.clone()];
        call_args.extend(extra_args.clone());
        apply_function(func.clone(), &call_args, &mut env.clone()).unwrap_or(current.clone())
    });

    Ok(new_value)
}

/// `(compare-and-set! atom expected new-value)` — atomically sets the atom if current equals expected.
pub fn compare_and_set_bang(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 3 {
        return Err(Error::RuntimeError(
            "compare-and-set! requires exactly three arguments: atom, expected, and new value"
                .to_string(),
        ));
    }
    match &args[0] {
        Value::Atom { value: atom, .. } => {
            let success = atom.compare_and_set(&args[1], args[2].clone());
            Ok(Value::Bool { span: synthetic_span!(), value: success })
        }
        other => Err(type_error("Atom", other)),
    }
}

//===----------------------------------------------------------------------===//
// STM References
//===----------------------------------------------------------------------===//

/// `(ref value)` — creates a new STM ref with the given initial value.
pub fn ref_new(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err(Error::RuntimeError(
            "ref requires exactly one argument: the initial value".to_string(),
        ));
    }
    Ok(Value::Ref {
        span: synthetic_span!(),
        value: Arc::new(Ref::new(args[0].clone())),
    })
}

/// `(ref-set ref value)` — sets the ref's value within a transaction.
pub fn ref_set(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err(Error::RuntimeError(
            "ref-set requires exactly two arguments: ref and new value".to_string(),
        ));
    }
    match &args[0] {
        Value::Ref { value: r, .. } => r.ref_set(args[1].clone()).map_err(|e| {
            Error::RuntimeError(e.to_string())
        }),
        other => Err(type_error("Ref", other)),
    }
}

/// `(alter ref f & args)` — alters the ref's value by applying f within a transaction.
pub fn alter(args: &[Value], env: &mut Env) -> Result<Value, Error> {
    if args.len() < 2 {
        return Err(Error::RuntimeError(
            "alter requires at least two arguments: ref and function".to_string(),
        ));
    }

    let r = match &args[0] {
        Value::Ref { value, .. } => value.clone(),
        other => return Err(type_error("Ref", other)),
    };

    let func = args[1].clone();
    let extra_args: Vec<Value> = args[2..].to_vec();
    let mut env_clone = env.clone();

    r.alter(|current| {
        let mut call_args = vec![current.clone()];
        call_args.extend(extra_args.clone());
        apply_function(func.clone(), &call_args, &mut env_clone).unwrap_or(current.clone())
    })
    .map_err(|e| Error::RuntimeError(e.to_string()))
}

//===----------------------------------------------------------------------===//
// Agents
//===----------------------------------------------------------------------===//

/// `(agent value)` — creates a new agent with the given initial state.
pub fn agent(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err(Error::RuntimeError(
            "agent requires exactly one argument: the initial state".to_string(),
        ));
    }
    Ok(Value::Agent {
        span: synthetic_span!(),
        value: Arc::new(Agent::new(args[0].clone())),
    })
}

/// `(send agent f & args)` — sends an action to the agent.
pub fn send(args: &[Value], env: &mut Env) -> Result<Value, Error> {
    if args.len() < 2 {
        return Err(Error::RuntimeError(
            "send requires at least two arguments: agent and function".to_string(),
        ));
    }

    let ag = match &args[0] {
        Value::Agent { value, .. } => value.clone(),
        other => return Err(type_error("Agent", other)),
    };

    let func = args[1].clone();
    let extra_args: Vec<Value> = args[2..].to_vec();
    let env_clone = env.clone();

    ag.send(move |current| {
        let mut call_args = vec![current.clone()];
        call_args.extend(extra_args);
        apply_function(func.clone(), &call_args, &mut env_clone.clone())
            .unwrap_or(current.clone())
    });

    Ok(args[0].clone())
}

/// `(await agent & agents)` — waits for all pending actions to complete.
pub fn await_agent(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.is_empty() {
        return Err(Error::RuntimeError(
            "await requires at least one agent".to_string(),
        ));
    }

    for arg in args {
        match arg {
            Value::Agent { value, .. } => value.await_agent(),
            other => return Err(type_error("Agent", other)),
        }
    }

    Ok(Value::Nil { span: synthetic_span!() })
}

//===----------------------------------------------------------------------===//
// Condition System
//===----------------------------------------------------------------------===//

/// `(make-condition name & data)` — creates a new condition with the given name and data.
pub fn make_condition(args: &[Value], _env: &mut Env) -> Result<Value, Error> {
    if args.is_empty() {
        return Err(Error::RuntimeError(
            "make-condition requires at least one argument: the condition name".to_string(),
        ));
    }

    let name = match &args[0] {
        Value::Symbol { value, .. } => value.id(),
        Value::Keyword { value, .. } => {
            // Convert keyword to symbol
            let kw_str = interner::kw_to_str(*value);
            interner::intern_sym(&kw_str)
        }
        other => {
            return Err(type_error("Symbol or Keyword", other));
        }
    };

    let data: Vec<Value> = args[1..].to_vec();
    Ok(Value::Condition {
        span: synthetic_span!(),
        value: Arc::new(Condition::new(name, data)),
    })
}

//===----------------------------------------------------------------------===//
// Apply Function Helper
//===----------------------------------------------------------------------===//

fn apply_function(func: Value, args: &[Value], env: &mut Env) -> Result<Value, Error> {
    match func {
        Value::Function { params, body, env: fn_env, .. } => {
            if params.len() != args.len() {
                return Err(Error::SyntaxError(SyntaxError::WrongArgumentCount {
                    error_str: format!(
                        "Wrong number of arguments. Expected {}, got {}",
                        params.len(),
                        args.len()
                    ),
                }));
            }

            let mut binding_pairs = Vec::with_capacity(args.len() * 2);
            for (param, arg) in params.iter().zip(args.iter()) {
                binding_pairs.push(param.clone());
                binding_pairs.push(arg.clone());
            }
            let bindings = Arc::new(Vector::from_iter(binding_pairs));
            let mut scope_env = fn_env.create_child_with_bindings(bindings);

            let evaluator = Evaluator::new();
            let body_form =
                make_do_value(&evaluator, body.clone(), synthetic_span!());
            evaluator.eval(&body_form, &mut scope_env).map_err(|e| e.error)
        }
        Value::NativeFunction { f, .. } => f(args, env),
        other => Err(type_error("Function", &other)),
    }
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

fn type_error(expected: &str, actual: &Value) -> Error {
    Error::TypeError {
        expected: expected.to_string(),
        actual: actual.as_str().to_string(),
    }
}

fn list_from_iter<I>(iter: I) -> Value
where
    I: IntoIterator<Item = Value>,
{
    Value::List {
        span: synthetic_span!(),
        value: Arc::new(List::from_iter(iter)),
        meta: None,
    }
}

fn map_entry_value(key: Value, value: Value) -> Value {
    Value::Vector {
        span: synthetic_span!(),
        value: Arc::new(Vector::from_iter(vec![key, value])),
        meta: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sym_for_name_returns_registered_symbol() {
        let registry = NativeRegistry::new();
        let seq_sym = registry.sym_for_name("seq").expect("seq should be registered");
        assert!(registry.resolve(seq_sym).is_some());
    }

    #[test]
    fn sym_for_name_returns_none_for_unknown_function() {
        let registry = NativeRegistry::new();
        assert!(registry.sym_for_name("not-a-native").is_none());
    }
}
