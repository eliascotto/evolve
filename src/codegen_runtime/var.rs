//! Global variable management for codegen runtime.
//!
//! This module provides runtime support for Var operations in compiled code.
//! Variables are stored in a global registry keyed by symbol ID.

use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::{OnceLock, RwLock};

use super::memory::BoxedHeader;
use super::value::make_nil;
use super::BoxedType;

/// VarObject layout in memory:
/// - BoxedHeader (type_tag=6)
/// - sym_id: i64
/// - value: i64 (tagged value)
#[repr(C)]
pub struct VarObject {
    pub header: BoxedHeader,
    pub sym_id: i64,
    pub value: i64,
}

/// Global registry mapping symbol IDs to their current values.
static VAR_REGISTRY: OnceLock<RwLock<HashMap<i64, i64>>> = OnceLock::new();

/// Initialize the var registry. Must be called before any var operations.
pub fn init_var_registry() {
    VAR_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()));
}

fn get_registry() -> &'static RwLock<HashMap<i64, i64>> {
    VAR_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Get the value of a var by symbol ID.
/// Returns the value or nil if not found.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_var_get(sym_id: i64) -> i64 {
    let registry = get_registry().read().unwrap();
    registry.get(&sym_id).copied().unwrap_or_else(make_nil)
}

/// Define a var with the given symbol ID and value.
/// Stores the value in the registry and returns it.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_var_def(sym_id: i64, value: i64) -> i64 {
    let mut registry = get_registry().write().unwrap();
    registry.insert(sym_id, value);
    value
}

/// Set the value of an existing var.
/// Returns the new value.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_var_set(sym_id: i64, value: i64) -> i64 {
    let mut registry = get_registry().write().unwrap();
    registry.insert(sym_id, value);
    value
}

/// Check if a var is bound (has a value).
/// Returns 1 (true) if bound, 0 (false) otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_var_bound(sym_id: i64) -> i64 {
    let registry = get_registry().read().unwrap();
    if registry.contains_key(&sym_id) { 1 } else { 0 }
}

/// Create a VarObject on the heap and return its tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_var_create(sym_id: i64, value: i64) -> i64 {
    let size = std::mem::size_of::<VarObject>() as i64;
    let ptr = super::memory::evolve_alloc(size);
    if ptr.is_null() {
        super::error::evolve_panic(b"Failed to allocate var\0".as_ptr() as *const i8);
    }

    unsafe {
        let var = ptr as *mut VarObject;
        (*var).header.type_tag = BoxedType::Var as u8;
        (*var).header.ref_count = AtomicU32::new(1);
        (*var).sym_id = sym_id;
        (*var).value = value;
    }

    ptr as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_def_and_get() {
        init_var_registry();
        let sym_id = 12345i64;
        let value = 42i64;

        evolve_var_def(sym_id, value);
        let result = evolve_var_get(sym_id);
        assert_eq!(result, value);
    }

    #[test]
    fn test_var_get_unbound() {
        init_var_registry();
        let result = evolve_var_get(99999i64);
        assert_eq!(result, make_nil());
    }

    #[test]
    fn test_var_bound() {
        init_var_registry();
        let sym_id = 11111i64;
        assert_eq!(evolve_var_bound(sym_id), 0);

        evolve_var_def(sym_id, 100);
        assert_eq!(evolve_var_bound(sym_id), 1);
    }
}
