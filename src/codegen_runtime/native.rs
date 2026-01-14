//! Native function registry for codegen runtime.
//!
//! This module provides runtime support for calling native (Rust) functions
//! from compiled code. Native functions are registered by symbol ID and can
//! be invoked with a standard calling convention.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Native function signature: takes argc and argv, returns a tagged value.
pub type NativeFunction = extern "C" fn(i32, *const i64) -> i64;

/// Global registry mapping symbol IDs to native function pointers.
static NATIVE_REGISTRY: OnceLock<RwLock<HashMap<i64, NativeFunction>>> = OnceLock::new();

/// Initialize the native function registry.
/// Must be called before any native function operations.
pub fn init_native_registry() {
    NATIVE_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()));
    register_core_natives();
}

fn get_registry() -> &'static RwLock<HashMap<i64, NativeFunction>> {
    NATIVE_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a native function by symbol ID.
pub fn register_native(sym_id: i64, f: NativeFunction) {
    let mut registry = get_registry().write().unwrap();
    registry.insert(sym_id, f);
}

/// Look up a native function by symbol ID.
pub fn lookup_native(sym_id: i64) -> Option<NativeFunction> {
    let registry = get_registry().read().unwrap();
    registry.get(&sym_id).copied()
}

/// Call a native function by symbol ID.
/// Panics if the function is not registered.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_native_call(sym_id: i64, argc: i32, argv: *const i64) -> i64 {
    let registry = get_registry().read().unwrap();
    match registry.get(&sym_id) {
        Some(f) => f(argc, argv),
        None => panic!("Native function not found for symbol ID: {}", sym_id),
    }
}

/// Try to call a native function by symbol ID.
/// Returns the result, or -1 if the function is not registered.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_native_try_call(sym_id: i64, argc: i32, argv: *const i64) -> i64 {
    let registry = get_registry().read().unwrap();
    match registry.get(&sym_id) {
        Some(f) => f(argc, argv),
        None => -1,
    }
}

/// Check if a native function is registered.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_native_exists(sym_id: i64) -> i64 {
    let registry = get_registry().read().unwrap();
    if registry.contains_key(&sym_id) { 1 } else { 0 }
}

/// Register core native functions (arithmetic, etc.).
fn register_core_natives() {
    // TODO: Register core native functions here
    // Examples:
    // - evolve.core/+ -> native_add
    // - evolve.core/- -> native_sub
    // - evolve.core/* -> native_mul
    // - evolve.core// -> native_div
    // - evolve.core/= -> native_eq
    // - evolve.core/< -> native_lt
    // - evolve.core/> -> native_gt
    // - evolve.core/print -> native_print
    // - evolve.core/println -> native_println
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn test_native_add(_argc: i32, argv: *const i64) -> i64 {
        unsafe {
            let a = *argv;
            let b = *argv.add(1);
            a + b
        }
    }

    #[test]
    fn test_register_and_call() {
        init_native_registry();
        register_native(12345, test_native_add);

        let args = [10i64, 20i64];
        let result = evolve_native_call(12345, 2, args.as_ptr());
        assert_eq!(result, 30);
    }

    #[test]
    fn test_native_exists() {
        init_native_registry();
        register_native(99999, test_native_add);

        assert_eq!(evolve_native_exists(99999), 1);
        assert_eq!(evolve_native_exists(88888), 0);
    }
}
