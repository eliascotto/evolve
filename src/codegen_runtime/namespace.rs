//! Namespace management for codegen runtime.
//!
//! This module provides runtime support for namespace operations in compiled code.
//! Namespaces are stored in a global registry keyed by namespace ID.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::{OnceLock, RwLock};

use super::memory::BoxedHeader;
use super::BoxedType;

/// NamespaceObject layout in memory:
/// - BoxedHeader (type_tag=7)
/// - ns_id: i64
#[repr(C)]
pub struct NamespaceObject {
    pub header: BoxedHeader,
    pub ns_id: i64,
}

/// Current namespace ID (atomic for thread safety).
static CURRENT_NS: OnceLock<AtomicI64> = OnceLock::new();

/// Global registry mapping namespace IDs to namespace values (tagged pointers).
static NS_REGISTRY: OnceLock<RwLock<HashMap<i64, i64>>> = OnceLock::new();

/// Initialize the namespace registry. Must be called before any namespace operations.
pub fn init_namespace_registry() {
    CURRENT_NS.get_or_init(|| AtomicI64::new(0));
    NS_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()));
}

fn get_current_ns_atomic() -> &'static AtomicI64 {
    CURRENT_NS.get_or_init(|| AtomicI64::new(0))
}

fn get_registry() -> &'static RwLock<HashMap<i64, i64>> {
    NS_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Create a NamespaceObject on the heap and return its pointer.
fn create_namespace_object(ns_id: i64) -> i64 {
    let size = std::mem::size_of::<NamespaceObject>() as i64;
    let ptr = super::memory::evolve_alloc(size);
    if ptr.is_null() {
        super::error::evolve_panic(b"Failed to allocate namespace\0".as_ptr() as *const i8);
    }

    unsafe {
        let ns = ptr as *mut NamespaceObject;
        (*ns).header.type_tag = BoxedType::Namespace as u8;
        (*ns).header.ref_count = AtomicU32::new(1);
        (*ns).ns_id = ns_id;
    }

    ptr as i64
}

/// Switch to a namespace by symbol ID.
/// Creates the namespace if it doesn't exist.
/// Sets it as the current namespace.
/// Returns the namespace value (tagged pointer).
#[unsafe(no_mangle)]
pub extern "C" fn evolve_ns_switch(sym_id: i64) -> i64 {
    let mut registry = get_registry().write().unwrap();

    let ns_value = if let Some(&existing) = registry.get(&sym_id) {
        existing
    } else {
        let new_ns = create_namespace_object(sym_id);
        registry.insert(sym_id, new_ns);
        new_ns
    };

    get_current_ns_atomic().store(sym_id, Ordering::SeqCst);
    ns_value
}

/// Get the current namespace value.
/// Returns the namespace value (tagged pointer) or 0 if none set.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_ns_current() -> i64 {
    let current_id = get_current_ns_atomic().load(Ordering::SeqCst);
    let registry = get_registry().read().unwrap();
    registry.get(&current_id).copied().unwrap_or(0)
}

/// Get the current namespace ID.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_ns_current_id() -> i64 {
    get_current_ns_atomic().load(Ordering::SeqCst)
}

/// Look up a namespace by ID.
/// Returns the namespace value or 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_ns_lookup(ns_id: i64) -> i64 {
    let registry = get_registry().read().unwrap();
    registry.get(&ns_id).copied().unwrap_or(0)
}

/// Register a namespace with the given ID.
/// Returns the namespace value.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_ns_register(ns_id: i64) -> i64 {
    let mut registry = get_registry().write().unwrap();
    if let Some(&existing) = registry.get(&ns_id) {
        existing
    } else {
        let new_ns = create_namespace_object(ns_id);
        registry.insert(ns_id, new_ns);
        new_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ns_switch_creates_namespace() {
        init_namespace_registry();
        let ns_id = 100i64;
        let ns_value = evolve_ns_switch(ns_id);
        assert_ne!(ns_value, 0);
        assert_eq!(evolve_ns_current_id(), ns_id);
    }

    #[test]
    fn test_ns_lookup() {
        init_namespace_registry();
        let ns_id = 200i64;

        assert_eq!(evolve_ns_lookup(ns_id), 0);

        let registered = evolve_ns_register(ns_id);
        assert_ne!(registered, 0);
        assert_eq!(evolve_ns_lookup(ns_id), registered);
    }
}
