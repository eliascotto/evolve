//! Runtime support for LLVM-compiled Evolve code.
//!
//! This module provides C-compatible runtime functions that compiled
//! LLVM code can call for memory management, value manipulation,
//! and other runtime services.

pub mod closure;
pub mod dispatch;
pub mod error;
pub mod list;
pub mod map;
pub mod memory;
pub mod namespace;
pub mod native;
pub mod quote;
pub mod set;
pub mod string;
pub mod value;
pub mod var;
pub mod vector;

pub use error::evolve_panic;
pub use memory::{evolve_alloc, evolve_release, evolve_retain, BoxedHeader};

/// Type tags for boxed objects on the heap.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxedType {
    String = 0,
    List = 1,
    Vector = 2,
    Map = 3,
    Set = 4,
    Closure = 5,
    Var = 6,
    Namespace = 7,
    Float = 8,
}

/// Initialize the runtime. Must be called before any compiled code executes.
///
/// This function:
/// 1. Initializes global variable registry
/// 2. Creates default namespace ("user")
/// 3. Registers native functions
/// 4. Sets up error handlers
#[unsafe(no_mangle)]
pub extern "C" fn evolve_runtime_init() {
    // Initialize global registries
    var::init_var_registry();
    namespace::init_namespace_registry();
    native::init_native_registry();

    // Create and switch to "user" namespace as default
    // Note: This requires a symbol ID for "user" which should be
    // provided by the compiler. For now, use 0 as placeholder.
    // evolve_ns_switch(user_sym_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_init() {
        evolve_runtime_init();
    }

    #[test]
    fn test_boxed_type_values() {
        assert_eq!(BoxedType::String as u8, 0);
        assert_eq!(BoxedType::List as u8, 1);
        assert_eq!(BoxedType::Vector as u8, 2);
        assert_eq!(BoxedType::Map as u8, 3);
        assert_eq!(BoxedType::Set as u8, 4);
        assert_eq!(BoxedType::Closure as u8, 5);
        assert_eq!(BoxedType::Var as u8, 6);
        assert_eq!(BoxedType::Namespace as u8, 7);
        assert_eq!(BoxedType::Float as u8, 8);
    }
}
