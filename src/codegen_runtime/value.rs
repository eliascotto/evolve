//! Value boxing/unboxing utilities for the Evolve codegen runtime.
//!
//! This module provides helper functions for converting between tagged
//! values and Rust types at runtime.

pub use crate::codegen::value::{BoxedType, ValueTag};

/// Mask to extract the tag from a tagged value
pub const TAG_MASK: i64 = 0b111;

/// Number of bits used for the tag
pub const TAG_BITS: u32 = 3;

/// Tag for boxed pointers
pub const TAG_BOXED: i64 = ValueTag::Boxed as i64;

/// Tag for nil value
pub const TAG_NIL: i64 = ValueTag::Nil as i64;

/// Boxed type constants for convenience
pub const BOXED_TYPE_STRING: u8 = BoxedType::String as u8;
pub const BOXED_TYPE_LIST: u8 = BoxedType::List as u8;
pub const BOXED_TYPE_VECTOR: u8 = BoxedType::Vector as u8;
pub const BOXED_TYPE_MAP: u8 = BoxedType::Map as u8;
pub const BOXED_TYPE_SET: u8 = BoxedType::Set as u8;
pub const BOXED_TYPE_CLOSURE: u8 = BoxedType::Closure as u8;
pub const BOXED_TYPE_VAR: u8 = BoxedType::Var as u8;
pub const BOXED_TYPE_NAMESPACE: u8 = BoxedType::Namespace as u8;
pub const BOXED_TYPE_FLOAT: u8 = BoxedType::Float as u8;

/// Get the tag from a tagged value
#[inline]
pub fn get_tag(value: i64) -> i64 {
    value & TAG_MASK
}

/// Check if a value is a boxed pointer (tag == 0b000)
#[inline]
pub fn is_boxed(value: i64) -> bool {
    (value & TAG_MASK) == 0
}

/// Extract the pointer from a boxed value.
///
/// # Safety
/// The caller must ensure the value is actually a boxed pointer.
#[inline]
pub fn extract_ptr(value: i64) -> *mut u8 {
    (value & !TAG_MASK) as *mut u8
}

/// Get the boxed type tag from a boxed object pointer.
///
/// # Safety
/// The pointer must point to a valid BoxedHeader.
#[inline]
pub unsafe fn get_boxed_type(ptr: *mut u8) -> u8 {
    unsafe { *ptr }
}

/// Check if a tagged value is truthy.
///
/// In Evolve/Clojure semantics:
/// - `nil` is falsy
/// - `false` is falsy
/// - Everything else is truthy (including 0, empty string, etc.)
#[unsafe(no_mangle)]
pub extern "C" fn evolve_is_truthy(value: i64) -> bool {
    let tag = value & TAG_MASK;

    // Nil (tag 0b010) is falsy
    if tag == ValueTag::Nil as i64 {
        return false;
    }

    // Bool (tag 0b011) - check bit 3 for the actual value
    if tag == ValueTag::Bool as i64 {
        return (value >> TAG_BITS) & 1 == 1;
    }

    // Everything else is truthy
    true
}

/// Extract an integer from a tagged value.
///
/// # Safety
/// The caller should ensure the value has tag 0b001 (Int).
#[inline]
pub fn extract_int(value: i64) -> i64 {
    value >> TAG_BITS
}

/// Create a tagged integer value.
#[inline]
pub fn make_int(n: i64) -> i64 {
    (n << TAG_BITS) | (ValueTag::Int as i64)
}

/// Create a nil value.
#[inline]
pub const fn make_nil() -> i64 {
    ValueTag::Nil as i64
}

/// Create a boolean value.
#[inline]
pub fn make_bool(b: bool) -> i64 {
    let val = if b { 1i64 << TAG_BITS } else { 0 };
    val | (ValueTag::Bool as i64)
}

/// Create a boxed pointer value.
#[inline]
pub fn make_boxed(ptr: *mut u8) -> i64 {
    ptr as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_truthy_nil() {
        let nil = make_nil();
        assert!(!evolve_is_truthy(nil));
    }

    #[test]
    fn test_is_truthy_bool() {
        let true_val = make_bool(true);
        let false_val = make_bool(false);
        assert!(evolve_is_truthy(true_val));
        assert!(!evolve_is_truthy(false_val));
    }

    #[test]
    fn test_is_truthy_int() {
        let zero = make_int(0);
        let one = make_int(1);
        let neg = make_int(-42);
        assert!(evolve_is_truthy(zero)); // 0 is truthy in Lisp
        assert!(evolve_is_truthy(one));
        assert!(evolve_is_truthy(neg));
    }

    #[test]
    fn test_is_boxed() {
        let ptr = 0x1000i64; // aligned pointer, tag bits are 0
        assert!(is_boxed(ptr));

        let int_val = make_int(42);
        assert!(!is_boxed(int_val));

        let nil = make_nil();
        assert!(!is_boxed(nil));
    }

    #[test]
    fn test_extract_ptr() {
        let ptr = 0x1000i64;
        assert_eq!(extract_ptr(ptr), 0x1000 as *mut u8);
    }

    #[test]
    fn test_int_roundtrip() {
        for n in [-1000, -1, 0, 1, 42, 1000000] {
            let tagged = make_int(n);
            assert_eq!(extract_int(tagged), n);
        }
    }
}
