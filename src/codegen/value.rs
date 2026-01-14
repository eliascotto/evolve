//! Value representation helpers for LLVM code generation.
//!
//! This module defines how Evolve values are represented in LLVM IR
//! using a tagged pointer scheme.
//!
//! # Tagged Pointer Layout
//!
//! All Evolve values are represented as 64-bit integers with the following structure:
//!
//! ```text
//! 63                                                    3  2  1  0
//! +-----------------------------------------------------+--+--+--+
//! |                    Payload (61 bits)                | Tag (3)|
//! +-----------------------------------------------------+--+--+--+
//! ```
//!
//! # Value Tags
//!
//! - `000` (0): Boxed pointer - pointer to heap-allocated object
//! - `001` (1): Integer - signed 61-bit integer in upper bits
//! - `010` (2): Nil - the nil value
//! - `011` (3): Boolean - bit 3 contains true/false
//! - `100` (4): Character - Unicode codepoint in upper bits
//! - `101` (5): Keyword - KeywId in upper bits
//! - `110` (6): Symbol - SymId in upper bits (for unresolved symbols)
//! - `111` (7): Float - pointer to boxed f64
//!
//! # Boxed Object Layout
//!
//! Heap-allocated objects have a common header:
//!
//! ```text
//! +--------+------------+------------------+
//! | Type   | Ref Count  | Type-specific    |
//! | (i8)   | (i32)      | data...          |
//! +--------+------------+------------------+
//! ```

/// Value tag constants for the tagged pointer representation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTag {
    /// Pointer to boxed object (collections, closures, strings, etc.)
    Boxed = 0b000,
    /// Immediate integer (shifted left by 3 bits)
    Int = 0b001,
    /// Nil value
    Nil = 0b010,
    /// Boolean (bit 3 is the value)
    Bool = 0b011,
    /// Character (Unicode codepoint in upper bits)
    Char = 0b100,
    /// Keyword ID
    Keyword = 0b101,
    /// Symbol ID (for unresolved symbols)
    Symbol = 0b110,
    /// Float (pointer to boxed f64)
    Float = 0b111,
}

impl ValueTag {
    /// The mask to extract the tag from a value.
    pub const MASK: u64 = 0b111;

    /// The number of bits used for the tag.
    pub const BITS: u32 = 3;

    /// Extract the tag from a raw value.
    pub fn from_raw(value: u64) -> Self {
        match value & Self::MASK {
            0b000 => ValueTag::Boxed,
            0b001 => ValueTag::Int,
            0b010 => ValueTag::Nil,
            0b011 => ValueTag::Bool,
            0b100 => ValueTag::Char,
            0b101 => ValueTag::Keyword,
            0b110 => ValueTag::Symbol,
            0b111 => ValueTag::Float,
            _ => unreachable!(),
        }
    }
}

/// Type tags for boxed objects on the heap.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxedType {
    /// String: { header, len: i64, data: [u8] }
    String = 0,
    /// List: { header, len: i64, head: *Node }
    List = 1,
    /// Vector: { header, len: i64, ... }
    Vector = 2,
    /// Map: { header, len: i64, ... }
    Map = 3,
    /// Set: { header, len: i64, ... }
    Set = 4,
    /// Closure: { header, fn_ptr: ptr, env_count: i32, env: [Value] }
    Closure = 5,
    /// Var: { header, sym_id: i64, value: Value }
    Var = 6,
    /// Namespace: { header, ns_id: i64, ... }
    Namespace = 7,
    /// Float (boxed): { header, value: f64 }
    Float = 8,
}

/// Helper functions for working with tagged values.
pub mod helpers {
    use super::*;

    /// Create a nil value.
    #[inline]
    pub const fn nil() -> u64 {
        ValueTag::Nil as u64
    }

    /// Create a boolean value.
    #[inline]
    pub const fn bool_val(b: bool) -> u64 {
        let tag = ValueTag::Bool as u64;
        let val = if b { 1u64 << 3 } else { 0 };
        tag | val
    }

    /// Create an integer value.
    #[inline]
    pub const fn int_val(n: i64) -> u64 {
        let tag = ValueTag::Int as u64;
        let shifted = (n << 3) as u64;
        shifted | tag
    }

    /// Create a character value.
    #[inline]
    pub const fn char_val(c: char) -> u64 {
        let tag = ValueTag::Char as u64;
        let codepoint = (c as u64) << 3;
        codepoint | tag
    }

    /// Create a keyword value.
    #[inline]
    pub const fn keyword_val(kw_id: u32) -> u64 {
        let tag = ValueTag::Keyword as u64;
        let id = (kw_id as u64) << 3;
        id | tag
    }

    /// Create a symbol value.
    #[inline]
    pub const fn symbol_val(sym_id: u32) -> u64 {
        let tag = ValueTag::Symbol as u64;
        let id = (sym_id as u64) << 3;
        id | tag
    }

    /// Create a boxed pointer value.
    #[inline]
    pub const fn boxed_val(ptr: u64) -> u64 {
        // Pointers should be aligned, so low bits are zero
        // We just use the pointer as-is (tag is 0)
        ptr
    }

    /// Extract integer from tagged value.
    #[inline]
    pub const fn extract_int(value: u64) -> i64 {
        (value as i64) >> 3
    }

    /// Extract boolean from tagged value.
    #[inline]
    pub const fn extract_bool(value: u64) -> bool {
        (value >> 3) & 1 == 1
    }

    /// Extract character from tagged value.
    #[inline]
    pub const fn extract_char(value: u64) -> char {
        // Safety: we trust that valid char values were stored
        let codepoint = (value >> 3) as u32;
        // Use unsafe transmute since const fn can't use char::from_u32
        // In real code, validate the codepoint
        unsafe { std::mem::transmute(codepoint) }
    }

    /// Extract keyword ID from tagged value.
    #[inline]
    pub const fn extract_keyword(value: u64) -> u32 {
        (value >> 3) as u32
    }

    /// Extract symbol ID from tagged value.
    #[inline]
    pub const fn extract_symbol(value: u64) -> u32 {
        (value >> 3) as u32
    }

    /// Extract pointer from boxed value.
    #[inline]
    pub const fn extract_ptr(value: u64) -> *const u8 {
        // Mask off tag bits (though they should be 0 for boxed)
        (value & !ValueTag::MASK) as *const u8
    }

    /// Check if value is truthy (not nil and not false).
    #[inline]
    pub const fn is_truthy(value: u64) -> bool {
        let tag = value & ValueTag::MASK;
        if tag == ValueTag::Nil as u64 {
            return false;
        }
        if tag == ValueTag::Bool as u64 {
            return extract_bool(value);
        }
        true
    }

    /// Check if value is nil.
    #[inline]
    pub const fn is_nil(value: u64) -> bool {
        (value & ValueTag::MASK) == ValueTag::Nil as u64
    }

    /// Check if value is a boxed pointer.
    #[inline]
    pub const fn is_boxed(value: u64) -> bool {
        (value & ValueTag::MASK) == ValueTag::Boxed as u64
    }

    /// Maximum integer that can be represented as an immediate value.
    /// Since we use 61 bits for the integer (63 - 3 for tag + 1 for sign),
    /// the range is approximately -2^60 to 2^60-1.
    pub const MAX_IMMEDIATE_INT: i64 = (1i64 << 60) - 1;
    pub const MIN_IMMEDIATE_INT: i64 = -(1i64 << 60);
}

#[cfg(test)]
mod tests {
    use super::*;
    use helpers::*;

    #[test]
    fn test_nil() {
        let val = nil();
        assert_eq!(ValueTag::from_raw(val), ValueTag::Nil);
        assert!(is_nil(val));
        assert!(!is_truthy(val));
    }

    #[test]
    fn test_bool() {
        let true_val = bool_val(true);
        let false_val = bool_val(false);

        assert_eq!(ValueTag::from_raw(true_val), ValueTag::Bool);
        assert_eq!(ValueTag::from_raw(false_val), ValueTag::Bool);

        assert!(extract_bool(true_val));
        assert!(!extract_bool(false_val));

        assert!(is_truthy(true_val));
        assert!(!is_truthy(false_val));
    }

    #[test]
    fn test_int() {
        let val = int_val(42);
        assert_eq!(ValueTag::from_raw(val), ValueTag::Int);
        assert_eq!(extract_int(val), 42);
        assert!(is_truthy(val));

        let neg_val = int_val(-100);
        assert_eq!(extract_int(neg_val), -100);

        let zero_val = int_val(0);
        assert_eq!(extract_int(zero_val), 0);
        assert!(is_truthy(zero_val)); // 0 is truthy in Lisp
    }

    #[test]
    fn test_char() {
        let val = char_val('A');
        assert_eq!(ValueTag::from_raw(val), ValueTag::Char);
        assert_eq!(extract_char(val), 'A');

        let unicode_val = char_val('λ');
        assert_eq!(extract_char(unicode_val), 'λ');
    }

    #[test]
    fn test_keyword() {
        let val = keyword_val(123);
        assert_eq!(ValueTag::from_raw(val), ValueTag::Keyword);
        assert_eq!(extract_keyword(val), 123);
    }

    #[test]
    fn test_symbol() {
        let val = symbol_val(456);
        assert_eq!(ValueTag::from_raw(val), ValueTag::Symbol);
        assert_eq!(extract_symbol(val), 456);
    }

    #[test]
    fn test_int_range() {
        // Test max value
        let max_val = int_val(MAX_IMMEDIATE_INT);
        assert_eq!(extract_int(max_val), MAX_IMMEDIATE_INT);

        // Test min value
        let min_val = int_val(MIN_IMMEDIATE_INT);
        assert_eq!(extract_int(min_val), MIN_IMMEDIATE_INT);
    }
}
