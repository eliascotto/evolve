//! String operations for the Evolve codegen runtime.
//!
//! Evolve strings are heap-allocated boxed objects with the following layout:
//!
//! ```text
//! +----------------+--------+------------------+
//! | BoxedHeader    | Length | String data...   |
//! | (8 bytes)      | (i64)  | (null-terminated)|
//! +----------------+--------+------------------+
//! ```

use std::sync::atomic::AtomicU32;

use super::memory::{evolve_alloc, BoxedHeader};
use super::BoxedType;

/// String object layout.
///
/// Note: This struct represents the fixed-size prefix of a string object.
/// The actual string data follows immediately after in memory.
#[repr(C)]
pub struct StringObject {
    /// Object header with type tag and ref count
    pub header: BoxedHeader,
    /// Length of the string in bytes (not including null terminator)
    pub length: i64,
    // String data follows immediately after (flexible array member pattern)
}

impl StringObject {
    /// Size of the fixed portion (header + length)
    pub const PREFIX_SIZE: usize = std::mem::size_of::<StringObject>();

    /// Get a pointer to the string data
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        unsafe { (self as *const Self as *const u8).add(Self::PREFIX_SIZE) }
    }

    /// Get a mutable pointer to the string data
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        unsafe { (self as *mut Self as *mut u8).add(Self::PREFIX_SIZE) }
    }
}

/// Create a new string from raw bytes.
///
/// Allocates a new string object, copies the data, and returns a tagged pointer.
///
/// # Arguments
/// * `data` - Pointer to the string bytes (does not need to be null-terminated)
/// * `len` - Length of the string in bytes
///
/// # Returns
/// A tagged pointer to the new string (tag 0b000 for boxed)
///
/// # Safety
/// - `data` must point to at least `len` valid bytes
/// - Returns 0 (null) on allocation failure
#[unsafe(no_mangle)]
pub extern "C" fn evolve_string_new(data: *const u8, len: i64) -> i64 {
    if len < 0 {
        return 0;
    }

    let len_usize = len as usize;
    let total_size = StringObject::PREFIX_SIZE + len_usize + 1; // +1 for null terminator

    unsafe {
        let ptr = evolve_alloc(total_size as i64);
        if ptr.is_null() {
            return 0;
        }

        let string_obj = ptr as *mut StringObject;

        // Initialize header
        (*string_obj).header.type_tag = BoxedType::String as u8;
        (*string_obj).header.ref_count = AtomicU32::new(1);
        (*string_obj).length = len;

        // Copy string data
        let data_dst = (*string_obj).data_ptr_mut();
        if !data.is_null() && len > 0 {
            std::ptr::copy_nonoverlapping(data, data_dst, len_usize);
        }

        // Null-terminate
        *data_dst.add(len_usize) = 0;

        // Return as tagged pointer (tag 0b000 for boxed)
        ptr as i64
    }
}

/// Get the length of a string.
///
/// # Arguments
/// * `str_val` - A tagged pointer to a string object
///
/// # Returns
/// The length of the string in bytes
///
/// # Safety
/// The caller must ensure `str_val` points to a valid StringObject.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_string_len(str_val: i64) -> i64 {
    if str_val == 0 {
        return 0;
    }

    unsafe {
        let ptr = (str_val & !0b111) as *const StringObject;
        (*ptr).length
    }
}

/// Get a pointer to the string data.
///
/// # Arguments
/// * `str_val` - A tagged pointer to a string object
///
/// # Returns
/// A pointer to the null-terminated string data
///
/// # Safety
/// The caller must ensure `str_val` points to a valid StringObject.
/// The returned pointer is valid as long as the string object is alive.
#[unsafe(no_mangle)]
pub extern "C" fn evolve_string_data(str_val: i64) -> *const u8 {
    if str_val == 0 {
        return std::ptr::null();
    }

    unsafe {
        let ptr = (str_val & !0b111) as *const StringObject;
        (*ptr).data_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_object_size() {
        // Header (8) + length (8) = 16
        assert_eq!(StringObject::PREFIX_SIZE, 16);
    }

    #[test]
    fn test_string_new_and_access() {
        let test_str = b"hello";
        let str_val = evolve_string_new(test_str.as_ptr(), test_str.len() as i64);
        assert_ne!(str_val, 0);

        // Check length
        assert_eq!(evolve_string_len(str_val), 5);

        // Check data
        let data_ptr = evolve_string_data(str_val);
        assert!(!data_ptr.is_null());

        unsafe {
            let slice = std::slice::from_raw_parts(data_ptr, 5);
            assert_eq!(slice, b"hello");

            // Check null terminator
            assert_eq!(*data_ptr.add(5), 0);
        }
    }

    #[test]
    fn test_empty_string() {
        let str_val = evolve_string_new(std::ptr::null(), 0);
        assert_ne!(str_val, 0);
        assert_eq!(evolve_string_len(str_val), 0);

        let data_ptr = evolve_string_data(str_val);
        unsafe {
            // Should be null-terminated
            assert_eq!(*data_ptr, 0);
        }
    }

    #[test]
    fn test_null_input() {
        assert_eq!(evolve_string_len(0), 0);
        assert!(evolve_string_data(0).is_null());
    }

    #[test]
    fn test_negative_length() {
        let str_val = evolve_string_new(std::ptr::null(), -1);
        assert_eq!(str_val, 0);
    }
}
