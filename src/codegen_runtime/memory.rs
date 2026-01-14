//! Memory management with atomic reference counting for the Evolve runtime.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::codegen::value::ValueTag;

use super::error::allocation_failed;

/// Header for all boxed (heap-allocated) objects.
///
/// Layout:
/// ```text
/// +----------+------------+------------------+
/// | type_tag | ref_count  | Type-specific    |
/// | (u8)     | (u32)      | data...          |
/// +----------+------------+------------------+
/// ```
#[repr(C)]
pub struct BoxedHeader {
    pub type_tag: u8,
    _padding: [u8; 3],
    pub ref_count: AtomicU32,
}

impl BoxedHeader {
    pub const SIZE: usize = std::mem::size_of::<BoxedHeader>();
    pub const ALIGN: usize = std::mem::align_of::<BoxedHeader>();
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_alloc(size: i64) -> *mut u8 {
    if size <= 0 {
        return std::ptr::null_mut();
    }

    let total_size = size as usize;
    let layout = match Layout::from_size_align(total_size, BoxedHeader::ALIGN) {
        Ok(layout) => layout,
        Err(_) => allocation_failed(total_size),
    };

    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        allocation_failed(total_size);
    }

    let header = ptr as *mut BoxedHeader;
    unsafe {
        (*header).ref_count = AtomicU32::new(1);
    }

    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_retain(value: i64) {
    let value = value as u64;
    let tag = value & ValueTag::MASK;

    if tag != ValueTag::Boxed as u64 {
        return;
    }

    let ptr = (value & !ValueTag::MASK) as *mut BoxedHeader;
    if ptr.is_null() {
        return;
    }

    unsafe {
        (*ptr).ref_count.fetch_add(1, Ordering::AcqRel);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_release(value: i64) {
    let value = value as u64;
    let tag = value & ValueTag::MASK;

    if tag != ValueTag::Boxed as u64 {
        return;
    }

    let ptr = (value & !ValueTag::MASK) as *mut BoxedHeader;
    if ptr.is_null() {
        return;
    }

    let old_count = unsafe { (*ptr).ref_count.fetch_sub(1, Ordering::AcqRel) };

    if old_count == 1 {
        let header = unsafe { &*ptr };
        let total_size = get_object_size(header);
        let layout = Layout::from_size_align(total_size, BoxedHeader::ALIGN)
            .expect("invalid layout during deallocation");
        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
    }
}

fn get_object_size(header: &BoxedHeader) -> usize {
    use super::BoxedType;

    match header.type_tag {
        t if t == BoxedType::String as u8 => BoxedHeader::SIZE + 8 + 256, // estimate
        t if t == BoxedType::List as u8 => BoxedHeader::SIZE + 16,
        t if t == BoxedType::Vector as u8 => BoxedHeader::SIZE + 32,
        t if t == BoxedType::Map as u8 => BoxedHeader::SIZE + 32,
        t if t == BoxedType::Set as u8 => BoxedHeader::SIZE + 32,
        t if t == BoxedType::Closure as u8 => BoxedHeader::SIZE + 16,
        t if t == BoxedType::Var as u8 => BoxedHeader::SIZE + 16,
        t if t == BoxedType::Namespace as u8 => BoxedHeader::SIZE + 16,
        t if t == BoxedType::Float as u8 => BoxedHeader::SIZE + 8,
        _ => BoxedHeader::SIZE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_release() {
        let ptr = evolve_alloc(64);
        assert!(!ptr.is_null());

        let header = ptr as *mut BoxedHeader;
        unsafe {
            assert_eq!((*header).ref_count.load(Ordering::SeqCst), 1);
        }

        let value = ptr as i64;
        evolve_retain(value);
        unsafe {
            assert_eq!((*header).ref_count.load(Ordering::SeqCst), 2);
        }

        evolve_release(value);
        unsafe {
            assert_eq!((*header).ref_count.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn test_non_boxed_values_ignored() {
        let int_value = (42i64 << 3) | (ValueTag::Int as i64);
        evolve_retain(int_value);
        evolve_release(int_value);
    }
}
