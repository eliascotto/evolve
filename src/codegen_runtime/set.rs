//! Set runtime support for Evolve.

use std::sync::atomic::AtomicU32;

use super::memory::BoxedHeader;
use super::value::{extract_ptr, make_boxed};
use super::BoxedType;

#[repr(C)]
pub struct SetObject {
    pub header: BoxedHeader,
    pub size: i64,
    pub capacity: i64,
}

impl SetObject {
    fn items_ptr(&self) -> *const i64 {
        unsafe {
            let base = self as *const Self;
            base.add(1) as *const i64
        }
    }

    fn items_ptr_mut(&mut self) -> *mut i64 {
        unsafe {
            let base = self as *mut Self;
            base.add(1) as *mut i64
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_set_new(count: i32, items: *const i64) -> i64 {
    let header_size = std::mem::size_of::<SetObject>();
    let items_size = (count as usize) * std::mem::size_of::<i64>();
    let total_size = header_size + items_size;

    let ptr = super::memory::evolve_alloc(total_size as i64);
    if ptr.is_null() {
        panic!("Failed to allocate set");
    }

    unsafe {
        let set = ptr as *mut SetObject;
        (*set).header.type_tag = BoxedType::Set as u8;
        (*set).header.ref_count = AtomicU32::new(1);
        (*set).size = count as i64;
        (*set).capacity = count as i64;

        if count > 0 && !items.is_null() {
            let dest = (*set).items_ptr_mut();
            for i in 0..count as isize {
                let item = *items.offset(i);
                super::evolve_retain(item);
                *dest.offset(i) = item;
            }
        }
    }

    make_boxed(ptr)
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_set_contains(set: i64, item: i64) -> bool {
    let ptr = extract_ptr(set);
    if ptr.is_null() {
        return false;
    }

    unsafe {
        let obj = ptr as *const SetObject;
        let size = (*obj).size as isize;
        let items = (*obj).items_ptr();

        for i in 0..size {
            if *items.offset(i) == item {
                return true;
            }
        }
    }

    false
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_set_count(set: i64) -> i64 {
    let ptr = extract_ptr(set);
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let obj = ptr as *const SetObject;
        (*obj).size
    }
}
