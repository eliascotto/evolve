//! Vector operations for the Evolve codegen runtime.

use super::memory::{evolve_alloc, evolve_retain, BoxedHeader};
use super::value::{extract_ptr, is_boxed, make_boxed, make_nil, BoxedType};

#[repr(C)]
pub struct VectorObject {
    pub header: BoxedHeader,
    pub length: i64,
    pub capacity: i64,
}

impl VectorObject {
    #[inline]
    pub fn items_ptr(&self) -> *const i64 {
        unsafe { (self as *const Self).add(1) as *const i64 }
    }

    #[inline]
    pub fn items_ptr_mut(&mut self) -> *mut i64 {
        unsafe { (self as *mut Self).add(1) as *mut i64 }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_vector_new(count: i32, items: *const i64) -> i64 {
    let count = count as usize;
    let size = std::mem::size_of::<VectorObject>() + count * std::mem::size_of::<i64>();

    let ptr = evolve_alloc(size as i64);
    if ptr.is_null() {
        super::error::evolve_panic_cstr(c"Failed to allocate vector");
    }

    unsafe {
        let vec_obj = ptr as *mut VectorObject;
        (*vec_obj).header.type_tag = BoxedType::Vector as u8;
        (*vec_obj).length = count as i64;
        (*vec_obj).capacity = count as i64;

        let items_dest = (*vec_obj).items_ptr_mut();
        for i in 0..count {
            let item = *items.add(i);
            evolve_retain(item);
            *items_dest.add(i) = item;
        }

        make_boxed(ptr)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_vector_get(vec: i64, index: i64) -> i64 {
    if !is_boxed(vec) {
        super::error::evolve_panic_cstr(c"vector_get: not a boxed value");
    }

    let ptr = extract_ptr(vec);
    if ptr.is_null() {
        super::error::evolve_panic_cstr(c"vector_get: null pointer");
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        if (*header).type_tag != BoxedType::Vector as u8 {
            super::error::evolve_panic_cstr(c"vector_get: not a vector");
        }

        let vec_obj = ptr as *const VectorObject;
        let len = (*vec_obj).length;

        if index < 0 || index >= len {
            super::error::evolve_panic_cstr(c"vector_get: index out of bounds");
        }

        let items = (*vec_obj).items_ptr();
        *items.add(index as usize)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_vector_count(vec: i64) -> i64 {
    if !is_boxed(vec) {
        return 0;
    }

    let ptr = extract_ptr(vec);
    if ptr.is_null() {
        return 0;
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        if (*header).type_tag != BoxedType::Vector as u8 {
            return 0;
        }

        let vec_obj = ptr as *const VectorObject;
        (*vec_obj).length
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_vector_rest(vec: i64, start: i64) -> i64 {
    if !is_boxed(vec) {
        super::error::evolve_panic_cstr(c"vector_rest: not a boxed value");
    }

    let ptr = extract_ptr(vec);
    if ptr.is_null() {
        super::error::evolve_panic_cstr(c"vector_rest: null pointer");
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        if (*header).type_tag != BoxedType::Vector as u8 {
            super::error::evolve_panic_cstr(c"vector_rest: not a vector");
        }

        let vec_obj = ptr as *const VectorObject;
        let len = (*vec_obj).length;

        if start < 0 || start > len {
            super::error::evolve_panic_cstr(c"vector_rest: start out of bounds");
        }

        let new_len = len - start;
        if new_len <= 0 {
            return evolve_vector_new(0, std::ptr::null());
        }

        let items = (*vec_obj).items_ptr();
        let start_ptr = items.add(start as usize);
        evolve_vector_new(new_len as i32, start_ptr)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_vector_first(vec: i64) -> i64 {
    if !is_boxed(vec) {
        return make_nil();
    }

    let ptr = extract_ptr(vec);
    if ptr.is_null() {
        return make_nil();
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        if (*header).type_tag != BoxedType::Vector as u8 {
            return make_nil();
        }

        let vec_obj = ptr as *const VectorObject;
        if (*vec_obj).length == 0 {
            return make_nil();
        }

        let items = (*vec_obj).items_ptr();
        *items
    }
}
