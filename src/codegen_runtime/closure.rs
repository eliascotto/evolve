//! Closure runtime support for Evolve.

use std::sync::atomic::AtomicU32;

use super::memory::BoxedHeader;
use super::value::{extract_ptr, make_boxed};
use super::BoxedType;

#[repr(C)]
pub struct ClosureObject {
    pub header: BoxedHeader,
    pub function_ptr: *const (),
    pub env_count: i32,
    _padding: i32,
}

impl ClosureObject {
    fn env_ptr(&self) -> *const i64 {
        unsafe {
            let base = self as *const Self;
            base.add(1) as *const i64
        }
    }

    fn env_ptr_mut(&mut self) -> *mut i64 {
        unsafe {
            let base = self as *mut Self;
            base.add(1) as *mut i64
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_closure_new(
    fn_ptr: *const (),
    env_count: i32,
    env: *const i64,
) -> i64 {
    let header_size = std::mem::size_of::<ClosureObject>();
    let env_size = (env_count as usize) * std::mem::size_of::<i64>();
    let total_size = header_size + env_size;

    let ptr = super::memory::evolve_alloc(total_size as i64);
    if ptr.is_null() {
        panic!("Failed to allocate closure");
    }

    unsafe {
        let closure = ptr as *mut ClosureObject;
        (*closure).header.type_tag = BoxedType::Closure as u8;
        (*closure).header.ref_count = AtomicU32::new(1);
        (*closure).function_ptr = fn_ptr;
        (*closure).env_count = env_count;

        if env_count > 0 && !env.is_null() {
            let dest = (*closure).env_ptr_mut();
            for i in 0..env_count as isize {
                let val = *env.offset(i);
                super::evolve_retain(val);
                *dest.offset(i) = val;
            }
        }
    }

    make_boxed(ptr)
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_closure_get_fn(closure: i64) -> *const () {
    let ptr = extract_ptr(closure);
    if ptr.is_null() {
        return std::ptr::null();
    }
    unsafe {
        let obj = ptr as *const ClosureObject;
        (*obj).function_ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_closure_get_env(closure: i64) -> *const i64 {
    let ptr = extract_ptr(closure);
    if ptr.is_null() {
        return std::ptr::null();
    }
    unsafe {
        let obj = ptr as *const ClosureObject;
        (*obj).env_ptr()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_closure_env_count(closure: i64) -> i32 {
    let ptr = extract_ptr(closure);
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let obj = ptr as *const ClosureObject;
        (*obj).env_count
    }
}
