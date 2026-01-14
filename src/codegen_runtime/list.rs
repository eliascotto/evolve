//! List operations for the Evolve codegen runtime.

use std::sync::atomic::{AtomicU32, Ordering};

use super::memory::{evolve_alloc, evolve_retain, BoxedHeader};
use super::value::{extract_ptr, is_boxed, make_boxed, make_nil, BoxedType};

#[repr(C)]
pub struct ListNode {
    pub value: i64,
    pub next: *mut ListNode,
    pub ref_count: AtomicU32,
}

#[repr(C)]
pub struct ListObject {
    pub header: BoxedHeader,
    pub length: i64,
    pub head: *mut ListNode,
}

fn alloc_node(value: i64, next: *mut ListNode) -> *mut ListNode {
    let size = std::mem::size_of::<ListNode>();

    let ptr = evolve_alloc(size as i64);
    if ptr.is_null() {
        super::error::evolve_panic_cstr(c"Failed to allocate list node");
    }

    unsafe {
        let node = ptr as *mut ListNode;
        std::ptr::write(&mut (*node).value, value);
        std::ptr::write(&mut (*node).next, next);
        std::ptr::write(&mut (*node).ref_count, AtomicU32::new(1));
        node
    }
}

fn retain_node(node: *mut ListNode) {
    if !node.is_null() {
        unsafe {
            (*node).ref_count.fetch_add(1, Ordering::AcqRel);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_list_new(count: i32, items: *const i64) -> i64 {
    let count = count as usize;

    let size = std::mem::size_of::<ListObject>();
    let ptr = evolve_alloc(size as i64);
    if ptr.is_null() {
        super::error::evolve_panic_cstr(c"Failed to allocate list");
    }

    unsafe {
        let list_obj = ptr as *mut ListObject;
        (*list_obj).header.type_tag = BoxedType::List as u8;

        if count == 0 {
            (*list_obj).length = 0;
            (*list_obj).head = std::ptr::null_mut();
            return make_boxed(ptr);
        }

        let mut head: *mut ListNode = std::ptr::null_mut();
        for i in (0..count).rev() {
            let item = *items.add(i);
            evolve_retain(item);
            head = alloc_node(item, head);
        }

        (*list_obj).length = count as i64;
        (*list_obj).head = head;

        make_boxed(ptr)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_list_first(list: i64) -> i64 {
    if !is_boxed(list) {
        return make_nil();
    }

    let ptr = extract_ptr(list);
    if ptr.is_null() {
        return make_nil();
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        if (*header).type_tag != BoxedType::List as u8 {
            return make_nil();
        }

        let list_obj = ptr as *const ListObject;
        let head = (*list_obj).head;

        if head.is_null() {
            return make_nil();
        }

        (*head).value
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_list_rest(list: i64) -> i64 {
    if !is_boxed(list) {
        super::error::evolve_panic_cstr(c"list_rest: not a boxed value");
    }

    let ptr = extract_ptr(list);
    if ptr.is_null() {
        super::error::evolve_panic_cstr(c"list_rest: null pointer");
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        if (*header).type_tag != BoxedType::List as u8 {
            super::error::evolve_panic_cstr(c"list_rest: not a list");
        }

        let list_obj = ptr as *const ListObject;
        let head = (*list_obj).head;

        if head.is_null() {
            return evolve_list_new(0, std::ptr::null());
        }

        let rest_head = (*head).next;
        let new_len = (*list_obj).length - 1;

        let size = std::mem::size_of::<ListObject>();
        let new_ptr = evolve_alloc(size as i64);
        if new_ptr.is_null() {
            super::error::evolve_panic_cstr(c"Failed to allocate list");
        }

        let new_list = new_ptr as *mut ListObject;
        (*new_list).header.type_tag = BoxedType::List as u8;
        (*new_list).length = new_len;
        (*new_list).head = rest_head;

        retain_node(rest_head);

        make_boxed(new_ptr)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_list_count(list: i64) -> i64 {
    if !is_boxed(list) {
        return 0;
    }

    let ptr = extract_ptr(list);
    if ptr.is_null() {
        return 0;
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        if (*header).type_tag != BoxedType::List as u8 {
            return 0;
        }

        let list_obj = ptr as *const ListObject;
        (*list_obj).length
    }
}
