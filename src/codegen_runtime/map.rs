//! Map runtime support for Evolve.

use std::sync::atomic::AtomicU32;

use super::memory::BoxedHeader;
use super::value::{extract_ptr, make_boxed, make_nil};
use super::BoxedType;

#[repr(C)]
pub struct MapEntry {
    pub key: i64,
    pub value: i64,
}

#[repr(C)]
pub struct MapObject {
    pub header: BoxedHeader,
    pub size: i64,
    pub capacity: i64,
}

impl MapObject {
    fn entries_ptr(&self) -> *const MapEntry {
        unsafe {
            let base = self as *const Self;
            base.add(1) as *const MapEntry
        }
    }

    fn entries_ptr_mut(&mut self) -> *mut MapEntry {
        unsafe {
            let base = self as *mut Self;
            base.add(1) as *mut MapEntry
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_map_new(count: i32, entries: *const i64) -> i64 {
    let header_size = std::mem::size_of::<MapObject>();
    let entries_size = (count as usize) * std::mem::size_of::<MapEntry>();
    let total_size = header_size + entries_size;

    let ptr = super::memory::evolve_alloc(total_size as i64);
    if ptr.is_null() {
        panic!("Failed to allocate map");
    }

    unsafe {
        let map = ptr as *mut MapObject;
        (*map).header.type_tag = BoxedType::Map as u8;
        (*map).header.ref_count = AtomicU32::new(1);
        (*map).size = count as i64;
        (*map).capacity = count as i64;

        if count > 0 && !entries.is_null() {
            let dest = (*map).entries_ptr_mut();
            for i in 0..count as isize {
                let key = *entries.offset(i * 2);
                let value = *entries.offset(i * 2 + 1);
                super::evolve_retain(key);
                super::evolve_retain(value);
                *dest.offset(i) = MapEntry { key, value };
            }
        }
    }

    make_boxed(ptr)
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_map_get(map: i64, key: i64) -> i64 {
    let ptr = extract_ptr(map);
    if ptr.is_null() {
        return make_nil();
    }

    unsafe {
        let obj = ptr as *const MapObject;
        let size = (*obj).size as isize;
        let entries = (*obj).entries_ptr();

        for i in 0..size {
            let entry = &*entries.offset(i);
            if entry.key == key {
                return entry.value;
            }
        }
    }

    make_nil()
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_map_count(map: i64) -> i64 {
    let ptr = extract_ptr(map);
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let obj = ptr as *const MapObject;
        (*obj).size
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_map_contains(map: i64, key: i64) -> bool {
    let ptr = extract_ptr(map);
    if ptr.is_null() {
        return false;
    }

    unsafe {
        let obj = ptr as *const MapObject;
        let size = (*obj).size as isize;
        let entries = (*obj).entries_ptr();

        for i in 0..size {
            let entry = &*entries.offset(i);
            if entry.key == key {
                return true;
            }
        }
    }

    false
}
