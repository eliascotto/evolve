//! Dynamic dispatch for Evolve runtime.

use super::closure::ClosureObject;
use super::memory::BoxedHeader;
use super::value::{extract_ptr, is_boxed};
use super::BoxedType;

pub type EvolveFunction = extern "C" fn(*const i64, i32, *const i64) -> i64;

#[repr(C)]
struct VarObject {
    header: BoxedHeader,
    sym_id: i64,
    value: i64,
}

#[unsafe(no_mangle)]
pub extern "C" fn evolve_call(fn_val: i64, argc: i32, argv: *const i64) -> i64 {
    if !is_boxed(fn_val) {
        panic!("evolve_call: not a function (not boxed)");
    }

    let ptr = extract_ptr(fn_val);
    if ptr.is_null() {
        panic!("evolve_call: null function pointer");
    }

    unsafe {
        let header = ptr as *const BoxedHeader;
        let boxed_type = (*header).type_tag;

        match boxed_type {
            t if t == BoxedType::Closure as u8 => {
                let closure = ptr as *const ClosureObject;
                let function_ptr = (*closure).function_ptr;
                let env_ptr = (closure.add(1)) as *const i64;

                let func: EvolveFunction = std::mem::transmute(function_ptr);
                func(env_ptr, argc, argv)
            }
            t if t == BoxedType::Var as u8 => {
                let var = ptr as *const VarObject;
                let inner_val = (*var).value;
                evolve_call(inner_val, argc, argv)
            }
            _ => {
                panic!("evolve_call: not a function (boxed_type={})", boxed_type);
            }
        }
    }
}
