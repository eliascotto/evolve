//! Error handling for the Evolve runtime.

use std::ffi::c_char;
use std::ffi::CStr;

#[unsafe(no_mangle)]
pub extern "C" fn evolve_panic(msg: *const c_char) {
    let message = if msg.is_null() {
        "unknown error"
    } else {
        unsafe { CStr::from_ptr(msg) }
            .to_str()
            .unwrap_or("invalid UTF-8 in error message")
    };
    eprintln!("Evolve runtime panic: {}", message);
    std::process::abort();
}

pub(crate) fn runtime_error(msg: &str) -> ! {
    eprintln!("Evolve runtime error: {}", msg);
    std::process::abort();
}

pub(crate) fn allocation_failed(size: usize) -> ! {
    runtime_error(&format!("allocation of {} bytes failed", size));
}

pub(crate) fn evolve_panic_cstr(msg: &CStr) -> ! {
    let message = msg.to_str().unwrap_or("invalid UTF-8 in error message");
    eprintln!("Evolve runtime panic: {}", message);
    std::process::abort();
}
