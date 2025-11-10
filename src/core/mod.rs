//! Language-specific functionality.

pub mod namespace;
pub mod recur;
pub mod var;
pub mod native_fns;

pub use namespace::Namespace;
pub use recur::RecurContext;
pub use var::Var;
