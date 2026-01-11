//! Language-specific functionality.

pub mod metadata;
pub mod module_loader;
pub mod namespace;
pub mod native_fns;
pub mod recur;
pub mod special_forms;
pub mod symbol;
pub mod var;

pub use metadata::Metadata;
pub use namespace::{
    find_or_create_ns, get_current_ns, lookup_ns, set_current_ns, update_ns, Namespace,
    NamespaceRegistry,
};
pub use recur::RecurContext;
pub use special_forms::SpecialFormRegistry;
pub use symbol::Symbol;
pub use var::Var;
