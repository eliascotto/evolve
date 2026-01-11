//! Module loader for loading Evolve source files.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use once_cell::sync::Lazy;
use rustc_hash::FxHashSet;

use crate::core::namespace::find_or_create_ns;
use crate::error::{Error, SpannedError};
use crate::interner::NsId;
use crate::reader::Source;
use crate::runtime::RuntimeRef;
use crate::synthetic_span;

/// Global module loader state
static MODULE_LOADER: Lazy<Mutex<ModuleLoaderState>> =
    Lazy::new(|| Mutex::new(ModuleLoaderState::new()));

/// State for the module loader
struct ModuleLoaderState {
    /// Search paths for finding module files
    search_paths: Vec<PathBuf>,
    /// Set of already loaded namespaces (to prevent re-loading)
    loaded: FxHashSet<NsId>,
    /// Set of currently loading namespaces (to detect circular dependencies)
    loading: FxHashSet<NsId>,
}

impl ModuleLoaderState {
    fn new() -> Self {
        Self {
            search_paths: vec![PathBuf::from(".")],
            loaded: FxHashSet::default(),
            loading: FxHashSet::default(),
        }
    }
}

/// Adds a search path for module loading
pub fn add_search_path(path: impl AsRef<Path>) {
    let mut state = MODULE_LOADER.lock().unwrap();
    state.search_paths.push(path.as_ref().to_path_buf());
}

/// Sets the search paths for module loading
pub fn set_search_paths(paths: Vec<PathBuf>) {
    let mut state = MODULE_LOADER.lock().unwrap();
    state.search_paths = paths;
}

/// Gets the current search paths
pub fn get_search_paths() -> Vec<PathBuf> {
    let state = MODULE_LOADER.lock().unwrap();
    state.search_paths.clone()
}

/// Checks if a namespace has been loaded
pub fn is_loaded(ns_id: NsId) -> bool {
    let state = MODULE_LOADER.lock().unwrap();
    state.loaded.contains(&ns_id)
}

/// Marks a namespace as loaded
pub fn mark_loaded(ns_id: NsId) {
    let mut state = MODULE_LOADER.lock().unwrap();
    state.loaded.insert(ns_id);
    state.loading.remove(&ns_id);
}

/// Checks if a namespace is currently being loaded (circular dependency check)
pub fn is_loading(ns_id: NsId) -> bool {
    let state = MODULE_LOADER.lock().unwrap();
    state.loading.contains(&ns_id)
}

/// Marks a namespace as currently loading
pub fn mark_loading(ns_id: NsId) {
    let mut state = MODULE_LOADER.lock().unwrap();
    state.loading.insert(ns_id);
}

/// Clears the loading state (used on error)
pub fn clear_loading(ns_id: NsId) {
    let mut state = MODULE_LOADER.lock().unwrap();
    state.loading.remove(&ns_id);
}

/// Converts a namespace name to a file path
/// e.g., "foo.bar.baz" -> "foo/bar/baz.ev"
pub fn ns_to_path(ns_name: &str) -> PathBuf {
    let parts: Vec<&str> = ns_name.split('.').collect();
    let mut path = PathBuf::new();
    for part in &parts[..parts.len() - 1] {
        path.push(part);
    }
    path.push(format!("{}.ev", parts[parts.len() - 1]));
    path
}

/// Finds the file path for a namespace
pub fn find_module_file(ns_name: &str) -> Option<PathBuf> {
    let state = MODULE_LOADER.lock().unwrap();
    let relative_path = ns_to_path(ns_name);

    for search_path in &state.search_paths {
        let full_path = search_path.join(&relative_path);
        if full_path.exists() {
            return Some(full_path);
        }
    }

    // Also try .evl extension
    let relative_path_evl = {
        let mut p = relative_path.clone();
        p.set_extension("evl");
        p
    };
    for search_path in &state.search_paths {
        let full_path = search_path.join(&relative_path_evl);
        if full_path.exists() {
            return Some(full_path);
        }
    }

    None
}

/// Loads a module from a file
pub fn load_module(
    ns_name: &str,
    runtime: RuntimeRef,
) -> Result<(), SpannedError> {
    let ns = find_or_create_ns(ns_name);

    // Check if already loaded
    if is_loaded(ns.id) {
        return Ok(());
    }

    // Check for circular dependency
    if is_loading(ns.id) {
        return Err(SpannedError {
            error: Error::RuntimeError(format!(
                "Circular dependency detected while loading namespace: {}",
                ns_name
            )),
            span: synthetic_span!(),
        });
    }

    // Find the module file
    let file_path = find_module_file(ns_name).ok_or_else(|| SpannedError {
        error: Error::RuntimeError(format!(
            "Module not found: {} (searched in {:?})",
            ns_name,
            get_search_paths()
        )),
        span: synthetic_span!(),
    })?;

    // Mark as loading
    mark_loading(ns.id);

    // Read the file
    let source = std::fs::read_to_string(&file_path).map_err(|e| SpannedError {
        error: Error::RuntimeError(format!(
            "Failed to read module file {}: {}",
            file_path.display(),
            e
        )),
        span: synthetic_span!(),
    })?;

    // Parse and evaluate the file
    let file_source = Source::File(file_path.clone());
    let result = runtime.clone().rep(&source, file_source);

    match result {
        Ok(_) => {
            mark_loaded(ns.id);
            Ok(())
        }
        Err(diagnostic) => {
            clear_loading(ns.id);
            Err(SpannedError {
                error: Error::RuntimeError(format!(
                    "Error loading module {}: {:?}",
                    ns_name, diagnostic.error
                )),
                span: diagnostic.span,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ns_to_path() {
        assert_eq!(ns_to_path("foo"), PathBuf::from("foo.ev"));
        assert_eq!(ns_to_path("foo.bar"), PathBuf::from("foo/bar.ev"));
        assert_eq!(ns_to_path("foo.bar.baz"), PathBuf::from("foo/bar/baz.ev"));
    }
}
