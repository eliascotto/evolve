use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::core::namespace;
use crate::env::Env;
use crate::error::{Diagnostic, SpannedError};
use crate::eval::Evaluator;
use crate::reader::{Reader, Source};
use crate::value::Value;
use crate::core::Symbol;
use crate::interner::SymId;

pub type RuntimeRef = Arc<Runtime>;

#[derive(Debug)]
pub struct Runtime {
    pub evaluator: Evaluator,
    env: Env,
    id: AtomicU32,
}

impl Runtime {
    pub fn new() -> Arc<Self> {
        // Create the user default namespace and initial environment
        let user_ns = namespace::ns_find_or_create("user");
        let env = Env::new(user_ns);

        Arc::new(Self {
            evaluator: Evaluator::new(),
            env: env,
            id: AtomicU32::new(0),
        })
    }

    /// Evaluates the input and returns the result as a string.
    pub fn rep(
        self: Arc<Self>,
        input: &str,
        file: Source,
    ) -> Result<Value, Diagnostic> {
        let ast = Reader::read(input, file.clone(), self.clone())?;

        let value = match self.evaluator.eval(&ast, &mut self.env.clone()) {
            Ok(value) => value,
            Err(eval_err) => {
                let SpannedError { error, span } = eval_err;
                return Err(Diagnostic {
                    error,
                    span,
                    source: input.to_string(),
                    file: file.clone(),
                    secondary_spans: None,
                    notes: None,
                });
            }
        };

        Ok(value)
    }

    /// Returns the next unique identifier.
    pub fn next_id(&self) -> u32 {
        self.id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn get_native_fn_sym(&self, name: &str) -> SymId {
        self.evaluator.native_fns.sym_for_name(name).unwrap()
    }

    // TODO move it into compiler
    pub fn resolve_symbol(&self, sym: &mut Symbol) -> Symbol {
        if sym.is_qualified() || self.evaluator.is_special_form(sym.id()) {
            return sym.clone();
        }

        match self.env.get(sym.id()) {
            Some(value) => match value {
                Value::Var { value: var, .. } => {
                    sym.set_namespace(var.ns.id());
                    return sym.clone();
                }
                _ => {}
            },
            None => {}
        }

        sym.set_namespace(self.env.ns.id());
        sym.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rep() {
        let runtime = Runtime::new();
        let result = runtime.rep("(count [1 2 3])", Source::REPL).unwrap();
        assert_eq!(result.to_string(), "3");
    }
}
