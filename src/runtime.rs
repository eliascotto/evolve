use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::core::namespace;
use crate::env::Env;
use crate::error::{Diagnostic, Error};
use crate::eval::Evaluator;
use crate::reader::{Reader, Source};
use crate::value::Value;

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
    pub fn rep(self: Arc<Self>, input: &str, file: Source) -> Result<Value, Diagnostic> {
        let ast = Reader::read(input, file.clone(), self.clone())?;

        let value = match self.evaluator.eval(&ast, &mut self.env.clone()) {
            Ok(value) => value,
            Err(e) => {
                return Err(Diagnostic {
                    error: Error::RuntimeError(e.to_string()),
                    span: 0..0,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rep() {
        let runtime = Runtime::new();
        let result = runtime.rep("(+ 1 2 3)", Source::REPL).unwrap();
        assert_eq!(result.to_string(), "6");
    }
}
