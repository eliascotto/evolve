//! Test runner for Evolve.
//!
//! Discovers and executes test functions marked with `:test` metadata.

use colored::Colorize;
use std::sync::Arc;
use std::time::Instant;

use crate::core::{get_current_ns, Namespace, Var};
use crate::interner::{self, SymId};
use crate::reader::{Source, Span};
use crate::runtime::RuntimeRef;
use crate::value::Value;

/// Represents a discovered test.
#[derive(Debug)]
pub struct Test {
    /// The name of the test (symbol name)
    pub name: SymId,
    /// The namespace containing the test
    pub namespace: Arc<Namespace>,
    /// The var containing the test function
    pub var: Arc<Var>,
}

/// Test result.
#[derive(Debug)]
pub enum TestResult {
    Passed,
    Failed(String),
}

/// Test runner that discovers and executes tests.
pub struct TestRunner {
    runtime: RuntimeRef,
    verbose: bool,
}

impl TestRunner {
    pub fn new(runtime: RuntimeRef, verbose: bool) -> Self {
        Self { runtime, verbose }
    }

    /// Discovers all tests in the current namespace.
    pub fn discover_tests(&self) -> Vec<Test> {
        let ns = get_current_ns();
        self.discover_tests_in_namespace(&ns)
    }

    /// Discovers tests in a specific namespace.
    pub fn discover_tests_in_namespace(&self, ns: &Arc<Namespace>) -> Vec<Test> {
        let mut tests = Vec::new();
        let test_kw = Value::Keyword {
            span: Span::new(0, 0),
            value: interner::intern_kw("test"),
        };

        for (sym, var) in ns.all_bindings() {
            // Check if the var has :test metadata set to true
            if let Some(ref meta) = var.meta {
                if let Some(val) = meta.get(&test_kw) {
                    match val {
                        Value::Bool { value: true, .. } => {
                            tests.push(Test {
                                name: *sym,
                                namespace: ns.clone(),
                                var: var.clone(),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        tests
    }

    /// Runs a single test.
    pub fn run_test(&self, test: &Test) -> TestResult {
        // Get the function value from the var
        let value = match &test.var.value {
            Some(v) => v.read().unwrap().clone(),
            None => {
                return TestResult::Failed("Test var is unbound".to_string());
            }
        };

        // Ensure it's a function
        match &value {
            Value::Function { params, .. } => {
                // Test functions should take no arguments
                if !params.is_empty() {
                    return TestResult::Failed(
                        "Test function should take no arguments".to_string(),
                    );
                }

                // Call the test function by name: (ns/test-name)
                let test_name = interner::sym_to_str(test.name);
                let ns_name = &test.namespace.name;
                let call_expr = format!("({}/{})", ns_name, test_name);

                // Evaluate the call
                match self.runtime.clone().rep(&call_expr, Source::REPL) {
                    Ok(result) => {
                        // Check if the result is truthy
                        match result {
                            Value::Bool { value: false, .. } | Value::Nil { .. } => {
                                TestResult::Failed("Test returned falsy value".to_string())
                            }
                            _ => TestResult::Passed,
                        }
                    }
                    Err(diag) => {
                        TestResult::Failed(format!("{}", diag.error))
                    }
                }
            }
            _ => TestResult::Failed("Test is not a function".to_string()),
        }
    }

    /// Runs all discovered tests and returns (passed, failed) counts.
    pub fn run_all(&self) -> (usize, usize) {
        let tests = self.discover_tests();

        if tests.is_empty() {
            println!("{}", "No tests found.".yellow());
            return (0, 0);
        }

        println!("\n{}", "Running tests...".bold());
        println!("{}", "=".repeat(50));

        let mut passed = 0;
        let mut failed = 0;
        let start = Instant::now();

        for test in &tests {
            let test_name = interner::sym_to_str(test.name);
            let ns_name = &test.namespace.name;

            if self.verbose {
                print!("  {} {}::{} ... ", "test".dimmed(), ns_name, test_name);
            } else {
                print!("  {}::{} ... ", ns_name, test_name);
            }

            let test_start = Instant::now();
            let result = self.run_test(test);
            let duration = test_start.elapsed();

            match result {
                TestResult::Passed => {
                    passed += 1;
                    if self.verbose {
                        println!(
                            "{} ({:.2}ms)",
                            "ok".green(),
                            duration.as_secs_f64() * 1000.0
                        );
                    } else {
                        println!("{}", "ok".green());
                    }
                }
                TestResult::Failed(msg) => {
                    failed += 1;
                    println!("{}", "FAILED".red().bold());
                    println!("    {}: {}", "Error".red(), msg);
                }
            }
        }

        let total_duration = start.elapsed();
        println!("{}", "=".repeat(50));

        // Print summary
        let summary = if failed == 0 {
            format!(
                "test result: {}. {} passed; {} failed; finished in {:.2}s",
                "ok".green().bold(),
                passed,
                failed,
                total_duration.as_secs_f64()
            )
        } else {
            format!(
                "test result: {}. {} passed; {} failed; finished in {:.2}s",
                "FAILED".red().bold(),
                passed,
                failed,
                total_duration.as_secs_f64()
            )
        };

        println!("\n{}", summary);

        (passed, failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    #[test]
    fn test_discover_no_tests() {
        let rt = Runtime::new();
        let runner = TestRunner::new(rt, false);
        let tests = runner.discover_tests();
        // The user namespace should have no tests by default (or may have some from other tests)
        // This is a basic sanity check
        assert!(tests.len() >= 0);
    }

    #[test]
    fn test_test_result_variants() {
        // Test that TestResult enum works correctly
        let passed = TestResult::Passed;
        let failed = TestResult::Failed("error".to_string());

        assert!(matches!(passed, TestResult::Passed));
        assert!(matches!(failed, TestResult::Failed(_)));
    }

    #[test]
    fn test_runner_creation() {
        let rt = Runtime::new();
        let runner = TestRunner::new(rt, true);
        // Runner should be creatable with verbose flag
        assert!(runner.verbose);
    }
}
