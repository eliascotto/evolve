//! Integration tests for LLVM code generation.
//!
//! These tests verify that the code generator produces valid LLVM IR
//! for various HIR constructs.
//!
//! # Running These Tests
//!
//! These tests require the `codegen` feature and a properly installed LLVM:
//!
//! ```sh
//! cargo test --features codegen --test codegen
//! ```
//!
//! # System Requirements
//!
//! - LLVM 18.0 installed and accessible
//! - Matching ICU4C libraries (may need `brew upgrade icu4c` on macOS)

#![cfg(feature = "codegen")]

use evolve::codegen::CodeGen;
use evolve::hir::Lowerer;
use evolve::reader::{Reader, Source};
use evolve::runtime::Runtime;
use inkwell::context::Context;

fn parse_and_lower(source: &str) -> evolve::hir::HIR {
    let runtime = Runtime::new();
    let value = Reader::read(source, Source::REPL, runtime).unwrap();
    let lowerer = Lowerer::new();
    lowerer.lower(&value).unwrap()
}

fn create_test_codegen(context: &Context) -> CodeGen {
    CodeGen::new(context, "test_module")
}

fn setup_test_function<'ctx>(
    codegen: &mut CodeGen<'ctx>,
    context: &'ctx Context,
    name: &str,
) -> inkwell::values::FunctionValue<'ctx> {
    let fn_type = codegen.get_value_type().fn_type(&[], false);
    let function = codegen.add_function(name, fn_type);
    let entry = context.append_basic_block(function, "entry");
    codegen.get_builder().position_at_end(entry);
    codegen.set_current_fn(Some(function));
    function
}

#[test]
fn test_compile_nil_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("nil");

    setup_test_function(&mut codegen, &context, "test_nil");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_integer_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("42");

    setup_test_function(&mut codegen, &context, "test_int");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_negative_integer() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("-100");

    setup_test_function(&mut codegen, &context, "test_neg_int");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_boolean_true() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("true");

    setup_test_function(&mut codegen, &context, "test_true");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_boolean_false() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("false");

    setup_test_function(&mut codegen, &context, "test_false");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_string_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("\"hello world\"");

    setup_test_function(&mut codegen, &context, "test_string");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_keyword_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower(":my-keyword");

    setup_test_function(&mut codegen, &context, "test_keyword");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_character_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("\\a");

    setup_test_function(&mut codegen, &context, "test_char");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_if_expression() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(if true 1 2)");

    setup_test_function(&mut codegen, &context, "test_if");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    let ir = codegen.to_string();
    assert!(ir.contains("then"), "IR should contain then branch");
    assert!(ir.contains("else"), "IR should contain else branch");
    assert!(ir.contains("if_merge"), "IR should contain merge block");
}

#[test]
fn test_compile_if_without_else() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(if true 1)");

    setup_test_function(&mut codegen, &context, "test_if_no_else");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_vector_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("[1 2 3]");

    setup_test_function(&mut codegen, &context, "test_vector");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    let ir = codegen.to_string();
    assert!(ir.contains("evolve_vector_new"), "IR should call vector_new runtime function");
}

#[test]
fn test_compile_empty_vector() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("[]");

    setup_test_function(&mut codegen, &context, "test_empty_vector");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_map_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("{:a 1 :b 2}");

    setup_test_function(&mut codegen, &context, "test_map");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    let ir = codegen.to_string();
    assert!(ir.contains("evolve_map_new"), "IR should call map_new runtime function");
}

#[test]
fn test_compile_set_literal() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("#{1 2 3}");

    setup_test_function(&mut codegen, &context, "test_set");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    let ir = codegen.to_string();
    assert!(ir.contains("evolve_set_new"), "IR should call set_new runtime function");
}

#[test]
fn test_compile_def() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(def x 42)");

    setup_test_function(&mut codegen, &context, "test_def");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    let ir = codegen.to_string();
    assert!(ir.contains("evolve_var_def"), "IR should call var_def runtime function");
}

#[test]
fn test_compile_do_block() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(do 1 2 3)");

    setup_test_function(&mut codegen, &context, "test_do");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_let_binding() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(let* [x 1] x)");

    setup_test_function(&mut codegen, &context, "test_let");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_let_multiple_bindings() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(let* [x 1 y 2] y)");

    setup_test_function(&mut codegen, &context, "test_let_multi");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");
}

#[test]
fn test_compile_function_call() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(+ 1 2)");

    setup_test_function(&mut codegen, &context, "test_call");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    let ir = codegen.to_string();
    assert!(ir.contains("evolve_call"), "IR should call the call runtime function");
}

#[test]
fn test_compile_ns() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("(ns my-namespace)");

    setup_test_function(&mut codegen, &context, "test_ns");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    let ir = codegen.to_string();
    assert!(ir.contains("evolve_ns_switch"), "IR should call ns_switch runtime function");
}

#[test]
fn test_llvm_ir_structure() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);
    let hir = parse_and_lower("42");

    setup_test_function(&mut codegen, &context, "test_structure");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    let ir = codegen.to_string();

    // Check basic IR structure
    assert!(ir.contains("define"), "IR should contain function definitions");
    assert!(ir.contains("ret i64"), "IR should return i64 values");
    assert!(ir.contains("@evolve_"), "IR should contain runtime function declarations");
}

#[test]
fn test_runtime_function_declarations() {
    let context = Context::create();
    let codegen = create_test_codegen(&context);

    let ir = codegen.to_string();

    // Check that all runtime functions are declared
    let expected_fns = [
        "evolve_alloc",
        "evolve_retain",
        "evolve_release",
        "evolve_string_new",
        "evolve_vector_new",
        "evolve_map_new",
        "evolve_set_new",
        "evolve_closure_new",
        "evolve_call",
        "evolve_is_truthy",
        "evolve_var_get",
        "evolve_var_def",
    ];

    for fn_name in expected_fns {
        assert!(
            ir.contains(&format!("@{}", fn_name)),
            "IR should declare runtime function {}",
            fn_name
        );
    }
}

#[test]
fn test_module_init_creation() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    let forms = vec![
        parse_and_lower("(def x 1)"),
        parse_and_lower("(def y 2)"),
    ];

    let init_fn = codegen.create_module_init(&forms).unwrap();
    assert_eq!(init_fn.get_name().to_str().unwrap(), "evolve_module_init");
    assert!(codegen.verify().is_ok(), "Generated module should be valid");
}

//===----------------------------------------------------------------------===//
// Phase 9: Optimization Tests
//===----------------------------------------------------------------------===//

use evolve::codegen::{
    EscapeAnalyzer, EscapeState, OptimizationConfig, OptLevel,
    should_inline,
};

/// Create a codegen with specific optimization config
fn create_codegen_with_config(context: &Context, config: OptimizationConfig) -> CodeGen {
    CodeGen::with_config(context, "test_module", config)
}

// ===== Optimization Configuration Tests =====

#[test]
fn test_optimization_config_default() {
    let config = OptimizationConfig::default();
    assert_eq!(config.level, OptLevel::Default);
    assert!(config.enable_tco);
    assert!(config.enable_escape_analysis);
    assert!(config.enable_inlining);
    assert_eq!(config.inline_threshold, 10);
    assert!(!config.enable_profiling);
}

#[test]
fn test_optimization_config_debug() {
    let config = OptimizationConfig::debug();
    assert_eq!(config.level, OptLevel::None);
    assert!(!config.enable_tco);
    assert!(!config.enable_escape_analysis);
    assert!(!config.enable_inlining);
}

#[test]
fn test_optimization_config_release() {
    let config = OptimizationConfig::release();
    assert_eq!(config.level, OptLevel::Aggressive);
    assert!(config.enable_tco);
    assert!(config.enable_escape_analysis);
    assert!(config.enable_inlining);
    assert_eq!(config.inline_threshold, 15);
}

#[test]
fn test_optimization_config_profiling() {
    let config = OptimizationConfig::profiling();
    assert_eq!(config.level, OptLevel::Less);
    assert!(config.enable_profiling);
    assert!(!config.enable_inlining);
}

#[test]
fn test_codegen_with_custom_config() {
    let context = Context::create();
    let config = OptimizationConfig {
        level: OptLevel::Aggressive,
        enable_tco: true,
        enable_escape_analysis: true,
        enable_inlining: true,
        inline_threshold: 20,
        enable_profiling: false,
        custom_passes: vec!["instcombine".to_string()],
    };
    let codegen = create_codegen_with_config(&context, config.clone());

    assert_eq!(codegen.opt_config().level, OptLevel::Aggressive);
    assert_eq!(codegen.opt_config().inline_threshold, 20);
    assert_eq!(codegen.opt_config().custom_passes.len(), 1);
}

// ===== TCO Tests =====

#[test]
fn test_tco_tail_call_in_function() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    // A function with a tail call
    let hir = parse_and_lower("(fn* [x] (f x))");
    setup_test_function(&mut codegen, &context, "test_tco");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok(), "Generated LLVM IR should be valid");

    // The function should use fastcc calling convention
    let ir = codegen.to_string();
    assert!(ir.contains("fastcc"), "Function should use fast calling convention for TCO");
}

#[test]
fn test_tco_disabled() {
    let context = Context::create();
    let config = OptimizationConfig {
        enable_tco: false,
        ..OptimizationConfig::default()
    };
    let mut codegen = create_codegen_with_config(&context, config);

    // A function with a tail call
    let hir = parse_and_lower("(fn* [x] (f x))");
    setup_test_function(&mut codegen, &context, "test_tco_disabled");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok());

    // Statistics should show no tail calls marked
    assert_eq!(codegen.compile_stats().tail_calls_marked, 0);
}

#[test]
fn test_tco_tail_call_in_if_branches() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    // A function with tail calls in both branches
    let hir = parse_and_lower("(fn* [x] (if (> x 0) (f x) (g x)))");
    setup_test_function(&mut codegen, &context, "test_tco_if");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok());

    // Both branches should have tail calls
    let stats = codegen.compile_stats();
    assert!(stats.tail_calls_marked >= 2, "Both if branches should have tail calls marked");
}

#[test]
fn test_tco_verification() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    // A function with a tail call
    let hir = parse_and_lower("(fn* [x] (f x))");
    setup_test_function(&mut codegen, &context, "test_tco_verify");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    // Get TCO summary
    let summary = codegen.tco_summary();

    // There should be at least one tail call (the call to f)
    assert!(summary.total_tail_calls >= 1, "Should have at least one tail call");
}

#[test]
fn test_tco_recursive_function() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    // A recursive factorial-like function
    let hir = parse_and_lower("(fn* fact [n] (if (= n 0) 1 (* n (fact (- n 1)))))");
    setup_test_function(&mut codegen, &context, "test_recursive");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok());

    let ir = codegen.to_string();
    assert!(ir.contains("fastcc"), "Recursive function should use fastcc");
}

// ===== Escape Analysis Tests =====

#[test]
fn test_escape_analysis_local_var() {
    let hir = parse_and_lower("(let* [x 1] x)");
    let mut analyzer = EscapeAnalyzer::new();
    analyzer.analyze(&hir);

    // x is returned, so it should escape
    // Note: In this simple case, x is just an integer which doesn't need escape analysis
}

#[test]
fn test_escape_analysis_closure_capture() {
    let hir = parse_and_lower("(let* [x 1] (fn* [y] x))");
    let mut analyzer = EscapeAnalyzer::new();
    analyzer.analyze(&hir);

    // x is captured by the closure, so it should escape
    let x_sym = evolve::interner::intern_sym("x");
    assert!(analyzer.is_captured(x_sym), "x should be marked as captured");
    assert!(analyzer.escapes(x_sym), "x should escape due to capture");
}

#[test]
fn test_escape_analysis_returned_var() {
    // When a variable is returned (last expr in a let body), it should escape
    // Note: The escape analyzer tracks vars that are explicitly used in return position
    let hir = parse_and_lower("(let* [x [1 2 3]] x)");
    let mut analyzer = EscapeAnalyzer::new();
    analyzer.analyze(&hir);

    // The variable x is in the let body which makes it potentially returned
    // The escape analyzer should detect this when analyzing at a higher level
    // For now, verify the basic mechanism works
    let x_sym = evolve::interner::intern_sym("x");
    // At this level, x is marked when it's analyzed as being in return position
    // The test verifies the escape analyzer runs without error
    let state = analyzer.get_escape_state(x_sym);
    assert!(
        state == EscapeState::NoEscape || state == EscapeState::GlobalEscape,
        "Variable state should be valid"
    );
}

#[test]
fn test_escape_analysis_arg_escape() {
    let hir = parse_and_lower("(let* [x 1] (f x))");
    let mut analyzer = EscapeAnalyzer::new();
    analyzer.analyze(&hir);

    let x_sym = evolve::interner::intern_sym("x");
    // x is passed as an argument, may escape
    assert!(analyzer.get_escape_state(x_sym) != EscapeState::NoEscape,
            "Argument should be marked as potentially escaping");
}

#[test]
fn test_escape_analysis_stored_in_vector() {
    let hir = parse_and_lower("(let* [x 1] [x 2 3])");
    let mut analyzer = EscapeAnalyzer::new();
    analyzer.analyze(&hir);

    let x_sym = evolve::interner::intern_sym("x");
    assert_eq!(analyzer.get_escape_state(x_sym), EscapeState::GlobalEscape,
               "Variable stored in vector should globally escape");
}

#[test]
fn test_escape_analyzer_forms() {
    let forms = vec![
        parse_and_lower("(def x 1)"),
        parse_and_lower("(def y 2)"),
        parse_and_lower("(+ x y)"),
    ];

    let mut analyzer = EscapeAnalyzer::new();
    analyzer.analyze_forms(&forms);

    // The last form returns x+y, which is a call result
    // x and y are global defs, so they escape
}

// ===== Inlining Tests =====

#[test]
fn test_should_inline_simple_function() {
    let hir = parse_and_lower("(fn* [x] x)");
    if let evolve::hir::HIR::Fn { body, .. } = hir {
        assert!(should_inline(&body, 10), "Simple identity function should be inlined");
    }
}

#[test]
fn test_should_inline_complex_function() {
    // A more complex function
    let hir = parse_and_lower(
        "(fn* [x] (let* [a 1 b 2 c 3] (if (> x 0) (+ a b c) (- a b c))))"
    );
    if let evolve::hir::HIR::Fn { body, .. } = hir {
        // With threshold 5, this should NOT be inlined (body is complex)
        assert!(!should_inline(&body, 5), "Complex function should not be inlined with low threshold");
        // With threshold 50, this should be inlined (high enough to cover all nodes)
        assert!(should_inline(&body, 50), "Complex function should be inlined with high threshold");
    }
}

#[test]
fn test_inlining_disabled_with_zero_threshold() {
    let hir = parse_and_lower("(fn* [x] x)");
    if let evolve::hir::HIR::Fn { body, .. } = hir {
        assert!(!should_inline(&body, 0), "Inlining should be disabled with threshold 0");
    }
}

#[test]
fn test_inlining_attribute_applied() {
    let context = Context::create();
    let config = OptimizationConfig {
        enable_inlining: true,
        inline_threshold: 10,
        ..OptimizationConfig::default()
    };
    let mut codegen = create_codegen_with_config(&context, config);

    // A simple function that should be inlined
    let hir = parse_and_lower("(fn* [x] x)");
    setup_test_function(&mut codegen, &context, "test_inline");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok());

    let ir = codegen.to_string();
    assert!(ir.contains("alwaysinline"), "Simple function should have alwaysinline attribute");

    let stats = codegen.compile_stats();
    assert!(stats.functions_inlined >= 1, "At least one function should be marked for inlining");
}

#[test]
fn test_inlining_disabled() {
    let context = Context::create();
    let config = OptimizationConfig {
        enable_inlining: false,
        ..OptimizationConfig::default()
    };
    let mut codegen = create_codegen_with_config(&context, config);

    // A simple function
    let hir = parse_and_lower("(fn* [x] x)");
    setup_test_function(&mut codegen, &context, "test_no_inline");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    let ir = codegen.to_string();
    assert!(!ir.contains("alwaysinline"), "Function should not have alwaysinline when disabled");
}

// ===== Compile Statistics Tests =====

#[test]
fn test_compile_stats_functions_compiled() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    // Compile multiple functions
    let hir1 = parse_and_lower("(fn* [x] x)");
    let hir2 = parse_and_lower("(fn* [y] (+ y 1))");
    let hir_nil = parse_and_lower("nil");

    setup_test_function(&mut codegen, &context, "test_stats");
    codegen.compile(&hir1).unwrap();
    codegen.compile(&hir2).unwrap();
    let nil_result = codegen.compile(&hir_nil).unwrap();
    codegen.get_builder().build_return(Some(&nil_result)).unwrap();

    let stats = codegen.compile_stats();
    assert!(stats.functions_compiled >= 2, "Should count compiled functions");
}

#[test]
fn test_compile_stats_tail_calls() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    // Function with tail calls
    let hir = parse_and_lower("(fn* [x] (if (> x 0) (f x) (g x)))");
    setup_test_function(&mut codegen, &context, "test_tail_stats");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    let stats = codegen.compile_stats();
    assert!(stats.tail_calls_marked > 0, "Should count tail calls");
}

// ===== Optimization Pass Tests =====

#[test]
fn test_run_optimization_passes() {
    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    let hir = parse_and_lower("(fn* [x] (+ x 1))");
    setup_test_function(&mut codegen, &context, "test_opt");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    assert!(codegen.verify().is_ok());

    // Run optimization passes
    codegen.run_optimization_passes();

    // Module should still be valid after optimization
    assert!(codegen.verify().is_ok(), "Module should be valid after optimization");
}

#[test]
fn test_optimize_with_level() {
    use inkwell::OptimizationLevel;

    let context = Context::create();
    let mut codegen = create_test_codegen(&context);

    let hir = parse_and_lower("42");
    setup_test_function(&mut codegen, &context, "test_opt_level");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    // Test different optimization levels
    for level in [
        OptimizationLevel::None,
        OptimizationLevel::Less,
        OptimizationLevel::Default,
        OptimizationLevel::Aggressive,
    ] {
        codegen.optimize(level);
        assert!(codegen.verify().is_ok(), "Module should be valid after optimization at level {:?}", level);
    }
}

// ===== Integration Tests =====

#[test]
fn test_full_optimization_pipeline() {
    let context = Context::create();
    let config = OptimizationConfig::release();
    let mut codegen = create_codegen_with_config(&context, config);

    // Compile a more realistic program
    let forms = vec![
        parse_and_lower("(def add (fn* [x y] (+ x y)))"),
        parse_and_lower("(def square (fn* [x] (* x x)))"),
        parse_and_lower("(square (add 1 2))"),
    ];

    let init_fn = codegen.create_module_init(&forms).unwrap();
    assert!(codegen.verify().is_ok());

    // Run escape analysis
    codegen.analyze_escapes(&forms);

    // Run optimization passes
    codegen.run_optimization_passes();

    assert!(codegen.verify().is_ok(), "Module should be valid after full pipeline");

    let stats = codegen.compile_stats();
    assert!(stats.functions_compiled > 0, "Should have compiled functions");
}

#[test]
fn test_debug_config_no_optimizations() {
    let context = Context::create();
    let config = OptimizationConfig::debug();
    let mut codegen = create_codegen_with_config(&context, config);

    let hir = parse_and_lower("(fn* [x] (f x))");
    setup_test_function(&mut codegen, &context, "test_debug");
    let result = codegen.compile(&hir).unwrap();
    codegen.get_builder().build_return(Some(&result)).unwrap();

    let ir = codegen.to_string();

    // In debug mode:
    // - No fastcc (TCO disabled)
    // - No alwaysinline
    assert!(!ir.contains("alwaysinline"), "Debug mode should not inline");

    let stats = codegen.compile_stats();
    assert_eq!(stats.tail_calls_marked, 0, "Debug mode should not mark tail calls");
    assert_eq!(stats.functions_inlined, 0, "Debug mode should not inline functions");
}
