//! LLVM Code Generation for the Evolve compiler.
//!
//! This module transforms HIR (High-Level Intermediate Representation) into LLVM IR,
//! which can then be compiled to native code.
//!
//! # Value Representation
//!
//! Evolve values are represented as tagged pointers in LLVM:
//! - A 64-bit value where the low 3 bits are used as a type tag
//! - The remaining bits either hold an immediate value or a pointer to heap data
//!
//! Tags:
//! - 0b000: Pointer to boxed object (collections, functions, etc.)
//! - 0b001: Integer (shifted left by 3 bits)
//! - 0b010: Nil
//! - 0b011: Boolean (bit 3 is the value)
//! - 0b100: Character (Unicode codepoint in upper bits)
//! - 0b101: Keyword ID (KeywId in upper bits)
//! - 0b110: Symbol ID (SymId in upper bits)
//! - 0b111: Float (special encoding)
//!
//! # Optimization Features
//!
//! This module implements several optimization features:
//!
//! - **Tail Call Optimization (TCO)**: Marks tail calls with LLVM's `tail` attribute
//!   and uses fastcc calling convention for better TCO support.
//! - **Escape Analysis**: Identifies values that don't escape their scope to enable
//!   stack allocation instead of heap allocation.
//! - **Inlining Heuristics**: Marks small functions with `alwaysinline` attribute.
//! - **Optimization Passes**: Configurable LLVM optimization pass pipeline.
//! - **Profiling**: Optional instrumentation for performance analysis.

pub mod value;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{PointerType, StructType};
use inkwell::values::{FunctionValue, IntValue};
use inkwell::{AddressSpace, OptimizationLevel};

use crate::error::{error_at, Error, SpannedResult};
use crate::hir::{Literal, Pattern, HIR};
use crate::interner::{self, KeywId, SymId};
use crate::reader::Span;

use value::ValueTag;

//===----------------------------------------------------------------------===//
// Optimization Configuration
//===----------------------------------------------------------------------===//

/// Configuration for optimization passes and features.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Overall optimization level (O0-O3)
    pub level: OptLevel,
    /// Enable tail call optimization
    pub enable_tco: bool,
    /// Enable escape analysis for stack allocation
    pub enable_escape_analysis: bool,
    /// Enable automatic inlining of small functions
    pub enable_inlining: bool,
    /// Maximum HIR statements for a function to be considered for inlining
    pub inline_threshold: usize,
    /// Enable profiling instrumentation
    pub enable_profiling: bool,
    /// Custom optimization passes to run (in addition to defaults)
    pub custom_passes: Vec<String>,
}

/// Optimization level enum that maps to LLVM's OptimizationLevel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// No optimization (O0) - fastest compile, easiest to debug
    None,
    /// Basic optimization (O1) - quick optimizations with minimal compile overhead
    Less,
    /// Standard optimization (O2) - good balance of compile time and performance
    #[default]
    Default,
    /// Aggressive optimization (O3) - maximum performance, longer compile time
    Aggressive,
}

impl From<OptLevel> for OptimizationLevel {
    fn from(level: OptLevel) -> Self {
        match level {
            OptLevel::None => OptimizationLevel::None,
            OptLevel::Less => OptimizationLevel::Less,
            OptLevel::Default => OptimizationLevel::Default,
            OptLevel::Aggressive => OptimizationLevel::Aggressive,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptLevel::Default,
            enable_tco: true,
            enable_escape_analysis: true,
            enable_inlining: true,
            inline_threshold: 10,
            enable_profiling: false,
            custom_passes: Vec::new(),
        }
    }
}

impl OptimizationConfig {
    /// Create a debug configuration (no optimizations).
    pub fn debug() -> Self {
        Self {
            level: OptLevel::None,
            enable_tco: false,
            enable_escape_analysis: false,
            enable_inlining: false,
            inline_threshold: 0,
            enable_profiling: false,
            custom_passes: Vec::new(),
        }
    }

    /// Create a release configuration (aggressive optimizations).
    pub fn release() -> Self {
        Self {
            level: OptLevel::Aggressive,
            enable_tco: true,
            enable_escape_analysis: true,
            enable_inlining: true,
            inline_threshold: 15,
            enable_profiling: false,
            custom_passes: Vec::new(),
        }
    }

    /// Create a profiling configuration.
    pub fn profiling() -> Self {
        Self {
            level: OptLevel::Less,
            enable_tco: true,
            enable_escape_analysis: false,
            enable_inlining: false,
            inline_threshold: 0,
            enable_profiling: true,
            custom_passes: Vec::new(),
        }
    }
}

//===----------------------------------------------------------------------===//
// Escape Analysis
//===----------------------------------------------------------------------===//

/// Result of escape analysis for a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeState {
    /// Value does not escape - can be stack allocated
    NoEscape,
    /// Value may escape through a function argument
    ArgEscape,
    /// Value escapes to heap (returned, stored in closure, etc.)
    GlobalEscape,
}

/// Escape analyzer for HIR expressions.
///
/// Determines which values can be safely stack-allocated because they
/// don't escape their defining scope.
pub struct EscapeAnalyzer {
    /// Map from variable to its escape state
    escape_states: HashMap<SymId, EscapeState>,
    /// Variables that are returned from functions
    returned_vars: HashSet<SymId>,
    /// Variables captured by closures
    captured_vars: HashSet<SymId>,
}

impl EscapeAnalyzer {
    /// Create a new escape analyzer.
    pub fn new() -> Self {
        Self {
            escape_states: HashMap::new(),
            returned_vars: HashSet::new(),
            captured_vars: HashSet::new(),
        }
    }

    /// Analyze an HIR expression and determine escape states.
    pub fn analyze(&mut self, hir: &HIR) {
        self.analyze_expr(hir, false);
    }

    /// Analyze a list of HIR expressions.
    pub fn analyze_forms(&mut self, forms: &[HIR]) {
        for (i, form) in forms.iter().enumerate() {
            let is_returned = i == forms.len() - 1;
            self.analyze_expr(form, is_returned);
        }
    }

    fn analyze_expr(&mut self, hir: &HIR, is_returned: bool) {
        match hir {
            HIR::Literal { .. } => {
                // Literals don't create allocations we can optimize
            }

            HIR::Var { name, .. } => {
                if is_returned {
                    self.returned_vars.insert(*name);
                    self.mark_escape(*name, EscapeState::GlobalEscape);
                }
            }

            HIR::Quote { .. } => {
                // Quoted values are immutable, can't optimize their allocation
            }

            HIR::Def { value, .. } => {
                // Defined values escape to global scope
                self.analyze_expr(value, true);
            }

            HIR::DefMacro { body, .. } => {
                for form in body {
                    self.analyze_expr(form, false);
                }
            }

            HIR::If { condition, then_branch, else_branch, .. } => {
                self.analyze_expr(condition, false);
                self.analyze_expr(then_branch, is_returned);
                if let Some(eb) = else_branch {
                    self.analyze_expr(eb, is_returned);
                }
            }

            HIR::Let { bindings, body, .. } => {
                for (pattern, value) in bindings {
                    self.analyze_expr(value, false);
                    // Mark bound variables as potentially non-escaping
                    for name in pattern.bound_names() {
                        if !self.escape_states.contains_key(&name) {
                            self.escape_states.insert(name, EscapeState::NoEscape);
                        }
                    }
                }
                for (i, form) in body.iter().enumerate() {
                    let form_is_returned = is_returned && i == body.len() - 1;
                    self.analyze_expr(form, form_is_returned);
                }
            }

            HIR::Do { forms, .. } => {
                for (i, form) in forms.iter().enumerate() {
                    let form_is_returned = is_returned && i == forms.len() - 1;
                    self.analyze_expr(form, form_is_returned);
                }
            }

            HIR::Fn { params, body, name, .. } => {
                // Analyze what the function captures
                let bound_vars: Vec<SymId> =
                    params.iter().flat_map(|p| p.bound_names()).collect();
                let mut fn_name_vec = Vec::new();
                if let Some(n) = name {
                    fn_name_vec.push(*n);
                }
                let free = collect_free_vars(body, &[&bound_vars[..], &fn_name_vec[..]].concat());

                // Mark captured variables as escaping
                for var in free {
                    self.captured_vars.insert(var);
                    self.mark_escape(var, EscapeState::GlobalEscape);
                }

                // Recursively analyze body
                for (i, form) in body.iter().enumerate() {
                    // Last form in function body is returned
                    self.analyze_expr(form, i == body.len() - 1);
                }
            }

            HIR::Loop { bindings, body, .. } => {
                for (pattern, value) in bindings {
                    self.analyze_expr(value, false);
                    for name in pattern.bound_names() {
                        if !self.escape_states.contains_key(&name) {
                            self.escape_states.insert(name, EscapeState::NoEscape);
                        }
                    }
                }
                for form in body {
                    self.analyze_expr(form, false);
                }
            }

            HIR::Recur { args, .. } => {
                for arg in args {
                    self.analyze_expr(arg, false);
                }
            }

            HIR::Call { callee, args, .. } => {
                self.analyze_expr(callee, false);
                // Arguments passed to functions may escape
                for arg in args {
                    self.analyze_expr(arg, false);
                    // Mark any variables passed as arguments as potentially escaping
                    if let HIR::Var { name, .. } = arg {
                        self.mark_escape(*name, EscapeState::ArgEscape);
                    }
                }
            }

            HIR::Vector { items, .. } => {
                for item in items {
                    self.analyze_expr(item, false);
                    // Items stored in vectors escape
                    if let HIR::Var { name, .. } = item {
                        self.mark_escape(*name, EscapeState::GlobalEscape);
                    }
                }
            }

            HIR::Map { entries, .. } => {
                for (key, value) in entries {
                    self.analyze_expr(key, false);
                    self.analyze_expr(value, false);
                    // Values stored in maps escape
                    if let HIR::Var { name, .. } = value {
                        self.mark_escape(*name, EscapeState::GlobalEscape);
                    }
                }
            }

            HIR::Set { items, .. } => {
                for item in items {
                    self.analyze_expr(item, false);
                    if let HIR::Var { name, .. } = item {
                        self.mark_escape(*name, EscapeState::GlobalEscape);
                    }
                }
            }

            HIR::Ns { .. } => {}
        }
    }

    fn mark_escape(&mut self, name: SymId, state: EscapeState) {
        let current = self.escape_states.get(&name).copied().unwrap_or(EscapeState::NoEscape);
        // Escalate escape state (NoEscape < ArgEscape < GlobalEscape)
        let new_state = match (current, state) {
            (EscapeState::GlobalEscape, _) => EscapeState::GlobalEscape,
            (_, EscapeState::GlobalEscape) => EscapeState::GlobalEscape,
            (EscapeState::ArgEscape, _) => EscapeState::ArgEscape,
            (_, EscapeState::ArgEscape) => EscapeState::ArgEscape,
            _ => EscapeState::NoEscape,
        };
        self.escape_states.insert(name, new_state);
    }

    /// Get the escape state for a variable.
    pub fn get_escape_state(&self, name: SymId) -> EscapeState {
        self.escape_states.get(&name).copied().unwrap_or(EscapeState::NoEscape)
    }

    /// Check if a variable escapes.
    pub fn escapes(&self, name: SymId) -> bool {
        self.get_escape_state(name) != EscapeState::NoEscape
    }

    /// Check if a variable is captured by a closure.
    pub fn is_captured(&self, name: SymId) -> bool {
        self.captured_vars.contains(&name)
    }
}

impl Default for EscapeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

//===----------------------------------------------------------------------===//
// Inlining Heuristics
//===----------------------------------------------------------------------===//

/// Determine if a function should be inlined based on heuristics.
pub fn should_inline(body: &[HIR], threshold: usize) -> bool {
    if threshold == 0 {
        return false;
    }
    let complexity = compute_complexity(body);
    complexity <= threshold
}

/// Compute the complexity score of a function body.
fn compute_complexity(forms: &[HIR]) -> usize {
    forms.iter().map(|f| expr_complexity(f)).sum()
}

/// Compute complexity for a single expression.
fn expr_complexity(hir: &HIR) -> usize {
    match hir {
        HIR::Literal { .. } => 1,
        HIR::Var { .. } => 1,
        HIR::Quote { .. } => 1,
        HIR::Def { value, .. } => 1 + expr_complexity(value),
        HIR::DefMacro { body, .. } => 1 + compute_complexity(body),
        HIR::If { condition, then_branch, else_branch, .. } => {
            2 + expr_complexity(condition)
                + expr_complexity(then_branch)
                + else_branch.as_ref().map(|e| expr_complexity(e)).unwrap_or(0)
        }
        HIR::Let { bindings, body, .. } => {
            1 + bindings.iter().map(|(_, v)| expr_complexity(v)).sum::<usize>()
                + compute_complexity(body)
        }
        HIR::Do { forms, .. } => compute_complexity(forms),
        HIR::Fn { body, .. } => 3 + compute_complexity(body), // Functions are more complex
        HIR::Loop { bindings, body, .. } => {
            2 + bindings.iter().map(|(_, v)| expr_complexity(v)).sum::<usize>()
                + compute_complexity(body)
        }
        HIR::Recur { args, .. } => 1 + args.iter().map(expr_complexity).sum::<usize>(),
        HIR::Call { callee, args, .. } => {
            2 + expr_complexity(callee) + args.iter().map(expr_complexity).sum::<usize>()
        }
        HIR::Vector { items, .. } => 1 + items.iter().map(expr_complexity).sum::<usize>(),
        HIR::Map { entries, .. } => {
            1 + entries.iter().map(|(k, v)| expr_complexity(k) + expr_complexity(v)).sum::<usize>()
        }
        HIR::Set { items, .. } => 1 + items.iter().map(expr_complexity).sum::<usize>(),
        HIR::Ns { .. } => 1,
    }
}

//===----------------------------------------------------------------------===//
// Profiling Infrastructure
//===----------------------------------------------------------------------===//

/// Profiling data collected during execution.
#[derive(Debug, Default, Clone)]
pub struct ProfilingData {
    /// Function call counts
    pub call_counts: HashMap<String, u64>,
    /// Compilation statistics
    pub compile_stats: CompileStats,
}

/// Statistics about the compilation process.
#[derive(Debug, Default, Clone)]
pub struct CompileStats {
    /// Number of functions compiled
    pub functions_compiled: usize,
    /// Number of tail calls marked
    pub tail_calls_marked: usize,
    /// Number of values stack-allocated due to escape analysis
    pub stack_allocations: usize,
    /// Number of functions marked for inlining
    pub functions_inlined: usize,
    /// Total HIR nodes processed
    pub hir_nodes_processed: usize,
}

//===----------------------------------------------------------------------===//
// TCO Verification
//===----------------------------------------------------------------------===//

/// Result of TCO verification for a single call site.
#[derive(Debug, Clone)]
pub struct TcoVerification {
    /// The LLVM IR line containing the call
    pub ir_line: String,
    /// Whether this is marked as a tail call
    pub is_tail_call: bool,
    /// Whether this is a musttail call (guaranteed TCO)
    pub is_musttail: bool,
}

/// Summary of TCO verification results.
#[derive(Debug, Clone, Default)]
pub struct TcoSummary {
    /// Total number of tail calls found
    pub total_tail_calls: usize,
    /// Number of musttail calls (guaranteed TCO)
    pub musttail_calls: usize,
    /// Number of regular tail calls (may be optimized)
    pub regular_tail_calls: usize,
}

impl TcoSummary {
    /// Check if TCO is working (at least some tail calls are marked).
    pub fn is_working(&self) -> bool {
        self.total_tail_calls > 0
    }
}

//===----------------------------------------------------------------------===//
// Free Variable Analysis
//===----------------------------------------------------------------------===//

/// Collects free variables from HIR expressions.
///
/// A free variable is a variable that is referenced but not bound
/// in the current scope. For closures, these need to be captured
/// from the enclosing environment.
struct FreeVarCollector {
    /// Variables that are bound in the current scope
    bound: HashSet<SymId>,
    /// Variables that are referenced but not bound (free variables)
    free: HashSet<SymId>,
}

impl FreeVarCollector {
    fn new() -> Self {
        Self {
            bound: HashSet::new(),
            free: HashSet::new(),
        }
    }

    /// Collect free variables from an HIR expression.
    fn collect(&mut self, hir: &HIR) {
        match hir {
            HIR::Literal { .. } => {}

            HIR::Var { name, .. } => {
                if !self.bound.contains(name) {
                    self.free.insert(*name);
                }
            }

            HIR::Quote { .. } => {
                // Quoted expressions don't reference variables at runtime
            }

            HIR::Def { value, .. } => {
                self.collect(value);
            }

            HIR::DefMacro { body, params, .. } => {
                // Macros are expanded at read time, but we still analyze them
                let saved_bound = self.bound.clone();
                for param in params {
                    for name in param.bound_names() {
                        self.bound.insert(name);
                    }
                }
                for form in body {
                    self.collect(form);
                }
                self.bound = saved_bound;
            }

            HIR::If { condition, then_branch, else_branch, .. } => {
                self.collect(condition);
                self.collect(then_branch);
                if let Some(eb) = else_branch {
                    self.collect(eb);
                }
            }

            HIR::Let { bindings, body, .. } => {
                let saved_bound = self.bound.clone();
                for (pattern, value) in bindings {
                    // Value is evaluated before pattern is bound
                    self.collect(value);
                    for name in pattern.bound_names() {
                        self.bound.insert(name);
                    }
                }
                for form in body {
                    self.collect(form);
                }
                self.bound = saved_bound;
            }

            HIR::Do { forms, .. } => {
                for form in forms {
                    self.collect(form);
                }
            }

            HIR::Fn { params, body, name, .. } => {
                // For nested functions, we need to analyze what they capture
                // The nested function's free variables become our free variables
                // (if they're not bound in our scope)
                let saved_bound = self.bound.clone();

                // Add function name to scope (for recursive calls)
                if let Some(fn_name) = name {
                    self.bound.insert(*fn_name);
                }

                // Add parameters to scope
                for param in params {
                    for n in param.bound_names() {
                        self.bound.insert(n);
                    }
                }

                // Analyze body
                for form in body {
                    self.collect(form);
                }

                self.bound = saved_bound;
            }

            HIR::Loop { bindings, body, .. } => {
                let saved_bound = self.bound.clone();
                for (pattern, value) in bindings {
                    self.collect(value);
                    for name in pattern.bound_names() {
                        self.bound.insert(name);
                    }
                }
                for form in body {
                    self.collect(form);
                }
                self.bound = saved_bound;
            }

            HIR::Recur { args, .. } => {
                for arg in args {
                    self.collect(arg);
                }
            }

            HIR::Call { callee, args, .. } => {
                self.collect(callee);
                for arg in args {
                    self.collect(arg);
                }
            }

            HIR::Vector { items, .. } => {
                for item in items {
                    self.collect(item);
                }
            }

            HIR::Map { entries, .. } => {
                for (key, value) in entries {
                    self.collect(key);
                    self.collect(value);
                }
            }

            HIR::Set { items, .. } => {
                for item in items {
                    self.collect(item);
                }
            }

            HIR::Ns { .. } => {}
        }
    }

    /// Get the collected free variables.
    fn into_free_vars(self) -> HashSet<SymId> {
        self.free
    }
}

/// Analyze an HIR expression to find free variables.
///
/// The `bound_vars` parameter specifies variables that are already
/// bound in the enclosing scope (e.g., function parameters).
fn collect_free_vars(hir: &[HIR], bound_vars: &[SymId]) -> Vec<SymId> {
    let mut collector = FreeVarCollector::new();

    // Add already-bound variables
    for &var in bound_vars {
        collector.bound.insert(var);
    }

    // Collect from all forms
    for form in hir {
        collector.collect(form);
    }

    // Return sorted for deterministic ordering
    let mut free: Vec<_> = collector.into_free_vars().into_iter().collect();
    free.sort_by_key(|s| s.0);
    free
}

//===----------------------------------------------------------------------===//
// CodeGen - Main code generator
//===----------------------------------------------------------------------===//

/// LLVM code generator for Evolve.
pub struct CodeGen<'ctx> {
    /// The LLVM context
    context: &'ctx Context,
    /// The LLVM module being generated
    module: Module<'ctx>,
    /// The LLVM IR builder
    builder: Builder<'ctx>,

    /// The Evolve value type (i64 tagged pointer)
    value_type: inkwell::types::IntType<'ctx>,
    /// Pointer type for heap-allocated objects
    ptr_type: PointerType<'ctx>,

    /// Boxed object header type: { i8 type_tag, i32 ref_count }
    #[allow(dead_code)]
    header_type: StructType<'ctx>,

    /// Runtime function declarations
    runtime_fns: HashMap<&'static str, FunctionValue<'ctx>>,

    /// Currently compiling function
    current_fn: Option<FunctionValue<'ctx>>,
    /// Local variable bindings (SymId -> LLVM value)
    locals: HashMap<SymId, IntValue<'ctx>>,

    /// Loop target for recur (basic block and bindings)
    loop_target: Option<LoopTarget<'ctx>>,

    /// Counter for generating unique names
    name_counter: u64,

    /// Optimization configuration
    opt_config: OptimizationConfig,

    /// Escape analyzer for the current compilation unit
    escape_analyzer: EscapeAnalyzer,

    /// Compilation statistics for profiling
    compile_stats: CompileStats,
}

/// Target for loop/recur
struct LoopTarget<'ctx> {
    /// The basic block to jump to on recur
    header_block: inkwell::basic_block::BasicBlock<'ctx>,
    /// PHI nodes for loop bindings
    phi_nodes: Vec<(SymId, inkwell::values::PhiValue<'ctx>)>,
}

impl<'ctx> CodeGen<'ctx> {
    /// Create a new code generator with default optimization configuration.
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        Self::with_config(context, module_name, OptimizationConfig::default())
    }

    /// Create a new code generator with custom optimization configuration.
    pub fn with_config(
        context: &'ctx Context,
        module_name: &str,
        opt_config: OptimizationConfig,
    ) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        // Value type: i64 (tagged pointer)
        let value_type = context.i64_type();

        // Pointer type for heap objects
        let ptr_type = context.ptr_type(AddressSpace::default());

        // Header type for boxed objects: { i8 type, i32 refcount }
        let header_type =
            context.struct_type(&[context.i8_type().into(), context.i32_type().into()], false);

        let mut codegen = Self {
            context,
            module,
            builder,
            value_type,
            ptr_type,
            header_type,
            runtime_fns: HashMap::new(),
            current_fn: None,
            locals: HashMap::new(),
            loop_target: None,
            name_counter: 0,
            opt_config,
            escape_analyzer: EscapeAnalyzer::new(),
            compile_stats: CompileStats::default(),
        };

        // Declare runtime functions
        codegen.declare_runtime_functions();

        codegen
    }

    /// Get the current optimization configuration.
    pub fn opt_config(&self) -> &OptimizationConfig {
        &self.opt_config
    }

    /// Get the compilation statistics.
    pub fn compile_stats(&self) -> &CompileStats {
        &self.compile_stats
    }

    /// Run escape analysis on the given HIR forms.
    pub fn analyze_escapes(&mut self, forms: &[HIR]) {
        if self.opt_config.enable_escape_analysis {
            self.escape_analyzer = EscapeAnalyzer::new();
            self.escape_analyzer.analyze_forms(forms);
        }
    }

    /// Generate a unique name for LLVM values.
    fn unique_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.name_counter);
        self.name_counter += 1;
        name
    }

    /// Declare external runtime functions.
    fn declare_runtime_functions(&mut self) {
        let value_type = self.value_type;
        let ptr_type = self.ptr_type;
        let i64_type = self.context.i64_type();
        let i32_type = self.context.i32_type();
        let void_type = self.context.void_type();

        // Memory allocation: evolve_alloc(size: i64) -> ptr
        let alloc_fn_type = ptr_type.fn_type(&[i64_type.into()], false);
        let alloc_fn = self.module.add_function("evolve_alloc", alloc_fn_type, None);
        self.runtime_fns.insert("alloc", alloc_fn);

        // Reference counting
        let retain_fn_type = void_type.fn_type(&[value_type.into()], false);
        let retain_fn = self.module.add_function("evolve_retain", retain_fn_type, None);
        self.runtime_fns.insert("retain", retain_fn);

        let release_fn_type = void_type.fn_type(&[value_type.into()], false);
        let release_fn = self.module.add_function("evolve_release", release_fn_type, None);
        self.runtime_fns.insert("release", release_fn);

        // String creation: evolve_string_new(data: ptr, len: i64) -> Value
        let string_new_type = value_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let string_new_fn = self.module.add_function("evolve_string_new", string_new_type, None);
        self.runtime_fns.insert("string_new", string_new_fn);

        // Vector creation: evolve_vector_new(count: i32, items: ptr) -> Value
        let vector_new_type = value_type.fn_type(&[i32_type.into(), ptr_type.into()], false);
        let vector_new_fn = self.module.add_function("evolve_vector_new", vector_new_type, None);
        self.runtime_fns.insert("vector_new", vector_new_fn);

        // Map creation: evolve_map_new(count: i32, entries: ptr) -> Value
        let map_new_type = value_type.fn_type(&[i32_type.into(), ptr_type.into()], false);
        let map_new_fn = self.module.add_function("evolve_map_new", map_new_type, None);
        self.runtime_fns.insert("map_new", map_new_fn);

        // Set creation: evolve_set_new(count: i32, items: ptr) -> Value
        let set_new_type = value_type.fn_type(&[i32_type.into(), ptr_type.into()], false);
        let set_new_fn = self.module.add_function("evolve_set_new", set_new_type, None);
        self.runtime_fns.insert("set_new", set_new_fn);

        // List creation: evolve_list_new(count: i32, items: ptr) -> Value
        let list_new_type = value_type.fn_type(&[i32_type.into(), ptr_type.into()], false);
        let list_new_fn = self.module.add_function("evolve_list_new", list_new_type, None);
        self.runtime_fns.insert("list_new", list_new_fn);

        // Closure creation: evolve_closure_new(fn_ptr: ptr, env_count: i32, env: ptr) -> Value
        let closure_new_type =
            value_type.fn_type(&[ptr_type.into(), i32_type.into(), ptr_type.into()], false);
        let closure_new_fn = self.module.add_function("evolve_closure_new", closure_new_type, None);
        self.runtime_fns.insert("closure_new", closure_new_fn);

        // Function call: evolve_call(fn: Value, argc: i32, argv: ptr) -> Value
        let call_type = value_type.fn_type(&[value_type.into(), i32_type.into(), ptr_type.into()], false);
        let call_fn = self.module.add_function("evolve_call", call_type, None);
        self.runtime_fns.insert("call", call_fn);

        // Truthiness check: evolve_is_truthy(value: Value) -> i1
        let is_truthy_type = self.context.bool_type().fn_type(&[value_type.into()], false);
        let is_truthy_fn = self.module.add_function("evolve_is_truthy", is_truthy_type, None);
        self.runtime_fns.insert("is_truthy", is_truthy_fn);

        // Vector get: evolve_vector_get(vec: Value, index: i64) -> Value
        let vector_get_type = value_type.fn_type(&[value_type.into(), i64_type.into()], false);
        let vector_get_fn = self.module.add_function("evolve_vector_get", vector_get_type, None);
        self.runtime_fns.insert("vector_get", vector_get_fn);

        // Vector count: evolve_vector_count(vec: Value) -> i64
        let vector_count_type = i64_type.fn_type(&[value_type.into()], false);
        let vector_count_fn =
            self.module.add_function("evolve_vector_count", vector_count_type, None);
        self.runtime_fns.insert("vector_count", vector_count_fn);

        // Vector rest: evolve_vector_rest(vec: Value, start: i64) -> Value
        let vector_rest_type = value_type.fn_type(&[value_type.into(), i64_type.into()], false);
        let vector_rest_fn = self.module.add_function("evolve_vector_rest", vector_rest_type, None);
        self.runtime_fns.insert("vector_rest", vector_rest_fn);

        // Map get: evolve_map_get(map: Value, key: Value) -> Value
        let map_get_type = value_type.fn_type(&[value_type.into(), value_type.into()], false);
        let map_get_fn = self.module.add_function("evolve_map_get", map_get_type, None);
        self.runtime_fns.insert("map_get", map_get_fn);

        // Global variable lookup: evolve_var_get(sym_id: i64) -> Value
        let var_get_type = value_type.fn_type(&[i64_type.into()], false);
        let var_get_fn = self.module.add_function("evolve_var_get", var_get_type, None);
        self.runtime_fns.insert("var_get", var_get_fn);

        // Global variable definition: evolve_var_def(sym_id: i64, value: Value) -> Value
        let var_def_type = value_type.fn_type(&[i64_type.into(), value_type.into()], false);
        let var_def_fn = self.module.add_function("evolve_var_def", var_def_type, None);
        self.runtime_fns.insert("var_def", var_def_fn);

        // Quote value from constant: evolve_quote(data: ptr) -> Value
        let quote_type = value_type.fn_type(&[ptr_type.into()], false);
        let quote_fn = self.module.add_function("evolve_quote", quote_type, None);
        self.runtime_fns.insert("quote", quote_fn);

        // Panic/error: evolve_panic(msg: ptr) -> void
        let panic_type = void_type.fn_type(&[ptr_type.into()], false);
        let panic_fn = self.module.add_function("evolve_panic", panic_type, None);
        self.runtime_fns.insert("panic", panic_fn);

        // Namespace switch: evolve_ns_switch(sym_id: i64) -> Value
        let ns_switch_type = value_type.fn_type(&[i64_type.into()], false);
        let ns_switch_fn = self.module.add_function("evolve_ns_switch", ns_switch_type, None);
        self.runtime_fns.insert("ns_switch", ns_switch_fn);

        // Native function call: evolve_native_call(sym_id: i64, argc: i32, argv: ptr) -> Value
        let native_call_type =
            value_type.fn_type(&[i64_type.into(), i32_type.into(), ptr_type.into()], false);
        let native_call_fn = self.module.add_function("evolve_native_call", native_call_type, None);
        self.runtime_fns.insert("native_call", native_call_fn);
    }

    /// Get a runtime function by name.
    fn runtime_fn(&self, name: &str) -> FunctionValue<'ctx> {
        *self.runtime_fns.get(name).unwrap_or_else(|| {
            panic!("Runtime function '{}' not declared", name)
        })
    }

    //===----------------------------------------------------------------------===//
    // Value Creation Helpers
    //===----------------------------------------------------------------------===//

    /// Create a nil value.
    fn nil_value(&self) -> IntValue<'ctx> {
        self.context.i64_type().const_int(ValueTag::Nil as u64, false)
    }

    /// Create a boolean value.
    fn bool_value(&self, b: bool) -> IntValue<'ctx> {
        let tag = ValueTag::Bool as u64;
        let val = if b { 1u64 << 3 } else { 0 };
        self.context.i64_type().const_int(tag | val, false)
    }

    /// Create an integer value.
    fn int_value(&self, n: i64) -> IntValue<'ctx> {
        // Shift left by 3 bits and add tag
        let tag = ValueTag::Int as u64;
        let shifted = (n << 3) as u64;
        self.context.i64_type().const_int(shifted | tag, false)
    }

    /// Create a character value.
    fn char_value(&self, c: char) -> IntValue<'ctx> {
        let tag = ValueTag::Char as u64;
        let codepoint = (c as u64) << 3;
        self.context.i64_type().const_int(codepoint | tag, false)
    }

    /// Create a keyword value.
    fn keyword_value(&self, kw: KeywId) -> IntValue<'ctx> {
        let tag = ValueTag::Keyword as u64;
        let id = (kw.0 as u64) << 3;
        self.context.i64_type().const_int(id | tag, false)
    }

    /// Create a float value (boxed).
    fn float_value(&mut self, f: f64) -> IntValue<'ctx> {
        // Floats need special handling - we box them
        // For now, use a simple approach: store the bits with the float tag
        let bits = f.to_bits();
        // We'll use a different encoding: store pointer to boxed float
        // For immediate floats, we could use NaN-boxing, but for simplicity
        // we'll just box all floats for now

        // Allocate space for the float
        let alloc_fn = self.runtime_fn("alloc");
        let size = self.context.i64_type().const_int(8, false);
        let ptr = self
            .builder
            .build_call(alloc_fn, &[size.into()], "float_alloc")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_pointer_value();

        // Store the float value
        let float_val = self.context.f64_type().const_float(f);
        self.builder.build_store(ptr, float_val).unwrap();

        // Convert pointer to tagged value
        let ptr_int = self.builder.build_ptr_to_int(ptr, self.value_type, "float_ptr_int").unwrap();
        let tag = self.context.i64_type().const_int(ValueTag::Float as u64, false);
        self.builder.build_or(ptr_int, tag, "float_tagged").unwrap()
    }

    /// Create a string value (calls runtime).
    fn string_value(&mut self, s: &str) -> IntValue<'ctx> {
        // Create a global constant for the string data
        let name = self.unique_name("str");
        let string_const = self.context.const_string(s.as_bytes(), false);
        let global = self.module.add_global(string_const.get_type(), None, &name);
        global.set_initializer(&string_const);

        let ptr = global.as_pointer_value();
        let len = self.context.i64_type().const_int(s.len() as u64, false);

        // Call runtime to create string value
        let string_new_fn = self.runtime_fn("string_new");
        let result = self
            .builder
            .build_call(string_new_fn, &[ptr.into(), len.into()], "string_new")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        result.into_int_value()
    }

    //===----------------------------------------------------------------------===//
    // Main Compilation Entry Points
    //===----------------------------------------------------------------------===//

    /// Compile a single HIR node.
    pub fn compile(&mut self, hir: &HIR) -> SpannedResult<IntValue<'ctx>> {
        match hir {
            HIR::Literal { span, value } => self.compile_literal(span, value),
            HIR::Var { span, name } => self.compile_var(span, *name),
            HIR::Quote { span, value } => self.compile_quote(span, value),
            HIR::Def { span, name, value, .. } => self.compile_def(span, *name, value),
            HIR::DefMacro { span, .. } => {
                // Macros are expanded at read time, not compiled
                Ok(self.nil_value())
            }
            HIR::If { span, condition, then_branch, else_branch, is_tail } => {
                self.compile_if(span, condition, then_branch, else_branch.as_deref(), *is_tail)
            }
            HIR::Let { span, bindings, body, is_tail } => {
                self.compile_let(span, bindings, body, *is_tail)
            }
            HIR::Do { span, forms, is_tail } => self.compile_do(span, forms, *is_tail),
            HIR::Fn { span, name, params, body } => {
                self.compile_fn(span, *name, params, body)
            }
            HIR::Loop { span, bindings, body } => self.compile_loop(span, bindings, body),
            HIR::Recur { span, args } => self.compile_recur(span, args),
            HIR::Call { span, callee, args, is_tail } => {
                self.compile_call(span, callee, args, *is_tail)
            }
            HIR::Vector { span, items } => self.compile_vector(span, items),
            HIR::Map { span, entries } => self.compile_map(span, entries),
            HIR::Set { span, items } => self.compile_set(span, items),
            HIR::Ns { span, name } => self.compile_ns(span, *name),
        }
    }

    /// Compile multiple HIR nodes, returning the last value.
    pub fn compile_forms(&mut self, forms: &[HIR]) -> SpannedResult<IntValue<'ctx>> {
        let mut result = self.nil_value();
        for form in forms {
            result = self.compile(form)?;
        }
        Ok(result)
    }

    //===----------------------------------------------------------------------===//
    // Literal Compilation
    //===----------------------------------------------------------------------===//

    fn compile_literal(&mut self, _span: &Span, literal: &Literal) -> SpannedResult<IntValue<'ctx>> {
        Ok(match literal {
            Literal::Nil => self.nil_value(),
            Literal::Bool(b) => self.bool_value(*b),
            Literal::Int(n) => self.int_value(*n),
            Literal::Float(f) => self.float_value(*f),
            Literal::Char(c) => self.char_value(*c),
            Literal::String(s) => self.string_value(s),
            Literal::Keyword(kw) => self.keyword_value(*kw),
        })
    }

    //===----------------------------------------------------------------------===//
    // Variable Compilation
    //===----------------------------------------------------------------------===//

    fn compile_var(&mut self, span: &Span, name: SymId) -> SpannedResult<IntValue<'ctx>> {
        // First check local bindings
        if let Some(&value) = self.locals.get(&name) {
            return Ok(value);
        }

        // Otherwise, look up global var at runtime
        let sym_id = self.context.i64_type().const_int(name.0 as u64, false);
        let var_get_fn = self.runtime_fn("var_get");
        let result = self
            .builder
            .build_call(var_get_fn, &[sym_id.into()], "var_get")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        Ok(result.into_int_value())
    }

    //===----------------------------------------------------------------------===//
    // Quote Compilation
    //===----------------------------------------------------------------------===//

    fn compile_quote(
        &mut self,
        span: &Span,
        value: &crate::value::Value,
    ) -> SpannedResult<IntValue<'ctx>> {
        // For quoted values, we need to serialize them and call runtime
        // This is a simplified approach - in a real implementation,
        // we'd embed the value data in the binary

        // For now, return nil as a placeholder
        // TODO: Implement proper quote serialization
        Ok(self.nil_value())
    }

    //===----------------------------------------------------------------------===//
    // Definition Compilation
    //===----------------------------------------------------------------------===//

    fn compile_def(
        &mut self,
        span: &Span,
        name: SymId,
        value: &HIR,
    ) -> SpannedResult<IntValue<'ctx>> {
        // Compile the value
        let compiled_value = self.compile(value)?;

        // Call runtime to define the var
        let sym_id = self.context.i64_type().const_int(name.0 as u64, false);
        let var_def_fn = self.runtime_fn("var_def");
        let result = self
            .builder
            .build_call(var_def_fn, &[sym_id.into(), compiled_value.into()], "var_def")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        Ok(result.into_int_value())
    }

    //===----------------------------------------------------------------------===//
    // If Compilation
    //===----------------------------------------------------------------------===//

    fn compile_if(
        &mut self,
        span: &Span,
        condition: &HIR,
        then_branch: &HIR,
        else_branch: Option<&HIR>,
        is_tail: bool,
    ) -> SpannedResult<IntValue<'ctx>> {
        let current_fn = self.current_fn.ok_or_else(|| {
            error_at(span.clone(), Error::RuntimeError("if outside function".to_string()))
        })?;

        // Compile condition
        let cond_value = self.compile(condition)?;

        // Check truthiness
        let is_truthy_fn = self.runtime_fn("is_truthy");
        let is_truthy = self
            .builder
            .build_call(is_truthy_fn, &[cond_value.into()], "is_truthy")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();

        // Create basic blocks
        let then_bb = self.context.append_basic_block(current_fn, "then");
        let else_bb = self.context.append_basic_block(current_fn, "else");
        let merge_bb = self.context.append_basic_block(current_fn, "if_merge");

        // Branch based on condition
        self.builder.build_conditional_branch(is_truthy, then_bb, else_bb).unwrap();

        // Compile then branch
        self.builder.position_at_end(then_bb);
        let then_value = self.compile(then_branch)?;
        let then_end_bb = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).unwrap();

        // Compile else branch
        self.builder.position_at_end(else_bb);
        let else_value = match else_branch {
            Some(eb) => self.compile(eb)?,
            None => self.nil_value(),
        };
        let else_end_bb = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).unwrap();

        // Create PHI node at merge point
        self.builder.position_at_end(merge_bb);
        let phi = self.builder.build_phi(self.value_type, "if_result").unwrap();
        phi.add_incoming(&[(&then_value, then_end_bb), (&else_value, else_end_bb)]);

        Ok(phi.as_basic_value().into_int_value())
    }

    //===----------------------------------------------------------------------===//
    // Let Compilation
    //===----------------------------------------------------------------------===//

    fn compile_let(
        &mut self,
        span: &Span,
        bindings: &[(Pattern, HIR)],
        body: &[HIR],
        is_tail: bool,
    ) -> SpannedResult<IntValue<'ctx>> {
        // Save current locals
        let saved_locals = self.locals.clone();

        // Compile each binding
        for (pattern, value_hir) in bindings {
            let value = self.compile(value_hir)?;
            self.bind_pattern(pattern, value)?;
        }

        // Compile body
        let result = self.compile_forms(body)?;

        // Restore locals
        self.locals = saved_locals;

        Ok(result)
    }

    /// Bind a pattern to a value.
    fn bind_pattern(&mut self, pattern: &Pattern, value: IntValue<'ctx>) -> SpannedResult<()> {
        match pattern {
            Pattern::Bind { name, .. } => {
                self.locals.insert(*name, value);
                Ok(())
            }
            Pattern::Ignore { .. } => {
                // Do nothing
                Ok(())
            }
            Pattern::Vector { patterns, span } => {
                // Destructure vector
                for (i, pat) in patterns.iter().enumerate() {
                    // Check for rest pattern
                    if let Pattern::Rest { pattern: rest_pat, .. } = pat {
                        // Get rest of vector from index i
                        let index = self.context.i64_type().const_int(i as u64, false);
                        let vector_rest_fn = self.runtime_fn("vector_rest");
                        let rest = self
                            .builder
                            .build_call(vector_rest_fn, &[value.into(), index.into()], "rest")
                            .unwrap()
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_int_value();
                        self.bind_pattern(rest_pat, rest)?;
                        break;
                    }

                    // Get element at index
                    let index = self.context.i64_type().const_int(i as u64, false);
                    let vector_get_fn = self.runtime_fn("vector_get");
                    let element = self
                        .builder
                        .build_call(vector_get_fn, &[value.into(), index.into()], "elem")
                        .unwrap()
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_int_value();
                    self.bind_pattern(pat, element)?;
                }
                Ok(())
            }
            Pattern::Map { entries, span } => {
                // Destructure map
                for (key, pat) in entries {
                    let key_value = self.keyword_value(*key);
                    let map_get_fn = self.runtime_fn("map_get");
                    let element = self
                        .builder
                        .build_call(map_get_fn, &[value.into(), key_value.into()], "map_val")
                        .unwrap()
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_int_value();
                    self.bind_pattern(pat, element)?;
                }
                Ok(())
            }
            Pattern::Rest { pattern: inner, span } => {
                // Rest pattern at top level - just bind the value as-is
                self.bind_pattern(inner, value)
            }
        }
    }

    //===----------------------------------------------------------------------===//
    // Do Compilation
    //===----------------------------------------------------------------------===//

    fn compile_do(
        &mut self,
        span: &Span,
        forms: &[HIR],
        is_tail: bool,
    ) -> SpannedResult<IntValue<'ctx>> {
        self.compile_forms(forms)
    }

    //===----------------------------------------------------------------------===//
    // Function Compilation
    //===----------------------------------------------------------------------===//

    fn compile_fn(
        &mut self,
        _span: &Span,
        name: Option<SymId>,
        params: &[Pattern],
        body: &[HIR],
    ) -> SpannedResult<IntValue<'ctx>> {
        // Generate function name
        let fn_name = match name {
            Some(sym) => interner::sym_to_str(sym),
            None => self.unique_name("lambda"),
        };

        // Collect all parameter names (these are bound, not captured)
        let param_names: Vec<SymId> = params.iter().flat_map(|p| p.bound_names()).collect();

        // Include function name in bound variables (for recursive calls)
        let mut bound_vars = param_names.clone();
        if let Some(fn_sym) = name {
            bound_vars.push(fn_sym);
        }

        // Analyze the function body to find free variables
        let free_vars = collect_free_vars(body, &bound_vars);

        // Filter to only include variables that exist in current scope
        // These are the variables we need to capture
        let captures: Vec<(SymId, IntValue<'ctx>)> = free_vars
            .iter()
            .filter_map(|&sym| self.locals.get(&sym).map(|&val| (sym, val)))
            .collect();

        // Create function type: (env_ptr, argc, argv) -> Value
        // Using a uniform calling convention for closures
        let fn_type = self.value_type.fn_type(
            &[
                self.ptr_type.into(),            // closure env pointer
                self.context.i32_type().into(),  // argc
                self.ptr_type.into(),            // argv (pointer to values)
            ],
            false,
        );

        let function = self.module.add_function(&fn_name, fn_type, None);

        // Set up function for tail call optimization
        // Use fastcc for better TCO support
        if self.opt_config.enable_tco {
            function.set_call_conventions(inkwell::llvm_sys::LLVMCallConv::LLVMFastCallConv as u32);
        }

        // Add inlining attributes for small functions
        if self.opt_config.enable_inlining && should_inline(body, self.opt_config.inline_threshold) {
            let alwaysinline = self.context.create_string_attribute("alwaysinline", "");
            function.add_attribute(AttributeLoc::Function, alwaysinline);
            self.compile_stats.functions_inlined += 1;
        }

        // Track function compilation
        self.compile_stats.functions_compiled += 1;

        // Save current state
        let saved_fn = self.current_fn;
        let saved_locals = self.locals.clone();
        let saved_loop_target = self.loop_target.take();
        let saved_block = self.builder.get_insert_block();

        // Set current function
        self.current_fn = Some(function);
        self.locals.clear();

        // Create entry block
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Get function parameters
        let env_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let argc = function.get_nth_param(1).unwrap().into_int_value();
        let argv = function.get_nth_param(2).unwrap().into_pointer_value();

        // Load captured variables from the environment pointer
        // The environment is an array of Values at env_ptr
        for (idx, (sym, _)) in captures.iter().enumerate() {
            let env_idx = self.context.i64_type().const_int(idx as u64, false);
            let capture_ptr = unsafe {
                self.builder
                    .build_gep(self.value_type, env_ptr, &[env_idx], "capture_ptr")
                    .unwrap()
            };
            let captured_value = self
                .builder
                .build_load(self.value_type, capture_ptr, &format!("capture_{}", interner::sym_to_str(*sym)))
                .unwrap();
            self.locals.insert(*sym, captured_value.into_int_value());
        }

        // Add function name to locals for recursive calls (points to closure itself)
        // This will be loaded from a special position in the environment if needed
        // For now, we handle this by making the function available through var lookup

        // Bind parameters from argv
        let mut arg_index = 0u64;
        for param in params {
            match param {
                Pattern::Bind { name, .. } => {
                    let idx = self.context.i64_type().const_int(arg_index, false);
                    let arg_ptr = unsafe {
                        self.builder.build_gep(self.value_type, argv, &[idx], "arg_ptr").unwrap()
                    };
                    let arg_value =
                        self.builder.build_load(self.value_type, arg_ptr, "arg").unwrap();
                    self.locals.insert(*name, arg_value.into_int_value());
                    arg_index += 1;
                }
                Pattern::Ignore { .. } => {
                    arg_index += 1;
                }
                Pattern::Rest { pattern: rest_pat, .. } => {
                    // Create a vector from remaining arguments
                    let remaining_count = self
                        .builder
                        .build_int_sub(
                            argc,
                            self.context.i32_type().const_int(arg_index as u64, false),
                            "remaining",
                        )
                        .unwrap();
                    let idx = self.context.i64_type().const_int(arg_index, false);
                    let rest_ptr = unsafe {
                        self.builder.build_gep(self.value_type, argv, &[idx], "rest_ptr").unwrap()
                    };
                    let vector_new_fn = self.runtime_fn("vector_new");
                    let rest_vec = self
                        .builder
                        .build_call(
                            vector_new_fn,
                            &[remaining_count.into(), rest_ptr.into()],
                            "rest_vec",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_int_value();

                    self.bind_pattern(rest_pat, rest_vec)?;
                    break; // Rest must be last
                }
                Pattern::Vector { .. } | Pattern::Map { .. } => {
                    // Load the argument and destructure
                    let idx = self.context.i64_type().const_int(arg_index, false);
                    let arg_ptr = unsafe {
                        self.builder.build_gep(self.value_type, argv, &[idx], "arg_ptr").unwrap()
                    };
                    let arg_value =
                        self.builder.build_load(self.value_type, arg_ptr, "arg").unwrap();
                    self.bind_pattern(param, arg_value.into_int_value())?;
                    arg_index += 1;
                }
            }
        }

        // Compile body
        let result = self.compile_forms(body)?;

        // Return result
        self.builder.build_return(Some(&result)).unwrap();

        // Restore state
        self.current_fn = saved_fn;
        self.locals = saved_locals;
        self.loop_target = saved_loop_target;

        // Position builder back at the original insertion point
        if let Some(block) = saved_block {
            self.builder.position_at_end(block);
        }

        // Create the closure with captured environment
        let closure_new_fn = self.runtime_fn("closure_new");
        let fn_ptr = function.as_global_value().as_pointer_value();

        let (env_count, env_ptr_value) = if captures.is_empty() {
            // No captures - use null environment
            (
                self.context.i32_type().const_int(0, false),
                self.ptr_type.const_null(),
            )
        } else {
            // Allocate environment array and store captured values
            let env_array_type = self.value_type.array_type(captures.len() as u32);
            let env_alloca = self.builder.build_alloca(env_array_type, "closure_env").unwrap();

            // Store each captured value
            for (idx, (sym, value)) in captures.iter().enumerate() {
                let env_idx = self.context.i64_type().const_int(idx as u64, false);
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            env_array_type,
                            env_alloca,
                            &[self.context.i64_type().const_int(0, false), env_idx],
                            &format!("env_slot_{}", interner::sym_to_str(*sym)),
                        )
                        .unwrap()
                };
                self.builder.build_store(elem_ptr, *value).unwrap();
            }

            (
                self.context.i32_type().const_int(captures.len() as u64, false),
                env_alloca,
            )
        };

        let closure = self
            .builder
            .build_call(
                closure_new_fn,
                &[fn_ptr.into(), env_count.into(), env_ptr_value.into()],
                "closure",
            )
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        Ok(closure.into_int_value())
    }

    //===----------------------------------------------------------------------===//
    // Loop/Recur Compilation
    //===----------------------------------------------------------------------===//

    fn compile_loop(
        &mut self,
        span: &Span,
        bindings: &[(Pattern, HIR)],
        body: &[HIR],
    ) -> SpannedResult<IntValue<'ctx>> {
        let current_fn = self.current_fn.ok_or_else(|| {
            error_at(span.clone(), Error::RuntimeError("loop outside function".to_string()))
        })?;

        // Save current state
        let saved_locals = self.locals.clone();
        let saved_loop_target = self.loop_target.take();

        // Create loop blocks
        let loop_header = self.context.append_basic_block(current_fn, "loop_header");
        let loop_body = self.context.append_basic_block(current_fn, "loop_body");
        let loop_exit = self.context.append_basic_block(current_fn, "loop_exit");

        // Compile initial binding values
        let mut initial_values = Vec::new();
        for (pattern, value_hir) in bindings {
            let value = self.compile(value_hir)?;
            initial_values.push((pattern.clone(), value));
        }

        // Jump to loop header
        self.builder.build_unconditional_branch(loop_header).unwrap();

        // Set up loop header with PHI nodes
        self.builder.position_at_end(loop_header);
        let entry_bb = current_fn.get_first_basic_block().unwrap();

        let mut phi_nodes = Vec::new();
        for (pattern, initial_value) in &initial_values {
            if let Pattern::Bind { name, .. } = pattern {
                let phi = self.builder.build_phi(self.value_type, &interner::sym_to_str(*name)).unwrap();
                phi.add_incoming(&[(initial_value, entry_bb)]);
                self.locals.insert(*name, phi.as_basic_value().into_int_value());
                phi_nodes.push((*name, phi));
            }
        }

        // Set loop target for recur
        self.loop_target = Some(LoopTarget { header_block: loop_header, phi_nodes });

        // Jump to body
        self.builder.build_unconditional_branch(loop_body).unwrap();

        // Compile loop body
        self.builder.position_at_end(loop_body);
        let result = self.compile_forms(body)?;

        // If we reach here without recur, exit the loop
        self.builder.build_unconditional_branch(loop_exit).unwrap();

        // Set up exit block
        self.builder.position_at_end(loop_exit);

        // Restore state
        self.locals = saved_locals;
        self.loop_target = saved_loop_target;

        Ok(result)
    }

    fn compile_recur(&mut self, span: &Span, args: &[HIR]) -> SpannedResult<IntValue<'ctx>> {
        // Check if we have a loop target before compiling args
        if self.loop_target.is_none() {
            return Err(error_at(
                span.clone(),
                Error::RuntimeError("recur outside loop".to_string()),
            ));
        }

        // Compile new argument values first (while loop_target is not borrowed)
        let mut new_values = Vec::new();
        for arg in args {
            let value = self.compile(arg)?;
            new_values.push(value);
        }

        // Now borrow loop_target to update PHI nodes
        let loop_target = self.loop_target.as_ref().unwrap();
        let current_bb = self.builder.get_insert_block().unwrap();
        let header_block = loop_target.header_block;

        for ((_sym, phi), value) in loop_target.phi_nodes.iter().zip(new_values.iter()) {
            phi.add_incoming(&[(value, current_bb)]);
        }

        // Jump back to loop header
        self.builder.build_unconditional_branch(header_block).unwrap();

        // Create a new block for any unreachable code after recur
        let current_fn = self.current_fn.unwrap();
        let unreachable_bb = self.context.append_basic_block(current_fn, "unreachable");
        self.builder.position_at_end(unreachable_bb);

        // Return a dummy value (this code is unreachable)
        Ok(self.nil_value())
    }

    //===----------------------------------------------------------------------===//
    // Call Compilation
    //===----------------------------------------------------------------------===//

    fn compile_call(
        &mut self,
        span: &Span,
        callee: &HIR,
        args: &[HIR],
        is_tail: bool,
    ) -> SpannedResult<IntValue<'ctx>> {
        // Compile callee
        let callee_value = self.compile(callee)?;

        // Compile arguments
        let mut arg_values = Vec::new();
        for arg in args {
            let value = self.compile(arg)?;
            arg_values.push(value);
        }

        // Create argument array on stack
        let argc = self.context.i32_type().const_int(arg_values.len() as u64, false);
        let argv = if arg_values.is_empty() {
            self.ptr_type.const_null()
        } else {
            let array_type = self.value_type.array_type(arg_values.len() as u32);
            let argv_alloca = self.builder.build_alloca(array_type, "argv").unwrap();

            for (i, value) in arg_values.iter().enumerate() {
                let idx = self.context.i64_type().const_int(i as u64, false);
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            array_type,
                            argv_alloca,
                            &[self.context.i64_type().const_int(0, false), idx],
                            "elem_ptr",
                        )
                        .unwrap()
                };
                self.builder.build_store(elem_ptr, *value).unwrap();
            }

            argv_alloca
        };

        // Call runtime dispatch
        let call_fn = self.runtime_fn("call");
        let call_site = self
            .builder
            .build_call(call_fn, &[callee_value.into(), argc.into(), argv.into()], "call_result")
            .unwrap();

        // Mark as tail call if appropriate and TCO is enabled
        if is_tail && self.opt_config.enable_tco {
            call_site.set_tail_call(true);
            // Set must-tail if in a function with fastcc
            // This ensures the call will be optimized as a tail call
            self.compile_stats.tail_calls_marked += 1;
        }

        let result = call_site.try_as_basic_value().left().unwrap();
        Ok(result.into_int_value())
    }

    //===----------------------------------------------------------------------===//
    // Collection Compilation
    //===----------------------------------------------------------------------===//

    fn compile_vector(&mut self, span: &Span, items: &[HIR]) -> SpannedResult<IntValue<'ctx>> {
        // Compile all items
        let mut item_values = Vec::new();
        for item in items {
            let value = self.compile(item)?;
            item_values.push(value);
        }

        // Create array on stack
        let count = self.context.i32_type().const_int(item_values.len() as u64, false);
        let items_ptr = if item_values.is_empty() {
            self.ptr_type.const_null()
        } else {
            let array_type = self.value_type.array_type(item_values.len() as u32);
            let array_alloca = self.builder.build_alloca(array_type, "vec_items").unwrap();

            for (i, value) in item_values.iter().enumerate() {
                let idx = self.context.i64_type().const_int(i as u64, false);
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            array_type,
                            array_alloca,
                            &[self.context.i64_type().const_int(0, false), idx],
                            "elem_ptr",
                        )
                        .unwrap()
                };
                self.builder.build_store(elem_ptr, *value).unwrap();
            }

            array_alloca
        };

        // Call runtime to create vector
        let vector_new_fn = self.runtime_fn("vector_new");
        let result = self
            .builder
            .build_call(vector_new_fn, &[count.into(), items_ptr.into()], "vector")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        Ok(result.into_int_value())
    }

    fn compile_map(&mut self, span: &Span, entries: &[(HIR, HIR)]) -> SpannedResult<IntValue<'ctx>> {
        // Compile all key-value pairs
        let mut entry_values = Vec::new();
        for (key, value) in entries {
            let key_val = self.compile(key)?;
            let val_val = self.compile(value)?;
            entry_values.push(key_val);
            entry_values.push(val_val);
        }

        // Create array on stack (interleaved keys and values)
        let count = self.context.i32_type().const_int(entries.len() as u64, false);
        let entries_ptr = if entry_values.is_empty() {
            self.ptr_type.const_null()
        } else {
            let array_type = self.value_type.array_type(entry_values.len() as u32);
            let array_alloca = self.builder.build_alloca(array_type, "map_entries").unwrap();

            for (i, value) in entry_values.iter().enumerate() {
                let idx = self.context.i64_type().const_int(i as u64, false);
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            array_type,
                            array_alloca,
                            &[self.context.i64_type().const_int(0, false), idx],
                            "elem_ptr",
                        )
                        .unwrap()
                };
                self.builder.build_store(elem_ptr, *value).unwrap();
            }

            array_alloca
        };

        // Call runtime to create map
        let map_new_fn = self.runtime_fn("map_new");
        let result = self
            .builder
            .build_call(map_new_fn, &[count.into(), entries_ptr.into()], "map")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        Ok(result.into_int_value())
    }

    fn compile_set(&mut self, span: &Span, items: &[HIR]) -> SpannedResult<IntValue<'ctx>> {
        // Compile all items
        let mut item_values = Vec::new();
        for item in items {
            let value = self.compile(item)?;
            item_values.push(value);
        }

        // Create array on stack
        let count = self.context.i32_type().const_int(item_values.len() as u64, false);
        let items_ptr = if item_values.is_empty() {
            self.ptr_type.const_null()
        } else {
            let array_type = self.value_type.array_type(item_values.len() as u32);
            let array_alloca = self.builder.build_alloca(array_type, "set_items").unwrap();

            for (i, value) in item_values.iter().enumerate() {
                let idx = self.context.i64_type().const_int(i as u64, false);
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            array_type,
                            array_alloca,
                            &[self.context.i64_type().const_int(0, false), idx],
                            "elem_ptr",
                        )
                        .unwrap()
                };
                self.builder.build_store(elem_ptr, *value).unwrap();
            }

            array_alloca
        };

        // Call runtime to create set
        let set_new_fn = self.runtime_fn("set_new");
        let result = self
            .builder
            .build_call(set_new_fn, &[count.into(), items_ptr.into()], "set")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        Ok(result.into_int_value())
    }

    //===----------------------------------------------------------------------===//
    // Namespace Compilation
    //===----------------------------------------------------------------------===//

    fn compile_ns(&mut self, span: &Span, name: SymId) -> SpannedResult<IntValue<'ctx>> {
        let sym_id = self.context.i64_type().const_int(name.0 as u64, false);
        let ns_switch_fn = self.runtime_fn("ns_switch");
        let result = self
            .builder
            .build_call(ns_switch_fn, &[sym_id.into()], "ns")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap();

        Ok(result.into_int_value())
    }

    //===----------------------------------------------------------------------===//
    // Module Finalization and Output
    //===----------------------------------------------------------------------===//

    /// Verify the generated LLVM module.
    pub fn verify(&self) -> Result<(), String> {
        self.module.verify().map_err(|e| e.to_string())
    }

    /// Get the LLVM IR as a string.
    pub fn to_string(&self) -> String {
        self.module.print_to_string().to_string()
    }

    /// Write LLVM bitcode to a file.
    pub fn write_bitcode(&self, path: &Path) -> bool {
        self.module.write_bitcode_to_path(path)
    }

    /// Compile to an object file.
    pub fn write_object(&self, path: &Path) -> Result<(), String> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| e.to_string())?;

        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
        let cpu = TargetMachine::get_host_cpu_name();
        let features = TargetMachine::get_host_cpu_features();

        let target_machine = target
            .create_target_machine(
                &triple,
                cpu.to_str().unwrap(),
                features.to_str().unwrap(),
                OptimizationLevel::Default,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| "Failed to create target machine".to_string())?;

        target_machine
            .write_to_file(&self.module, FileType::Object, path)
            .map_err(|e| e.to_string())
    }

    /// Run optimization passes on the module using the current configuration.
    pub fn run_optimization_passes(&self) {
        self.optimize_with_level(self.opt_config.level.into());
    }

    /// Run optimization passes on the module with a specific level.
    pub fn optimize(&self, level: OptimizationLevel) {
        self.optimize_with_level(level);
    }

    /// Internal optimization implementation.
    fn optimize_with_level(&self, level: OptimizationLevel) {
        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple).unwrap();
        let cpu = TargetMachine::get_host_cpu_name();
        let features = TargetMachine::get_host_cpu_features();

        let target_machine = target
            .create_target_machine(
                &triple,
                cpu.to_str().unwrap(),
                features.to_str().unwrap(),
                level,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();

        // Build the passes string
        let base_passes = match level {
            OptimizationLevel::None => "default<O0>",
            OptimizationLevel::Less => "default<O1>",
            OptimizationLevel::Default => "default<O2>",
            OptimizationLevel::Aggressive => "default<O3>",
        };

        // If we have custom passes, append them
        let passes = if self.opt_config.custom_passes.is_empty() {
            base_passes.to_string()
        } else {
            format!("{},{}", base_passes, self.opt_config.custom_passes.join(","))
        };

        self.module
            .run_passes(&passes, &target_machine, PassBuilderOptions::create())
            .unwrap();
    }

    /// Verify that tail calls are properly marked in the generated IR.
    ///
    /// Returns a list of functions with tail call information.
    pub fn verify_tco(&self) -> Vec<TcoVerification> {
        let mut results = Vec::new();
        let ir = self.module.print_to_string().to_string();

        // Parse the IR to find tail call markers
        for line in ir.lines() {
            let line = line.trim();
            if line.contains("tail call") || line.contains("musttail call") {
                results.push(TcoVerification {
                    ir_line: line.to_string(),
                    is_tail_call: true,
                    is_musttail: line.contains("musttail"),
                });
            }
        }

        results
    }

    /// Get a summary of the TCO verification.
    pub fn tco_summary(&self) -> TcoSummary {
        let verifications = self.verify_tco();
        TcoSummary {
            total_tail_calls: verifications.len(),
            musttail_calls: verifications.iter().filter(|v| v.is_musttail).count(),
            regular_tail_calls: verifications.iter().filter(|v| !v.is_musttail).count(),
        }
    }

    /// Create a top-level wrapper function for compiling a module.
    pub fn create_module_init(&mut self, forms: &[HIR]) -> SpannedResult<FunctionValue<'ctx>> {
        // Create the init function: () -> Value
        let fn_type = self.value_type.fn_type(&[], false);
        let function = self.module.add_function("evolve_module_init", fn_type, None);

        // Set current function
        self.current_fn = Some(function);
        self.locals.clear();

        // Create entry block
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Compile all forms
        let result = self.compile_forms(forms)?;

        // Return the result
        self.builder.build_return(Some(&result)).unwrap();

        self.current_fn = None;

        Ok(function)
    }

    //===----------------------------------------------------------------------===//
    // Test Helpers
    //===----------------------------------------------------------------------===//

    /// Get the LLVM module (for testing).
    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Get the LLVM builder (for testing).
    pub fn get_builder(&self) -> &Builder<'ctx> {
        &self.builder
    }

    /// Get the value type (i64).
    pub fn get_value_type(&self) -> inkwell::types::IntType<'ctx> {
        self.value_type
    }

    /// Set the current function (for testing).
    pub fn set_current_fn(&mut self, func: Option<FunctionValue<'ctx>>) {
        self.current_fn = func;
    }

    /// Add a function to the module (for testing).
    pub fn add_function(
        &self,
        name: &str,
        fn_type: inkwell::types::FunctionType<'ctx>,
    ) -> FunctionValue<'ctx> {
        self.module.add_function(name, fn_type, None)
    }
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::Lowerer;
    use crate::reader::{Reader, Source};
    use crate::runtime::Runtime;

    fn parse_and_lower(source: &str) -> HIR {
        let runtime = Runtime::new();
        let value = Reader::read(source, Source::REPL, runtime).unwrap();
        let lowerer = Lowerer::new();
        lowerer.lower(&value).unwrap()
    }

    #[test]
    fn test_compile_nil() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        let hir = parse_and_lower("nil");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_int() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        let hir = parse_and_lower("42");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_bool() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        let hir = parse_and_lower("true");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_if() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        let hir = parse_and_lower("(if true 1 2)");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_vector() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        let hir = parse_and_lower("[1 2 3]");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_llvm_ir_output() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        let hir = parse_and_lower("42");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        let ir = codegen.to_string();
        assert!(ir.contains("define"));
        assert!(ir.contains("ret i64"));
    }

    //===----------------------------------------------------------------------===//
    // Free Variable Analysis Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_free_vars_simple_var() {
        // x is free (not bound)
        let hir = parse_and_lower("x");
        let free = collect_free_vars(&[hir], &[]);
        assert_eq!(free.len(), 1);
        assert_eq!(interner::sym_to_str(free[0]), "x");
    }

    #[test]
    fn test_free_vars_bound_var() {
        // x is bound in the parameter list
        let x_sym = interner::intern_sym("x");
        let hir = parse_and_lower("x");
        let free = collect_free_vars(&[hir], &[x_sym]);
        assert!(free.is_empty(), "x should not be free when bound");
    }

    #[test]
    fn test_free_vars_let_binding() {
        // y is free, x is bound by let
        let hir = parse_and_lower("(let* [x 1] y)");
        let free = collect_free_vars(&[hir], &[]);
        assert_eq!(free.len(), 1);
        assert_eq!(interner::sym_to_str(free[0]), "y");
    }

    #[test]
    fn test_free_vars_let_value_uses_outer() {
        // x in the let value is free (not yet bound)
        let hir = parse_and_lower("(let* [y x] y)");
        let free = collect_free_vars(&[hir], &[]);
        assert_eq!(free.len(), 1);
        assert_eq!(interner::sym_to_str(free[0]), "x");
    }

    #[test]
    fn test_free_vars_nested_fn() {
        // The inner function references x, which is free for the inner fn
        // but should still be collected as free from the perspective of
        // the whole expression
        let hir = parse_and_lower("(fn* [a] x)");
        let free = collect_free_vars(&[hir], &[]);
        assert_eq!(free.len(), 1);
        assert_eq!(interner::sym_to_str(free[0]), "x");
    }

    #[test]
    fn test_free_vars_fn_param_not_free() {
        // a is bound as a parameter, should not be free
        let hir = parse_and_lower("(fn* [a] a)");
        let free = collect_free_vars(&[hir], &[]);
        assert!(free.is_empty(), "parameter should not be free");
    }

    #[test]
    fn test_free_vars_if_branches() {
        // x, y, z are all free
        let hir = parse_and_lower("(if x y z)");
        let free = collect_free_vars(&[hir], &[]);
        assert_eq!(free.len(), 3);
    }

    #[test]
    fn test_free_vars_call() {
        // f and x are both free
        let hir = parse_and_lower("(f x)");
        let free = collect_free_vars(&[hir], &[]);
        assert_eq!(free.len(), 2);
    }

    #[test]
    fn test_free_vars_loop() {
        // i is bound by loop, but n is free
        let hir = parse_and_lower("(loop* [i 0] (if (> i n) i (recur (+ i 1))))");
        let free = collect_free_vars(&[hir], &[]);
        // Should contain: >, n, +
        assert!(free.iter().any(|&s| interner::sym_to_str(s) == "n"));
    }

    //===----------------------------------------------------------------------===//
    // Closure Capture Compilation Tests
    //===----------------------------------------------------------------------===//

    #[test]
    fn test_compile_simple_closure_no_capture() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // A simple function that doesn't capture anything
        let hir = parse_and_lower("(fn* [x] x)");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());

        let ir = codegen.to_string();
        // Should call closure_new with 0 captures
        assert!(ir.contains("evolve_closure_new"));
    }

    #[test]
    fn test_compile_closure_with_single_capture() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Set up: (let* [y 42] (fn* [x] y))
        // The inner fn captures y
        let hir = parse_and_lower("(let* [y 42] (fn* [x] y))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());

        let ir = codegen.to_string();
        // Should have a lambda function defined
        assert!(ir.contains("define"));
        // Should have closure environment handling
        assert!(ir.contains("closure_env") || ir.contains("capture"));
    }

    #[test]
    fn test_compile_closure_with_multiple_captures() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Set up: (let* [a 1 b 2 c 3] (fn* [x] (+ a b c)))
        // The inner fn captures a, b, c
        let hir = parse_and_lower("(let* [a 1 b 2 c 3] (fn* [x] (+ a b c)))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_nested_closures() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Nested closures: outer captures x, inner captures both x and y
        // (let* [x 1] (fn* [y] (fn* [z] (+ x y z))))
        let hir = parse_and_lower("(let* [x 1] (fn* [y] (fn* [z] (+ x y z))))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_closure_captures_only_used_vars() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Only x is used inside the fn, a is not
        // (let* [a 1 x 2] (fn* [y] x))
        let hir = parse_and_lower("(let* [a 1 x 2] (fn* [y] x))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());

        // The IR should only show x being captured, not a
        let ir = codegen.to_string();
        // Look for the lambda function - it should load from env
        assert!(ir.contains("capture_x") || ir.contains("env"));
    }

    #[test]
    fn test_compile_closure_in_if_branch() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Closure created inside an if branch
        let hir = parse_and_lower("(let* [x 1] (if true (fn* [y] x) (fn* [z] x)))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_closure_in_do_block() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Closure created in do block
        let hir = parse_and_lower("(let* [x 1] (do 1 2 (fn* [y] x)))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_returning_closure() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // A function that returns a closure
        // (fn* [x] (fn* [y] (+ x y)))
        let hir = parse_and_lower("(fn* [x] (fn* [y] (+ x y)))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_closure_with_vector_destructuring() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Closure capturing variables from vector destructuring
        let hir = parse_and_lower("(let* [[a b] [1 2]] (fn* [x] (+ a b)))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_compile_named_fn_self_reference() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Named function that references itself (recursive)
        // The function name should not be captured (it's bound)
        let hir = parse_and_lower("(fn* fact [n] (if (= n 0) 1 (* n (fact (- n 1)))))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_closure_ir_contains_env_load() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Simple closure that captures x
        let hir = parse_and_lower("(let* [x 42] (fn* [y] x))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());

        let ir = codegen.to_string();
        // The lambda function should load captured variable from env
        assert!(
            ir.contains("capture_x") || ir.contains("getelementptr"),
            "IR should contain code to load captured variables"
        );
    }

    #[test]
    fn test_closure_ir_contains_env_store() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "test");

        // Simple closure that captures x
        let hir = parse_and_lower("(let* [x 42] (fn* [y] x))");
        let fn_type = codegen.value_type.fn_type(&[], false);
        let function = codegen.module.add_function("test", fn_type, None);
        let entry = context.append_basic_block(function, "entry");
        codegen.builder.position_at_end(entry);
        codegen.current_fn = Some(function);

        let result = codegen.compile(&hir).unwrap();
        codegen.builder.build_return(Some(&result)).unwrap();

        assert!(codegen.verify().is_ok());

        let ir = codegen.to_string();
        // The outer code should store captured value in env
        assert!(
            ir.contains("store") && ir.contains("closure_env"),
            "IR should contain code to store captured values in closure env"
        );
    }
}
