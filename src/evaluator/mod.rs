//! Evaluation engines that power the high-level `Evaluator`.
//!
//! This module hosts experimental and alternative evaluators.  The goal is to
//! make it easy to reason about the control-flow machinery independently from
//! the user-facing API surface that lives in `eval.rs`.
//! 
//! The first implementation that ships here is a trampoline-based interpreter
//! that keeps evaluation in an explicit loop so tail calls do not rely on the
//! Rust call-stack.

pub mod trampoline;



