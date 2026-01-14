# Implementation Specifications

This document describes implementation-specific details of the Evolve compiler and runtime, including memory management, code generation, and compiler internals.

## Memory Management: Staged ARC

Evolve uses a staged Automatic Reference Counting (ARC) strategy, evolving from baseline safety to latency-friendly optimizations and high-throughput enhancements.

### Baseline (v0)

* All heap objects behind `Arc` (or custom RC header).
* Persistent data structure nodes use `Arc` for sharing.
* Atoms/refs/agents hold `Arc<Value>` and update via Compare-And-Swap (CAS).
* Weak references (compiler-internal) provided for back-edges to break cycles.
* Tiny, opt-in cycle collector automatically triggered only when closure cycles are detected. Memory pressure is defined as when the process is consuming too much memory relative to available system resources.

### Latency-Friendly Performance

* Arenas/regions for temporaries inside evaluation and macroexpansion; promote to ARC on escape.
* Transients (Clojure-like) implemented with arenas to minimize churn during batch updates.
* Goal: "discard immediately when superseded or region ends."

### Throughput Upgrades (Phase 2)

* Refcount deferral/batching on hot structures to reduce atomic contention.
* Line/region recycling (Immix-like) for ARC-managed blocks to improve locality and reduce fragmentation.
* Optional: small concurrent tracer to clean cycles and improve locality (RCImmix direction).

### Ergonomics & Safety

* Caches/memoizers bounded by default (LRU/TTL/size) to prevent unbounded growth.
* FFI benefits from stable addresses (ARC perk); no need for pinning in common cases.
* Profiling hooks (dev tools): inc/dec counters, arena stats, atom contention metrics. API TBD.

## Code Generation

### Backends

* x86-64 and AArch64 for Linux/macOS/Windows MVP.

### Calling Conventions

* External/FFI: platform C calling convention (`ccc`).
* Internal: custom fast CC for performance-critical paths; still LLVM-level (`fastcc`) where profitable.

### Exceptions

* Map conditions/errors to LLVM's zero-cost EH (landing pads). Do not let exceptions cross the FFI boundary by default.

### JIT

* ORC LLJIT with IR transform layers, lazy compile on first call, object linking via ObjectLinkingLayer/JITLink. Cache compiled objects to disk.

### Optimization Pipeline (defaults)

* Dev: `-O0 -g` + frame pointers; optional ASan/UBSan.
* Release: `-O2` + ThinLTO + optional PGO. Consider O3 only for hot numeric kernels.
* Custom passes (iterative rollout): boxing-elimination; RC optimizations; closure-escape; transient-datastructures; arity specialization; TCO enforcement.

## Compiler Internals

### Symbol/Keyword Interner

* Session-global, shardable symbol/keyword interner for efficient symbol comparison and storage.

### Expansion Barriers

* Rust expansion barriers in the compiler prevent infinite macro expansion loops.
* Multiple-pass expansion with barriers; `macroexpand` / `macroexpand-1` available.

### Reader

* Reader produces a CST (Concrete Syntax Tree) with spans (file/line/col) and metadata (docstrings, attributes).
* CST preserves all syntax details including whitespace and comments.

### Module Artifacts

* Separate compilation on by default.
* Artifacts: object files (`.o`/`.obj`) + metadata sidecar (`.meta`) in binary format for module summaries, export tables, docstrings, and inlineability hints.
* Long-term: support FASL-like cache for faster load in the REPL.

### Optimization Boundaries

* Optimize per module by default; opt-in ThinLTO/whole-program at release. ThinLTO operates via per-module summaries and cross-module importing.

### ABI

* C ABI at the boundary from day one; internal functions may use a custom fast CC. LLVM's default `ccc` matches the platform C ABI; `fastcc` available for internal hot paths.

## Runtime Services

* Allocator (ARC + arenas), scheduler, atomics/STM, reflection, printer, reader, condition/restart system, module loader, cycle collector (opt-in), profiling hooks.

## Debug Info

* Emit DWARF (and CodeView on Windows) with line tables mapping CST spans to machine code; compatible with GDB/LLDB/VS.

