# Language Goals & Semantics

## Purpose

Evolve is a modern, compiled Lisp for fast iterative development, high performance, and C interop.

## Priorities

1. Performance → 2. Simplicity → 3. Safety → 4. Expressiveness → 5. Interoperability.

## Execution & Effects

* Evaluation: strict (Clojure-like eager), left-to-right arg eval; lazy abstractions provided in libraries.
* Purity: “pure by default”; effects via explicit APIs (IO, STM, atoms).
* Concurrency: Software Transactional Memory + atoms/refs as primitives. Threading provided by the runtime scheduler.
* Determinism: aim for reproducible results where feasible; non-determinism exposed only via opt-in APIs.

## Values & Equality

* Numbers/strings/collections: modelled after Clojure semantics (ratios later; start with integers & IEEE floats; bigints later).
* Truthiness: only `nil` and `false` are falsey.
* Equality: Clojure-like `=` (value), `identical?` (object identity); ordered comparisons for ordered types.
* Strings: Unicode-first; NFC normalization for equality; substring/length are grapheme-aware.

## Names, Binding, Scope

* Lexical scope by default.
* Dynamic vars (Clojure-style) are opt-in and explicitly declared.
* Shadowing permitted.

## Control & Errors

* Tail calls: guaranteed TCO for self-tail and proper tail positions the compiler recognizes.
* Errors: Common Lisp–style conditions, signals, and restarts (restarts are dynamic handlers that can resume).
* Exceptions map to zero-cost EH in codegen (LLVM landing pads with unwind tables).

## Types

* Dynamic language in v0.
* Optional/static typing later (TypeScript-like: structural records, unions, generics, protocols/traits). Gradual typing roadmap, erased at codegen unless specialized.

## Collections

* Persistent: `list`, `vector`, `map` (insertion-ordered), `set`, `deque`.
* Transients: opt-in, single-owner, for batch mutation during pipelines.

## Macros & Phases

* Two phases: compile-time (macroexpansion) and run-time.
* Macros: Clojure-like (unhygienic with gensym tools); reader macros deferred.
* Multiple-pass expansion with expansion barriers; `macroexpand` / `macroexpand-1`.

## Reader / AST

* Reader produces a CST with spans (file/line/col) and metadata (docstrings, attributes).
* Session-global, shardable symbol/keyword interner.

# 2) Core Forms

**Special forms (kernel):**

* `def`, `defmacro`, `fn`/`lambda`, `if`, `let`, `do`, `quote`, `quasiquote`, `unquote`, `loop`/`recur`, `set!` (for vars/atoms only).
* Evaluation order of operands: left-to-right (like Clojure).

**Condition system (surface):**

* `(signal condition & [data])`
* `(handler-bind {Type handler-fn} body…)`
* `(restart-case body (name [args…] body…))`
* FFI boundaries translate condition → status/error codes by default; crossing exceptions disabled unless explicitly enabled.

**Destructuring:**

* Clojure-style for `let`, `fn` parameters, `loop`, and binding forms.

**STM & atoms (surface):**

* `(atom x)`, `(swap! a f ...)`, `(reset! a v)`
* `(ref x)`, `(dosync body…)`, `(alter r f ...)`, `(commute r f ...)`

**Modules & visibility controls surface forms:**

* `(ns my.ns :as m …)`, `^:public` metadata (or `defn-pub`) marks exports.
* `(:require [foo.bar :as fb :refer [x y] :rename {old new}])`
* Global import as `(:require [foo.bar :all])`
* Compiler “strict mode” flag can reject glob imports.

# 3) Modules & Package System

**Unit & naming**

* One source file = one **namespace** (Clojure-like namespaces; namespaced keywords).
* Private by default; export via `^:public` or explicit public defs.

**Imports**

* `:as` aliases; selective `:refer`; optional `:rename`; glob import is available via `:all` but discouraged (can be disabled by compiler flag).

**Init order**

* Top-level has effects; load order is: read → macroexpand → compile → link/load → execute top-level.
* Cycles allowed for values/functions via stubs; circular macro dependencies are errors.

**Separate compilation & artifacts**

* Separate compilation on by default.
* Artifacts: object files (`.o`/`.obj`) + metadata sidecar (`.meta`) for module summaries, export tables, docstrings, and inlineability hints.
* Long-term: support FASL-like cache for faster load in the REPL.

**Dependencies & reproducibility**

* Package manager with lockfile; exact versions only (no ranges) for reproducible builds.

**Optimization boundaries**

* Optimize per module by default; opt-in ThinLTO/whole-program at release. ThinLTO operates via per-module summaries and cross-module importing.

**ABI**

* C ABI at the boundary from day one; internal functions may use a custom fast CC. LLVM’s default `ccc` matches the platform C ABI; `fastcc` available for internal hot paths.

# 4) Runtime, Memory & Representation

**Representation**

* Boxed, tagged pointers as the uniform “Value”.
* Escape analysis: non-escaping closures/temps prefer stack.
* ARC/RC for heap allocations that escape; regions/arenas for short-lived phases (compiler-managed). Goal: “discard immediately when superseded or region ends.”

**Resources**

* FFI safety line TBD (see open items below).
* RAII-style `with` blocks for foreign handles in stdlib (finalizers avoided in v0).

**Runtime services**

* Allocator, scheduler, atomics/STM, reflection, printer, reader, condition/restart system, module loader.

**Debug info**

* Emit DWARF (and CodeView on Windows) with line tables mapping CST spans to machine code; compatible with GDB/LLDB/VS.

# 5) LLVM & JIT Workflow

**Backends**

* x86-64 and AArch64 for Linux/macOS/Windows MVP.

**Calling conventions**

* External/FFI: platform C calling convention (`ccc`).
* Internal: custom fast CC for performance-critical paths; still LLVM-level (`fastcc`) where profitable.

**Exceptions**

* Map conditions/errors to LLVM’s zero-cost EH (landing pads). Do not let exceptions cross the FFI boundary by default.

**JIT**

* ORC LLJIT with IR transform layers, lazy compile on first call, object linking via ObjectLinkingLayer/JITLink. Cache compiled objects to disk. 

**Optimization pipeline (defaults)**

* Dev: `-O0 -g` + frame pointers; optional ASan/UBSan.
* Release: `-O2` + ThinLTO + optional PGO. Consider O3 only for hot numeric kernels.
* Custom passes (iterative rollout): boxing-elimination; RC optimizations; closure-escape; transient-datastructures; arity specialization; TCO enforcement.

**Debugging**

* DWARF/CodeView emission; `disassemble`/`dump-ir` REPL commands; macroexpansion traces.

# 6) Tooling

* Single CLI (`evolve build|run|repl|test`), shebang support.
* Formatter; LSP (hover, go-to-def, doc on hover, macroexpansion preview).
* Built-in test runner; markdown docstrings exported to docs.
