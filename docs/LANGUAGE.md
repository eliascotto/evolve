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
* Dynamic vars (Clojure-style) are opt-in and explicitly declared with `^:dynamic` metadata: `(def ^:dynamic *var* value)`.
* Shadowing permitted.

## Control & Errors

* Tail calls: guaranteed TCO for self-tail and proper tail positions the compiler recognizes.
* Errors: Common Lisp–style conditions, signals, and restarts (restarts are dynamic handlers that can resume).
* Exceptions map to zero-cost EH in codegen (LLVM landing pads with unwind tables).

## Types

* Dynamic language in v0.
* Optional/static typing later (TypeScript-like: structural records, unions, generics, protocols/traits). Gradual typing roadmap, erased at codegen unless specialized.
* Protocols: similar to Clojure protocols - define sets of functions that types can implement, enabling polymorphic dispatch without inheritance.

## Collections

* Persistent: `list`, `vector`, `map` (insertion-ordered), `set`.
* Transients: opt-in, single-owner, for batch mutation during pipelines.
  * `(transient coll)` - create transient from persistent collection
  * `(conj! t x)`, `(assoc! t k v)`, `(dissoc! t k)` - mutating operations
  * `(persistent! t)` - convert back to persistent collection

## Macros & Phases

* Two phases: compile-time (macroexpansion) and run-time.
* Macros: Clojure-like (unhygienic with gensym tools); user-defined reader macros not present in v0.
* Multiple-pass expansion; `macroexpand` / `macroexpand-1`.

## Reader / AST

* Reader produces a CST (Concrete Syntax Tree) with spans (file/line/col) and metadata (docstrings, attributes).
* CST preserves all syntax details including whitespace and comments, unlike an AST which abstracts syntax.
* Session-global symbol/keyword interner.

# 2) Core Forms

**Special forms (kernel):**

* `def`, `defn`, `defmacro`, `fn`/`lambda`, `if`, `let`, `do`, `quote`, `quasiquote`, `unquote`, `loop`/`recur`, `set!` (for vars/atoms only), `try`/`catch`/`finally`.
* Evaluation order of operands: left-to-right (like Clojure).
* Quasiquote syntax: `` `(list ,x ,@xs) `` where `,` unquotes and `,@` splices.

**Condition system (surface):**

* `(signal condition & [data])` - signal a condition
* `(handler-bind {Type handler-fn} body…)` - bind condition handlers
* `(restart-case body (name [args…] body…))` - provide restarts
* `(try body… (catch Type var body…) (finally body…))` - error handling forms
* FFI boundaries translate condition → status/error codes by default; crossing exceptions disabled unless explicitly enabled.

**Destructuring:**

* Clojure-style for `let`, `fn` parameters, `loop`, and binding forms.

**STM & atoms (surface):**

* `(atom x)` - create atom, `(swap! a f ...)` - update atom with function, `(reset! a v)` - set atom value, `(deref a)` - read atom value
* `(ref x)` - create STM ref, `(dosync body…)` - transaction, `(alter r f ...)` - update ref, `(commute r f ...)` - commutative update
* `(agent x)` - create agent, `(send a f ...)` - send action to agent, `(await a)` - wait for agent actions to complete, `(agent-error a)` - get error if any

**Modules & visibility controls surface forms:**

* `(ns my.ns)` - define namespace, `(ns my.ns :require [foo.bar :as fb])` - namespace with requires
* `^:public` metadata marks exports: `(def ^:public x 42)` or `(defn ^:public my-fn [x] ...)`
* `(:require [foo.bar :as fb :refer [x y] :rename {old new}])` - require with aliases, selective refer, rename
* Global import as `(:require [foo.bar :all])` - discouraged, can be disabled by compiler flag
* Compiler "strict mode" flag can reject glob imports.

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
* Artifacts: object files (`.o`/`.obj`) + metadata sidecar (`.meta`) in binary format for module summaries, export tables, docstrings, and inlineability hints.
* Long-term: support FASL-like cache for faster load in the REPL.

**Dependencies & reproducibility**

* Package manager with lockfile; exact versions only (no ranges) for reproducible builds.
* Project configuration and dependencies declared in EDN format (project file).
* Package registry support planned for later versions.

**Optimization boundaries**

* Optimize per module by default; opt-in ThinLTO/whole-program at release. ThinLTO operates via per-module summaries and cross-module importing.

**ABI**

* C ABI at the boundary from day one; internal functions may use a custom fast CC. LLVM’s default `ccc` matches the platform C ABI; `fastcc` available for internal hot paths.

# 4) Runtime & Representation

**Representation**

* Boxed, tagged pointers as the uniform "Value".
* Escape analysis: non-escaping closures/temps prefer stack.

**Metadata**

* Metadata format follows Clojure conventions: `^:keyword` for keywords, `^{:key val}` for maps.
* Metadata can be attached to symbols, collections, and function definitions.
* Common metadata: `:doc`, `:public`, `:dynamic`, `:private`, `:test`.

**Resources**

* RAII-style `with` blocks for foreign handles in stdlib (finalizers avoided in v0).
* FFI safety details TBD.

**Runtime services**

* Allocator, scheduler, atomics/STM, reflection, printer, reader, condition/restart system, module loader.

**Debug info**

* Emit DWARF (and CodeView on Windows) with line tables mapping CST spans to machine code; compatible with GDB/LLDB/VS.

# 5) Code Generation & Optimization

**Code generation**

* Conditions/errors map to zero-cost exception handling. Do not let exceptions cross the FFI boundary by default.

**Optimization**

* Dev: `-O0 -g` + frame pointers; optional sanitizers.
* Release: `-O2` + ThinLTO + optional PGO. Consider O3 only for hot numeric kernels.
* Custom passes (iterative rollout): boxing-elimination; RC optimizations; closure-escape; transient-datastructures; arity specialization; TCO enforcement.

**Debugging**

* DWARF/CodeView emission; `disassemble`/`dump-ir` REPL commands; macroexpansion traces.

# 6) Tooling

* Single CLI (`evolve build|run|repl|test`), shebang support.
* Formatter; LSP (hover, go-to-def, doc on hover, macroexpansion preview).
* Built-in test runner; markdown docstrings exported to docs.
