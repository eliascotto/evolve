# Incremental Build Plan for Evolve Compiler

This document outlines a step-by-step approach to building the Evolve compiler from scratch, starting with a working interpreter and incrementally adding features and performance improvements.

## Philosophy

**Start simple, iterate fast.** We begin with a tree-walking interpreter to validate language semantics, then gradually add compilation, optimizations, and advanced features. Each phase produces a working system that can be tested and used.

### Why Interpreter First?

1. **Fast feedback loop**: Test language features immediately
2. **Validate semantics**: Ensure the language design works before optimizing
3. **Useful tool**: Interpreter remains useful for REPL, macro expansion, and development
4. **Incremental compilation**: Can compile functions incrementally while keeping interpreter for top-level

### Development Workflow

- **Phase 0-2**: Get basic interpreter working (can run simple programs)
- **Phase 3-4**: Add collections and macros (can write real programs)
- **Phase 5**: Add modules (can organize code)
- **Phase 6-7**: Add compiler (can generate fast code)
- **Phase 8-10**: Polish and optimize

---

## Phase 0: Project Setup & Infrastructure

### Purpose
Establish the project structure, tooling, and basic infrastructure needed for development.

### Steps

1. **Initialize Rust project**
   ```bash
   cargo new evolve --lib
   cd evolve
   ```

2. **Set up project structure**
   ```
   evolve/
   ├── Cargo.toml
   ├── src/
   │   ├── lib.rs
   │   ├── main.rs
   │   ├── value.rs          # Value representation
   │   ├── reader.rs          # Reader/parser
   │   ├── env.rs             # Environment/bindings
   │   ├── eval.rs            # Evaluator
   │   ├── error.rs           # Error types
   │   └── interner.rs        # Symbol/keyword interning
   ├── tests/
   │   └── integration/
   ├── examples/
   └── docs/
   ```

3. **Add dependencies to `Cargo.toml`**
   ```toml
   [dependencies]
   # For parsing (choose one approach):
   # Option A: Use a parser combinator library
   nom = "7.1"
   # Option B: Use a lexer generator
   logos = "0.13"
   
   # For error handling
   thiserror = "1.0"
   
   # For testing
   [dev-dependencies]
   rstest = "0.18"
   ```

4. **Set up basic error types**
   ```rust
   // src/error.rs
   use thiserror::Error;
   
   #[derive(Debug, Error)]
   pub enum Error {
       #[error("Syntax error: {0}")]
       Syntax(String),
       #[error("Runtime error: {0}")]
       Runtime(String),
       #[error("Type error: expected {expected}, got {actual}")]
       Type { expected: String, actual: String },
   }
   
   pub type Result<T> = std::result::Result<T, Error>;
   ```

5. **Set up basic interner**
   ```rust
   // src/interner.rs
   use std::collections::HashMap;
   use std::sync::Mutex;
   
   pub type SymId = u32;
   pub type KeywId = u32;
   pub type NsId = u32;
   
   pub struct Interner {
       symbols: Mutex<HashMap<String, SymId>>,
       keywords: Mutex<HashMap<String, KeywId>>,
       // ... reverse maps for lookup
   }
   ```

### Deliverables Checklist

- [ ] Rust project initialized with proper structure
- [ ] Basic error types defined (`Error`, `Result`)
- [ ] Symbol/keyword interner skeleton in place
- [ ] Project compiles without errors
- [ ] Basic test framework set up (can write and run a simple test)

### Testing

```rust
// tests/integration/phase0_test.rs
#[cfg(test)]
mod tests {
    #[test]
    fn test_project_setup() {
        assert!(true); // Placeholder
    }
}
```

---

## Phase 1: Reader & CST (Concrete Syntax Tree)

### Purpose
Build the reader that parses Evolve source code into a Concrete Syntax Tree (CST) that preserves all syntax information including spans and metadata.

### Steps

1. **Define CST Value types**
   ```rust
   // src/value.rs
   use logos::Span;
   use std::collections::BTreeMap;
   use crate::interner::{KeywId, SymId};
   
   #[derive(Debug, Clone)]
   pub enum Value {
       Nil { span: Span },
       Bool { span: Span, value: bool },
       Int { span: Span, value: i64 },
       Float { span: Span, value: f64 },
       String { span: Span, value: String },
       Char { span: Span, value: char },
       Symbol { span: Span, value: SymId },
       Keyword { span: Span, value: KeywId },
       List { 
           span: Span, 
           value: Vec<Value>,
           meta: Option<BTreeMap<KeywId, Value>>,
       },
       Vector { 
           span: Span, 
           value: Vec<Value>,
           meta: Option<BTreeMap<KeywId, Value>>,
       },
       Map { 
           span: Span, 
           value: BTreeMap<Value, Value>,
           meta: Option<BTreeMap<KeywId, Value>>,
       },
       Set { 
           span: Span, 
           value: BTreeSet<Value>,
           meta: Option<BTreeMap<KeywId, Value>>,
       },
   }
   ```

2. **Implement tokenizer/lexer**
   
   **Option A: Using `logos` (recommended for performance)**
   ```rust
   // src/reader.rs
   use logos::Logos;
   
   #[derive(Logos, Debug, PartialEq)]
   enum Token {
       #[token("(")]
       LParen,
       #[token(")")]
       RParen,
       #[token("[")]
       LBracket,
       #[token("]")]
       RBracket,
       #[token("{")]
       LBrace,
       #[token("}")]
       RBrace,
       #[token("'")]
       Quote,
       #[token("`")]
       QuasiQuote,
       #[token(",")]
       Unquote,
       #[token(",@")]
       UnquoteSplice,
       #[token("@")]
       Deref,
       #[token("~")]
       UnquoteAlt,
       #[token("~@")]
       UnquoteSpliceAlt,
       
       #[regex(r#""([^"\\]|\\.)*""#)]
       StringLit,
       #[regex(r"-?[0-9]+")]
       IntLit,
       #[regex(r"-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?")]
       FloatLit,
       #[regex(r"#[^ \t\n\r\(\)\[\]\{\}]+")]
       Keyword,
       #[regex(r"[^ \t\n\r\(\)\[\]\{\}'`,@~]+")]
       Symbol,
       
       #[regex(r"[ \t\n\r]+", logos::skip)]
       Whitespace,
       #[regex(r";[^\n]*", logos::skip)]
       Comment,
   }
   ```

   **Option B: Using `nom` (more flexible, functional style)**
   ```rust
   // Alternative approach with nom
   use nom::IResult;
   
   fn parse_value(input: &str) -> IResult<&str, Value> {
       // Implementation using nom combinators
   }
   ```

3. **Implement reader functions**
   ```rust
   // src/reader.rs
   pub struct Reader {
       source: String,
       interner: Arc<Interner>,
   }
   
   impl Reader {
       pub fn new(source: String, interner: Arc<Interner>) -> Self {
           Self { source, interner }
       }
       
       pub fn read(&self) -> Result<Value> {
           // Tokenize and parse into CST
       }
       
       fn read_form(&self, tokens: &mut TokenIter) -> Result<Value> {
           // Dispatch based on first token
       }
       
       fn read_list(&self, tokens: &mut TokenIter) -> Result<Value> {
           // Read list: (a b c)
       }
       
       fn read_vector(&self, tokens: &mut TokenIter) -> Result<Value> {
           // Read vector: [a b c]
       }
       
       fn read_map(&self, tokens: &mut TokenIter) -> Result<Value> {
           // Read map: {:a 1 :b 2}
       }
       
       fn read_metadata(&self, tokens: &mut TokenIter) -> Option<BTreeMap<KeywId, Value>> {
           // Read metadata: ^:keyword or ^{:key val}
       }
   }
   ```

4. **Handle metadata and reader macros**
   - Support `^:keyword` and `^{:key val}` metadata syntax
   - Support `'`, `` ` ``, `,`, `,@` reader macros
   - Preserve spans for error reporting

### Deliverables Checklist

- [ ] Tokenizer can parse all basic tokens (parens, brackets, literals, symbols)
- [ ] Reader can parse lists: `(1 2 3)`
- [ ] Reader can parse vectors: `[1 2 3]`
- [ ] Reader can parse maps: `{:a 1 :b 2}`
- [ ] Reader can parse sets: `#{1 2 3}`
- [ ] Reader handles metadata: `^:public`, `^{:doc "..."}`
- [ ] Reader handles reader macros: `'x`, `` `x ``, `,x`, `,@xs`
- [ ] All parsed values have correct spans
- [ ] Symbols and keywords are interned
- [ ] Error messages include file/line/column information

### Testing

```rust
// tests/integration/phase1_test.rs
#[test]
fn test_read_integer() {
    let reader = Reader::new("42".to_string(), interner());
    let value = reader.read().unwrap();
    assert!(matches!(value, Value::Int { value: 42, .. }));
}

#[test]
fn test_read_list() {
    let reader = Reader::new("(1 2 3)".to_string(), interner());
    let value = reader.read().unwrap();
    // Assert list structure
}

#[test]
fn test_read_with_metadata() {
    let reader = Reader::new("^:public (def x 42)".to_string(), interner());
    let value = reader.read().unwrap();
    // Assert metadata is attached
}
```

---

## Phase 2: Basic Interpreter (Tree-Walking)

### Purpose
Build a tree-walking interpreter that can evaluate basic Evolve code. This validates language semantics before moving to compilation.

### Steps

1. **Define runtime Value type**
   ```rust
   // src/value.rs (runtime representation)
   use std::sync::Arc;
   use crate::collections::{List, Vector, Map, Set};
   
   #[derive(Debug, Clone)]
   pub enum Value {
       Nil,
       Bool(bool),
       Int(i64),
       Float(f64),
       String(Arc<str>),
       Char(char),
       Symbol(SymId),
       Keyword(KeywId),
       List(Arc<List<Value>>),
       Vector(Arc<Vector<Value>>),
       Map(Arc<Map<Value, Value>>),
       Set(Arc<Set<Value>>),
       Function {
           params: Vec<SymId>,
           body: Arc<Value>,  // CST form
           env: Arc<Env>,     // Closure environment
       },
       SpecialForm(SymId),  // Built-in special forms
   }
   ```

2. **Implement Environment**
   
   **Important**: Environments need to support:
   - Lexical scoping (parent environments)
   - Mutable bindings (for `set!` and `def`)
   - Efficient lookup
   
   **Option A: Immutable with cloning (simpler, start here)**
   ```rust
   // src/env.rs
   use std::collections::HashMap;
   use std::sync::Arc;
   
   #[derive(Debug, Clone)]
   pub struct Env {
       bindings: Arc<HashMap<SymId, Value>>,
       parent: Option<Arc<Env>>,
   }
   
   impl Env {
       pub fn new() -> Self {
           Self {
               bindings: Arc::new(HashMap::new()),
               parent: None,
           }
       }
       
       pub fn with_parent(parent: Arc<Env>) -> Self {
           Self {
               bindings: Arc::new(HashMap::new()),
               parent: Some(parent),
           }
       }
       
       pub fn get(&self, sym: SymId) -> Option<Value> {
           self.bindings.get(&sym).cloned()
               .or_else(|| self.parent.as_ref()?.get(sym))
       }
       
       pub fn set(&self, sym: SymId, value: Value) -> Self {
           // Create new environment with updated binding
           let mut new_bindings = HashMap::clone(&self.bindings);
           new_bindings.insert(sym, value);
           Self {
               bindings: Arc::new(new_bindings),
               parent: self.parent.clone(),
           }
       }
   }
   ```
   
   **Option B: Mutable with Rc<RefCell> (for `set!` support)**
   ```rust
   use std::cell::RefCell;
   use std::rc::Rc;
   
   #[derive(Debug, Clone)]
   pub struct Env {
       bindings: Rc<RefCell<HashMap<SymId, Value>>>,
       parent: Option<Rc<Env>>,
   }
   
   impl Env {
       pub fn set_mut(&self, sym: SymId, value: Value) {
           self.bindings.borrow_mut().insert(sym, value);
       }
       
       pub fn find_mut(&self, sym: SymId) -> Option<Rc<RefCell<HashMap<SymId, Value>>>> {
           if self.bindings.borrow().contains_key(&sym) {
               return Some(self.bindings.clone());
           }
           self.parent.as_ref()?.find_mut(sym)
       }
   }
   ```
   
   **Recommendation**: Start with Option A (immutable). Add Option B later when implementing `set!`.

3. **Implement basic evaluator**
   ```rust
   // src/eval.rs
   use crate::value::Value;
   use crate::env::Env;
   use crate::error::Result;
   
   pub struct Evaluator {
       interner: Arc<Interner>,
   }
   
   impl Evaluator {
       pub fn eval(&self, form: &Value, env: &mut Env) -> Result<Value> {
           match form {
               Value::Symbol(sym) => {
                   env.get(*sym)
                       .ok_or_else(|| Error::Runtime(format!("Undefined symbol")))
               }
               Value::List(list) => {
                   if list.is_empty() {
                       Ok(Value::Nil)
                   } else {
                       self.eval_list(list, env)
                   }
               }
               Value::Vector(v) => {
                   // Evaluate all elements
                   let evaluated: Vec<Value> = v.iter()
                       .map(|v| self.eval(v, env))
                       .collect::<Result<Vec<_>>>()?;
                   Ok(Value::Vector(Arc::new(Vector::from_iter(evaluated))))
               }
               // ... handle other forms
               _ => Ok(form.clone()), // Self-evaluating
           }
       }
       
       fn eval_list(&self, list: &List<Value>, env: &mut Env) -> Result<Value> {
           let first = list.head().unwrap();
           match first {
               Value::SpecialForm(sym) => {
                   self.eval_special_form(sym, &list.tail().unwrap(), env)
               }
               _ => {
                   // Function call
                   let func = self.eval(first, env)?;
                   let args: Vec<Value> = list.tail().unwrap()
                       .iter()
                       .map(|v| self.eval(v, env))
                       .collect::<Result<Vec<_>>>()?;
                   self.apply(&func, &args, env)
               }
           }
       }
   }
   ```

4. **Implement special forms**
   ```rust
   impl Evaluator {
       fn eval_special_form(&self, sym: &SymId, args: &List<Value>, env: &mut Env) -> Result<Value> {
           let sym_name = self.interner.get_symbol(*sym);
           
           match sym_name.as_str() {
               "def" => self.eval_def(args, env),
               "if" => self.eval_if(args, env),
               "let" => self.eval_let(args, env),
               "fn" | "lambda" => self.eval_fn(args, env),
               "do" => self.eval_do(args, env),
               "quote" => self.eval_quote(args, env),
               _ => Err(Error::Runtime(format!("Unknown special form: {}", sym_name))),
           }
       }
       
       fn eval_def(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
           // (def name value)
           let name = args.head().and_then(|v| match v {
               Value::Symbol(s) => Some(*s),
               _ => None,
           }).ok_or_else(|| Error::Syntax("def requires a symbol".to_string()))?;
           
           let value = args.tail()
               .and_then(|t| t.head())
               .ok_or_else(|| Error::Syntax("def requires a value".to_string()))?;
           
           let evaluated = self.eval(value, env)?;
           env.set(name, evaluated.clone());
           Ok(evaluated)
       }
       
       fn eval_if(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
           // (if condition then else?)
           let condition = args.head()
               .ok_or_else(|| Error::Syntax("if requires condition".to_string()))?;
           let then_branch = args.tail()
               .and_then(|t| t.head())
               .ok_or_else(|| Error::Syntax("if requires then branch".to_string()))?;
           let else_branch = args.tail()
               .and_then(|t| t.tail())
               .and_then(|t| t.head());
           
           let cond_value = self.eval(condition, env)?;
           if self.is_truthy(&cond_value) {
               self.eval(then_branch, env)
           } else {
               else_branch.map_or(Ok(Value::Nil), |e| self.eval(e, env))
           }
       }
       
       fn eval_let(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
           // (let [bindings...] body...)
           // Implementation
       }
       
       fn eval_fn(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
           // (fn [params...] body...)
           // Implementation
       }
       
       fn is_truthy(&self, value: &Value) -> bool {
           match value {
               Value::Nil | Value::Bool(false) => false,
               _ => true,
           }
       }
   }
   ```

5. **Implement function application**
   ```rust
   impl Evaluator {
       fn apply(&self, func: &Value, args: &[Value], env: &mut Env) -> Result<Value> {
           match func {
               Value::Function { params, body, env: closure_env } => {
                   // Create new environment with bindings
                   let mut new_env = Env::with_parent(closure_env.clone());
                   for (param, arg) in params.iter().zip(args.iter()) {
                       new_env.set(*param, arg.clone());
                   }
                   self.eval(body, &mut new_env)
               }
               _ => Err(Error::Type {
                   expected: "function".to_string(),
                   actual: format!("{:?}", func),
               }),
           }
       }
   }
   ```

### Deliverables Checklist

- [ ] Environment can store and retrieve bindings
- [ ] Environment supports lexical scoping (parent environments)
- [ ] Evaluator can evaluate self-evaluating forms (numbers, strings, etc.)
- [ ] Evaluator can evaluate symbols (variable lookup)
- [ ] `def` special form works: `(def x 42)`
- [ ] `if` special form works: `(if true 1 2)`
- [ ] `let` special form works: `(let [x 1] x)`
- [ ] `fn` special form works: `(fn [x] x)`
- [ ] Function calls work: `((fn [x] x) 42)`
- [ ] `do` special form works (sequential evaluation)
- [ ] `quote` special form works: `(quote x)`
- [ ] Error messages are clear and helpful

### Testing

```rust
// tests/integration/phase2_test.rs
use crate::*;

fn setup() -> (Evaluator, Env) {
    let interner = Arc::new(Interner::new());
    let eval = Evaluator::new(interner.clone());
    let env = Env::new();
    (eval, env)
}

fn read(s: &str) -> Value {
    let interner = Arc::new(Interner::new());
    let reader = Reader::new(s.to_string(), interner);
    reader.read().unwrap()
}

fn sym(s: &str) -> SymId {
    let interner = Arc::new(Interner::new());
    interner.intern_symbol(s)
}

#[test]
fn test_eval_integer() {
    let (eval, mut env) = setup();
    let value = Value::Int(42);
    assert_eq!(eval.eval(&value, &mut env).unwrap(), Value::Int(42));
}

#[test]
fn test_eval_def() {
    let (eval, mut env) = setup();
    let form = read("(def x 42)");
    let new_env = eval.eval(&form, &mut env).unwrap();
    // With immutable env, def returns new env
    // Or store result and check
}

#[test]
fn test_eval_function_call() {
    let (eval, mut env) = setup();
    let form = read("((fn [x] x) 42)");
    let result = eval.eval(&form, &mut env).unwrap();
    assert_eq!(result, Value::Int(42));
}

#[test]
fn test_eval_if() {
    let (eval, mut env) = setup();
    let form = read("(if true 1 2)");
    let result = eval.eval(&form, &mut env).unwrap();
    assert_eq!(result, Value::Int(1));
}

#[test]
fn test_eval_let() {
    let (eval, mut env) = setup();
    let form = read("(let [x 1 y 2] (+ x y))");
    // Need + function first, or test with simpler form
    let form = read("(let [x 1] x)");
    let result = eval.eval(&form, &mut env).unwrap();
    assert_eq!(result, Value::Int(1));
}
```

---

## Phase 3: Collections & Basic Operations

### Purpose
Implement persistent collections (list, vector, map, set) and their basic operations.

### Steps

1. **Implement persistent List**
   ```rust
   // src/collections/list.rs
   use std::rc::Rc;
   
   #[derive(Clone)]
   pub struct List<T> {
       head: Option<Rc<Node<T>>>,
       len: usize,
   }
   
   struct Node<T> {
       elem: T,
       next: Option<Rc<Node<T>>>,
   }
   
   impl<T> List<T> {
       pub fn new() -> Self {
           Self { head: None, len: 0 }
       }
       
       pub fn prepend(&self, value: T) -> Self {
           let new_node = Rc::new(Node {
               elem: value,
               next: self.head.clone(),
           });
           Self {
               head: Some(new_node),
               len: self.len + 1,
           }
       }
       
       pub fn head(&self) -> Option<&T> {
           self.head.as_deref().map(|n| &n.elem)
       }
       
       pub fn tail(&self) -> Option<Self> {
           let node = self.head.as_ref()?;
           Some(Self {
               head: node.next.clone(),
               len: self.len - 1,
           })
       }
   }
   ```

2. **Implement persistent Vector (RRB-Tree)**
   - Use existing implementation from `src/collections/vector.rs`
   - Ensure it supports: `get`, `update`, `push_back`, `pop_back`

3. **Implement persistent Map (HAMT)**
   - Use existing implementation from `src/collections/map.rs`
   - Ensure it supports: `get`, `insert`, `remove`, iteration

4. **Implement persistent Set**
   - Use existing implementation from `src/collections/set.rs`
   - Ensure it supports: `contains`, `insert`, `remove`

5. **Add collection operations to evaluator**
   ```rust
   // Add built-in functions
   fn eval_builtin(&self, name: &str, args: &[Value]) -> Result<Value> {
       match name {
           "conj" => {
               // (conj coll x)
               // Implementation
           }
           "assoc" => {
               // (assoc map k v)
               // Implementation
           }
           "get" => {
               // (get coll key)
               // Implementation
           }
           "count" => {
               // (count coll)
               // Implementation
           }
           _ => Err(Error::Runtime(format!("Unknown function: {}", name))),
       }
   }
   ```

### Deliverables Checklist

- [ ] Persistent List implemented with structural sharing
- [ ] Persistent Vector implemented (RRB-Tree)
- [ ] Persistent Map implemented (HAMT with insertion order)
- [ ] Persistent Set implemented
- [ ] `conj` function works for all collections
- [ ] `assoc` function works for maps
- [ ] `get` function works for maps and vectors
- [ ] `count` function works for all collections
- [ ] Collections can be nested and shared
- [ ] Memory usage is reasonable (structural sharing works)

### Testing

```rust
#[test]
fn test_list_operations() {
    let list = List::new();
    let list2 = list.prepend(1).prepend(2);
    assert_eq!(list2.head(), Some(&2));
    assert_eq!(list2.tail().unwrap().head(), Some(&1));
}

#[test]
fn test_vector_operations() {
    let vec = Vector::new();
    let vec2 = vec.push_back(1).push_back(2);
    assert_eq!(vec2.get(0), Some(&1));
    assert_eq!(vec2.get(1), Some(&2));
}
```

---

## Phase 4: Macros & Quasiquote

### Purpose
Implement macro system with quasiquote/unquote support for code generation.

### Steps

1. **Implement `defmacro`**
   ```rust
   fn eval_defmacro(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
       // (defmacro name [params...] body...)
       // Similar to defn but stores as macro
   }
   ```

2. **Implement macro expansion**
   ```rust
   pub struct Expander {
       interner: Arc<Interner>,
   }
   
   impl Expander {
       pub fn expand(&self, form: &Value, env: &Env) -> Result<Value> {
           // Check if form is a macro call
           // If yes, expand it, otherwise return as-is
       }
       
       fn is_macro_call(&self, form: &Value, env: &Env) -> bool {
           // Check if first element is a macro
       }
       
       fn macroexpand_1(&self, form: &Value, env: &Env) -> Result<Value> {
           // Single pass of macro expansion
       }
       
       fn macroexpand(&self, form: &Value, env: &Env) -> Result<Value> {
           // Fully expand macros
           let mut current = form.clone();
           loop {
               let expanded = self.macroexpand_1(&current, env)?;
               if expanded == current {
                   return Ok(expanded);
               }
               current = expanded;
           }
       }
   }
   ```

3. **Implement quasiquote/unquote**
   ```rust
   fn eval_quasiquote(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
       // Handle ` (quasiquote)
       let form = args.head()
           .ok_or_else(|| Error::Syntax("quasiquote requires form".to_string()))?;
       self.quasiquote_expand(form, env)
   }
   
   fn quasiquote_expand(&self, form: &Value, env: &mut Env) -> Result<Value> {
       match form {
           Value::List(list) => {
               // Check for unquote/unquote-splice
               if let Some(Value::Symbol(sym)) = list.head() {
                   let sym_name = self.interner.get_symbol(*sym);
                   if sym_name == "unquote" {
                       // ,x - evaluate and return
                       return self.eval(list.tail().unwrap().head().unwrap(), env);
                   }
                   if sym_name == "unquote-splice" {
                       // ,@xs - evaluate and splice
                       let evaluated = self.eval(list.tail().unwrap().head().unwrap(), env)?;
                       // Return as list to splice
                   }
               }
               // Recursively quasiquote all elements
               // ...
           }
           _ => Ok(form.clone()),
       }
   }
   ```

4. **Integrate macro expansion into evaluator**
   ```rust
   impl Evaluator {
       pub fn eval(&self, form: &Value, env: &mut Env) -> Result<Value> {
           // First expand macros
           let expanded = self.expander.expand(form, env)?;
           // Then evaluate
           self.eval_expanded(&expanded, env)
       }
   }
   ```

### Deliverables Checklist

- [ ] `defmacro` special form works
- [ ] Macros can be defined and called
- [ ] `macroexpand-1` function works
- [ ] `macroexpand` function works (full expansion)
- [ ] Quasiquote `` ` `` works
- [ ] Unquote `~` works
- [ ] Unquote-splice `~@` works
- [ ] Macros can generate code with quasiquote
- [ ] Macro expansion happens before evaluation
- [ ] Nested macros work correctly

### Testing

```rust
#[test]
fn test_simple_macro() {
    let eval = Evaluator::new(interner());
    let mut env = Env::new();
    // (defmacro inc [x] `(+ ,x 1))
    // (inc 41) => 42
}

#[test]
fn test_quasiquote() {
    let eval = Evaluator::new(interner());
    let mut env = Env::new();
    // `(list ,x ,@xs) => (list x xs...)
}
```

---

## Phase 5: Modules & Namespaces

### Purpose
Implement namespace system and module loading.

### Steps

1. **Implement namespace structure**
   ```rust
   // src/ns.rs
   pub struct Namespace {
       name: NsId,
       vars: HashMap<SymId, Value>,
       macros: HashMap<SymId, Value>,  // Macro definitions
       public: HashSet<SymId>,  // Public exports
   }
   
   pub struct NamespaceManager {
       namespaces: HashMap<NsId, Namespace>,
       current: NsId,
   }
   ```

2. **Implement `ns` special form**
   ```rust
   fn eval_ns(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
       // (ns my.ns)
       // (ns my.ns :require [foo.bar :as fb])
       // Parse namespace name and requires
   }
   ```

3. **Implement `:require`**
   ```rust
   fn process_require(&self, require_spec: &Value, env: &mut Env) -> Result<()> {
       // Parse :require specification
       // Load module if needed
       // Add bindings to current namespace
   }
   ```

4. **Implement module loading**
   ```rust
   pub struct ModuleLoader {
       search_paths: Vec<PathBuf>,
   }
   
   impl ModuleLoader {
       pub fn load(&self, ns_name: &str) -> Result<Value> {
           // Find source file
           // Read and parse
           // Evaluate in namespace context
       }
       
       fn find_module(&self, ns_name: &str) -> Option<PathBuf> {
           // Search in paths for ns_name.evolve
       }
   }
   ```

5. **Implement `^:public` metadata handling**
   ```rust
   fn eval_def(&self, args: &List<Value>, env: &mut Env) -> Result<Value> {
       // Check for metadata
       // If ^:public, mark as exported
   }
   ```

### Deliverables Checklist

- [ ] `ns` special form works: `(ns my.ns)`
- [ ] Namespaces can be created and switched
- [ ] `:require` works: `(:require [foo.bar :as fb])`
- [ ] `:as` aliasing works
- [ ] `:refer` selective import works
- [ ] `:rename` works
- [ ] `^:public` metadata marks exports
- [ ] Modules can be loaded from files
- [ ] Circular dependencies are handled (or error clearly)
- [ ] Private by default, public only with `^:public`

### Testing

```rust
#[test]
fn test_namespace() {
    // Create namespace
    // Define vars
    // Switch namespace
    // Access vars
}

#[test]
fn test_module_loading() {
    // Create module file
    // Load it
    // Use its exports
}
```

---

## Phase 6: Compiler - HIR Generation

### Purpose
Transform CST into High-Level IR (HIR) suitable for code generation. HIR is a simplified, compiler-friendly representation that:
- Removes metadata (kept separately if needed)
- Normalizes syntax (e.g., `defn` becomes `def` + `fn`)
- Identifies tail positions for TCO
- Prepares for code generation

**Note**: At this point, you have a working interpreter. The compiler adds a new code path but the interpreter remains available (useful for REPL, macro expansion, etc.).

### Steps

1. **Define HIR structure**
   ```rust
   // src/hir.rs
   pub enum HIR {
       Def {
           name: SymId,
           value: Box<HIR>,
           meta: Option<Metadata>,
       },
       If {
           cond: Box<HIR>,
           then_: Box<HIR>,
           else_: Option<Box<HIR>>,
       },
       Let {
           bindings: Vec<(Pattern, HIR)>,
           body: Vec<HIR>,
       },
       Do {
           forms: Vec<HIR>,
       },
       Fn {
           params: Vec<Pattern>,
           body: Vec<HIR>,
           name: Option<SymId>,
       },
       Call {
           callee: Box<HIR>,
           args: Vec<HIR>,
       },
       Var {
           id: SymId,
       },
       Literal {
           value: Value,
       },
       Loop {
           bindings: Vec<(Pattern, HIR)>,
           body: Vec<HIR>,
       },
       Recur {
           args: Vec<HIR>,
       },
   }
   ```

2. **Implement CST to HIR lowering**
   ```rust
   pub struct Lowerer {
       interner: Arc<Interner>,
   }
   
   impl Lowerer {
       pub fn lower(&self, cst: &Value, env: &Env) -> Result<HIR> {
           match cst {
               Value::List(list) => self.lower_list(list, env),
               Value::Symbol(sym) => Ok(HIR::Var { id: *sym }),
               _ => Ok(HIR::Literal { value: cst.clone() }),
           }
       }
       
       fn lower_list(&self, list: &List<Value>, env: &Env) -> Result<HIR> {
           let first = list.head().ok_or_else(|| Error::Syntax("Empty list".to_string()))?;
           match first {
               Value::Symbol(sym) => {
                   let name = self.interner.get_symbol(*sym);
                   match name.as_str() {
                       "def" => self.lower_def(list, env),
                       "if" => self.lower_if(list, env),
                       "fn" => self.lower_fn(list, env),
                       _ => self.lower_call(list, env),
                   }
               }
               _ => self.lower_call(list, env),
           }
       }
   }
   ```

3. **Handle tail call optimization markers**
   ```rust
   fn lower_fn(&self, list: &List<Value>, env: &Env) -> Result<HIR> {
       // Convert (fn [params...] body...) to HIR::Fn
       // Mark tail positions for TCO
   }
   
   fn is_tail_position(&self, form: &HIR) -> bool {
       // Determine if form is in tail position
   }
   ```

4. **Implement pattern matching for destructuring**
   ```rust
   pub enum Pattern {
       Bind(SymId),
       Vector(Vec<Pattern>),
       Map(Vec<(Pattern, Pattern)>),
       Ignore,
   }
   ```

### Deliverables Checklist

- [ ] CST can be lowered to HIR
- [ ] All special forms have HIR representation
- [ ] Function calls are represented in HIR
- [ ] Tail positions are identified
- [ ] Destructuring patterns are supported
- [ ] HIR preserves semantics of CST
- [ ] HIR is simpler than CST (no metadata, spans minimal)

### Testing

```rust
#[test]
fn test_lower_simple_form() {
    let interner = Arc::new(Interner::new());
    let lowerer = Lowerer::new(interner.clone());
    let reader = Reader::new("(def x 42)".to_string(), interner);
    let cst = reader.read().unwrap();
    let env = Env::new();
    let hir = lowerer.lower(&cst, &env).unwrap();
    
    match hir {
        HIR::Def { name, value, .. } => {
            assert_eq!(name, interner.intern_symbol("x"));
            match *value {
                HIR::Literal { value: Value::Int(42) } => {},
                _ => panic!("Expected integer literal"),
            }
        }
        _ => panic!("Expected Def HIR"),
    }
}

#[test]
fn test_lower_function() {
    let interner = Arc::new(Interner::new());
    let lowerer = Lowerer::new(interner.clone());
    let reader = Reader::new("(fn [x] x)".to_string(), interner);
    let cst = reader.read().unwrap();
    let env = Env::new();
    let hir = lowerer.lower(&cst, &env).unwrap();
    
    match hir {
        HIR::Fn { params, body, .. } => {
            assert_eq!(params.len(), 1);
            assert_eq!(body.len(), 1);
        }
        _ => panic!("Expected Fn HIR"),
    }
}
```

---

## Phase 7: LLVM Code Generation

### Purpose
Generate LLVM IR from HIR and compile to native code.

### Steps

1. **Set up LLVM bindings**
   ```toml
   # Cargo.toml
   [dependencies]
   inkwell = "0.2"  # LLVM bindings for Rust
   # OR
   llvm-sys = "180"  # Lower-level bindings
   ```

2. **Implement code generator**
   ```rust
   // src/codegen.rs
   use inkwell::context::Context;
   use inkwell::module::Module;
   use inkwell::values::FunctionValue;
   
   pub struct CodeGen<'ctx> {
       context: &'ctx Context,
       module: Module<'ctx>,
       builder: Builder<'ctx>,
   }
   
   impl<'ctx> CodeGen<'ctx> {
       pub fn new(context: &'ctx Context) -> Self {
           let module = context.create_module("evolve");
           let builder = context.create_builder();
           Self {
               context,
               module,
               builder,
           }
       }
       
       pub fn compile(&mut self, hir: &HIR) -> Result<()> {
           match hir {
               HIR::Def { name, value, .. } => {
                   // Compile definition
               }
               HIR::Fn { params, body, name, .. } => {
                   self.compile_function(name, params, body)
               }
               // ...
           }
       }
       
       fn compile_function(&mut self, name: Option<SymId>, params: &[Pattern], body: &[HIR]) -> Result<FunctionValue> {
           // Create LLVM function
           // Compile body
           // Return function value
       }
   }
   ```

3. **Implement value representation in LLVM**
   ```rust
   // Boxed, tagged pointers
   // Use struct { tag: i64, data: [8 x i8] } or similar
   fn compile_value(&mut self, hir: &HIR) -> Result<Value> {
       match hir {
           HIR::Literal { value } => {
               match value {
                   Value::Int(i) => {
                       // Create boxed integer
                   }
                   // ...
               }
           }
           _ => todo!(),
       }
   }
   ```

4. **Implement tail call optimization**
   ```rust
   fn compile_tail_call(&mut self, call: &HIR::Call) -> Result<Value> {
       // Use LLVM tail call instruction
       // self.builder.build_tail_call(...)
   }
   ```

5. **Link and generate object files**
   ```rust
   pub fn emit_object(&self, path: &Path) -> Result<()> {
       // Compile module to object file
   }
   ```

### Deliverables Checklist

- [ ] HIR can be compiled to LLVM IR
- [ ] Functions are generated correctly
- [ ] Basic operations (arithmetic, comparisons) work
- [ ] Function calls work
- [ ] Tail calls are optimized
- [ ] Object files can be generated
- [ ] Generated code can be linked and executed
- [ ] Error handling works (exceptions map to LLVM EH)

### Testing

```rust
#[test]
fn test_codegen_simple_function() {
    let context = Context::create();
    let mut codegen = CodeGen::new(&context);
    let hir = lower("(fn [x] x)");
    codegen.compile(&hir).unwrap();
    // Verify LLVM IR
}
```

---

## Phase 8: Advanced Features

### Purpose
Add STM, atoms, agents, condition system, and other advanced features.

### Steps

1. **Implement Atoms**
   ```rust
   // src/atom.rs
   use std::sync::Arc;
   use std::sync::atomic::{AtomicPtr, Ordering};
   
   pub struct Atom {
       value: Arc<AtomicPtr<Value>>,
   }
   
   impl Atom {
       pub fn new(value: Value) -> Self {
           Self {
               value: Arc::new(AtomicPtr::new(Box::into_raw(Box::new(value)))),
           }
       }
       
       pub fn swap<F>(&self, f: F) -> Value
       where
           F: FnOnce(&Value) -> Value,
       {
           // CAS loop
       }
   }
   ```

2. **Implement STM (Software Transactional Memory)**
   ```rust
   // src/stm.rs
   pub struct Ref {
       value: Arc<RwLock<Value>>,
       version: Arc<AtomicU64>,
   }
   
   pub fn dosync<F>(f: F) -> Result<Value>
   where
       F: FnOnce() -> Result<Value>,
   {
       // Transaction with retry on conflict
   }
   ```

3. **Implement Agents**
   ```rust
   // src/agent.rs
   use std::sync::mpsc;
   
   pub struct Agent {
       state: Arc<Mutex<Value>>,
       actions: mpsc::Sender<Action>,
   }
   
   impl Agent {
       pub fn send<F>(&self, f: F) -> Result<()>
       where
           F: FnOnce(&Value) -> Value + Send + 'static,
       {
           // Send action to agent thread
       }
   }
   ```

4. **Implement Condition System**
   ```rust
   // src/condition.rs
   pub fn signal(condition: &Value, data: &[Value]) -> Result<()> {
       // Signal condition
   }
   
   pub fn handler_bind<F>(handlers: &Map<Value, Value>, body: F) -> Result<Value>
   where
       F: FnOnce() -> Result<Value>,
   {
       // Bind condition handlers
   }
   ```

### Deliverables Checklist

- [ ] Atoms work: `(atom x)`, `(swap! a f)`, `(reset! a v)`, `(deref a)`
- [ ] STM refs work: `(ref x)`, `(dosync ...)`, `(alter r f)`
- [ ] Agents work: `(agent x)`, `(send a f)`, `(await a)`
- [ ] Condition system works: `(signal ...)`, `(handler-bind ...)`, `(restart-case ...)`
- [ ] `try`/`catch`/`finally` work
- [ ] All features integrate with compiler

---

## Phase 9: Optimization & Performance

### Purpose
Add optimizations to improve performance of generated code.

### Steps

1. **Implement TCO enforcement**
   - Ensure all tail calls are optimized
   - Verify in generated LLVM IR

2. **Implement escape analysis**
   - Identify values that don't escape
   - Allocate on stack when possible

3. **Implement inlining**
   - Inline small functions
   - Use LLVM inliner

4. **Add optimization passes**
   ```rust
   // Use LLVM optimization passes
   module.run_on_function(pass_manager, function);
   ```

5. **Profile and optimize hot paths**
   - Add profiling hooks
   - Identify bottlenecks
   - Optimize incrementally

### Deliverables Checklist

- [ ] TCO works for all tail calls
- [ ] Escape analysis reduces allocations
- [ ] Inlining improves performance
- [ ] Optimization passes are applied
- [ ] Performance is acceptable for MVP

---

## Phase 10: Tooling & Polish

### Purpose
Add CLI, REPL, formatter, and other developer tools.

### Steps

1. **Implement CLI**
   ```rust
   // src/main.rs
   use clap::Parser;
   
   #[derive(Parser)]
   struct Cli {
       #[command(subcommand)]
       command: Command,
   }
   
   #[derive(Subcommand)]
   enum Command {
       Build { file: PathBuf },
       Run { file: PathBuf },
       Repl,
       Test,
   }
   ```

2. **Enhance REPL**
   - Better error messages
   - History
   - Auto-completion (future)

3. **Add formatter**
   - Format Evolve code
   - Consistent style

4. **Add test runner**
   - Run tests marked with `:test` metadata
   - Report results

### Deliverables Checklist

- [ ] CLI works: `evolve build`, `evolve run`, `evolve repl`, `evolve test`
- [ ] REPL is usable
- [ ] Formatter works
- [ ] Test runner works
- [ ] Documentation is complete

---

## Decision Points & Options

### Parser Approach
- **Option A: `logos`** - Fast, generated lexer. Good for performance.
- **Option B: `nom`** - Parser combinators. More flexible, functional style.
- **Recommendation**: Start with `logos` for speed, can switch if needed.

### Value Representation
- **Option A: `Arc<Value>` everywhere** - Simple, safe, but may be slower.
- **Option B: Custom RC with optimizations** - More complex, but faster.
- **Recommendation**: Start with `Arc`, optimize later (Phase 9).

### Macro Expansion
- **Option A: Expand before lowering to HIR** - Simpler, but less optimization opportunity.
- **Option B: Expand during HIR generation** - More complex, but better for optimization.
- **Recommendation**: Start with Option A, can refactor later.

### Testing Strategy
- **Unit tests** for each component
- **Integration tests** for end-to-end scenarios
- **Property-based tests** for collections (using `proptest` or `quickcheck`)
- **Performance benchmarks** for hot paths

---

## Next Steps After MVP

1. **Runtime Library** - Implement the evolve_* runtime functions in C or Rust
2. **JIT Compilation** - Add ORC LLJIT for runtime compilation
3. **Quote Compilation** - Serialize quoted values to embedded data
4. **Advanced Optimizations** - ThinLTO, PGO, custom passes
5. **FFI** - C interop with stable ABI
6. **Package Manager** - Dependency management
7. **LSP Server** - Language server protocol
8. **Gradual Typing** - Optional type system
9. **Concurrent GC** - Advanced memory management

---

## Notes

- **Start simple**: Each phase should produce working code
- **Test early**: Write tests as you implement
- **Refactor freely**: Early phases can be simplified, optimize later
- **Document decisions**: Keep notes on why choices were made
- **Iterate**: Don't be afraid to go back and improve earlier phases

---

## Summary: Quick Reference

### Phase Progression

| Phase | Focus | Deliverable | Can Run? |
|-------|-------|-------------|----------|
| 0 | Setup | Project structure | No |
| 1 | Reader | Parse code to CST | No |
| 2 | Interpreter | Evaluate basic code | **Yes** (simple programs) |
| 3 | Collections | Use data structures | **Yes** (real programs) |
| 4 | Macros | Code generation | **Yes** (powerful programs) |
| 5 | Modules | Organize code | **Yes** (multi-file) |
| 6 | HIR | Compiler IR | No (prep for compilation) |
| 7 | LLVM | Native code | **Yes** (fast programs) |
| 8 | Advanced | STM, atoms, etc. | **Yes** (concurrent) |
| 9 | Optimize | Performance | **Yes** (faster) |
| 10 | Tooling | Developer experience | **Yes** (polished) |

### Key Milestones

1. **Milestone 1 (Phase 2)**: Can evaluate `(def x 42)` and `((fn [x] x) 42)`
2. **Milestone 2 (Phase 3)**: Can use collections: `(conj [1 2] 3)`
3. **Milestone 3 (Phase 4)**: Can write macros: `(defmacro inc [x] \`(+ ,x 1))`
4. **Milestone 4 (Phase 5)**: Can organize code in modules
5. **Milestone 5 (Phase 7)**: Can compile to native code
6. **Milestone 6 (Phase 10)**: Complete toolchain with CLI, REPL, formatter

### Testing Strategy

- **Unit tests**: Test each component in isolation
- **Integration tests**: Test end-to-end scenarios
- **Example programs**: Keep a suite of example Evolve programs that should work
- **Regression tests**: When fixing bugs, add tests to prevent regressions

### Common Pitfalls to Avoid

1. **Over-engineering early**: Start simple, optimize later
2. **Skipping tests**: Write tests as you go, not at the end
3. **Premature optimization**: Get it working first, then make it fast
4. **Ignoring error messages**: Good error messages are crucial for developer experience
5. **Forgetting documentation**: Document APIs and design decisions

### Getting Help

- Review existing code in `src/` for reference implementations
- Check `docs/LANGUAGE.md` for language semantics
- Check `docs/IMPLEMENTATION.md` for implementation details
- Test incrementally: after each feature, write a test program that uses it

