## Runtime Implementation for Code Generation

This section specifies the implementation of the runtime library that supports compiled Evolve code. The runtime provides all functions declared in the code generator and manages memory, collections, function dispatch, and global state.

### Architecture Overview

The runtime is implemented as a C-compatible library (`libevolve_runtime`) that can be linked with compiled Evolve code. It provides:

1. **Memory Management**: ARC-based heap allocation and reference counting
2. **Value Representation**: Boxing/unboxing of tagged pointer values
3. **Collection Runtime**: Support for strings, vectors, maps, sets, and lists
4. **Function Dispatch**: Closure execution and native function calls
5. **Global State**: Variable and namespace management
6. **Error Handling**: Panic and error reporting

### Value Representation

All Evolve values are represented as 64-bit tagged pointers (see `src/codegen/value.rs`):

* **Immediate values** (nil, bool, int, char, keyword, symbol): Encoded directly in the 64-bit value
* **Boxed values** (strings, collections, closures, floats): Pointer to heap-allocated object with tag `0b000`

#### Boxed Object Header

All heap-allocated objects share a common header:

```c
struct BoxedHeader {
    uint8_t type_tag;      // BoxedType enum value
    uint32_t ref_count;    // Reference count (atomic)
    // Type-specific data follows...
};
```

Type tags:
- `0`: String
- `1`: List
- `2`: Vector
- `3`: Map
- `4`: Set
- `5`: Closure
- `6`: Var
- `7`: Namespace
- `8`: Float (boxed)

### Memory Management

#### Allocation (`evolve_alloc`)

```c
void* evolve_alloc(int64_t size);
```

* Allocates `size` bytes on the heap
* Returns pointer aligned to 8 bytes
* Initializes ref_count to 1
* Uses system allocator (malloc) in v0; can be replaced with custom allocator later

#### Reference Counting (`evolve_retain`, `evolve_release`)

```c
void evolve_retain(int64_t value);
void evolve_release(int64_t value);
```

* `evolve_retain`: Increments ref_count atomically for boxed values (no-op for immediates)
* `evolve_release`: Decrements ref_count atomically; frees object if count reaches 0
* Must handle immediate values (check tag before dereferencing pointer)
* Thread-safe using atomic operations

**Implementation Notes:**
- Extract tag: `tag = value & 0b111`
- If tag == 0 (boxed), extract pointer: `ptr = value & ~0b111`
- Access header: `header = (BoxedHeader*)(ptr - offsetof(BoxedHeader, type_tag))`
- Use atomic increment/decrement for ref_count

### String Runtime

#### String Creation (`evolve_string_new`)

```c
int64_t evolve_string_new(const char* data, int64_t len);
```

**Structure:**
```c
struct StringObject {
    BoxedHeader header;    // type_tag = 0
    int64_t length;
    char data[];          // Flexible array member
};
```

**Implementation:**
1. Allocate: `sizeof(BoxedHeader) + sizeof(int64_t) + len + 1` bytes
2. Initialize header (type_tag=0, ref_count=1)
3. Store length
4. Copy `len` bytes from `data` to `data[]`
5. Null-terminate for C compatibility
6. Return tagged pointer: `(uintptr_t)obj | 0b000`

### Collection Runtime

#### Vector Operations

**Vector Creation (`evolve_vector_new`):**
```c
int64_t evolve_vector_new(int32_t count, const int64_t* items);
```

**Structure:**
```c
struct VectorObject {
    BoxedHeader header;    // type_tag = 2
    int64_t length;
    int64_t capacity;
    int64_t items[];      // Flexible array
};
```

**Implementation:**
1. Allocate space for header + length + capacity + items array
2. Initialize header (type_tag=2, ref_count=1)
3. Store count as length
4. Copy `count` values from `items` array
5. Retain each item (increment ref_count)
6. Return tagged pointer

**Vector Get (`evolve_vector_get`):**
```c
int64_t evolve_vector_get(int64_t vec, int64_t index);
```

1. Extract pointer and verify type_tag == 2
2. Bounds check: `index >= 0 && index < length`
3. Return `items[index]` (caller must retain if storing)
4. Panic on bounds error

**Vector Count (`evolve_vector_count`):**
```c
int64_t evolve_vector_count(int64_t vec);
```

Returns the length field from VectorObject.

**Vector Rest (`evolve_vector_rest`):**
```c
int64_t evolve_vector_rest(int64_t vec, int64_t start);
```

Creates a new vector containing elements from `start` to end. Retains all elements in the new vector.

#### Map Operations

**Map Creation (`evolve_map_new`):**
```c
int64_t evolve_map_new(int32_t count, const int64_t* entries);
```

**Structure:**
```c
struct MapObject {
    BoxedHeader header;    // type_tag = 3
    int64_t size;
    // HAMT structure follows (implementation detail)
    // For v0, use a simple hash table or array of key-value pairs
};
```

**Implementation:**
- `entries` is interleaved: `[key0, val0, key1, val1, ...]`
- Create map structure (HAMT or hash table)
- Retain all keys and values
- Return tagged pointer

**Map Get (`evolve_map_get`):**
```c
int64_t evolve_map_get(int64_t map, int64_t key);
```

1. Extract pointer and verify type_tag == 3
2. Lookup key in map structure
3. Return value (or nil if not found)
4. Caller must retain returned value if storing

#### Set Operations

**Set Creation (`evolve_set_new`):**
```c
int64_t evolve_set_new(int32_t count, const int64_t* items);
```

Similar to vector creation but uses set data structure (can be implemented as map with nil values).

#### List Operations

**List Creation (`evolve_list_new`):**
```c
int64_t evolve_list_new(int32_t count, const int64_t* items);
```

**Structure:**
```c
struct ListNode {
    int64_t value;
    struct ListNode* next;
    uint32_t ref_count;
};

struct ListObject {
    BoxedHeader header;    // type_tag = 1
    int64_t length;
    ListNode* head;
};
```

**Implementation:**
- Build linked list of nodes
- Retain each value
- Return tagged pointer

### Closure Runtime

#### Closure Creation (`evolve_closure_new`)

```c
int64_t evolve_closure_new(void* fn_ptr, int32_t env_count, const int64_t* env);
```

**Structure:**
```c
struct ClosureObject {
    BoxedHeader header;    // type_tag = 5
    void* function_ptr;   // Pointer to compiled function
    int32_t env_count;
    int64_t env[];        // Flexible array of captured values
};
```

**Implementation:**
1. Allocate: `sizeof(BoxedHeader) + sizeof(void*) + sizeof(int32_t) + env_count * sizeof(int64_t)`
2. Initialize header (type_tag=5, ref_count=1)
3. Store function pointer
4. Store env_count
5. Copy and retain each captured value from `env` array
6. Return tagged pointer

**Closure Execution:**
Closures are executed via `evolve_call` (see Function Dispatch below).

### Function Dispatch

#### Function Call (`evolve_call`)

```c
int64_t evolve_call(int64_t fn, int32_t argc, const int64_t* argv);
```

**Dispatch Logic:**
1. Extract tag from `fn`
2. If tag == 5 (Closure):
   - Extract ClosureObject pointer
   - Load function_ptr and env
   - Call function with signature: `(env_ptr, argc, argv) -> int64_t`
   - Function signature matches codegen convention
3. If tag == 6 (Var):
   - Lookup var value
   - If value is a function, recurse with `evolve_call`
   - Otherwise, error
4. If tag == 0 (Native function boxed):
   - Extract native function pointer
   - Call via `evolve_native_call`
5. Otherwise, panic with "not a function" error

**Calling Convention:**
- Compiled functions expect: `(env_ptr: void*, argc: i32, argv: int64_t*) -> int64_t`
- `env_ptr` points to closure environment (or NULL if no captures)
- `argv` is array of `argc` tagged values
- Return value is tagged

### Global Variable Management

#### Variable Lookup (`evolve_var_get`)

```c
int64_t evolve_var_get(int64_t sym_id);
```

**Implementation:**
1. Lookup symbol in current namespace (via global namespace registry)
2. Get Var from namespace bindings
3. Read Var's value (thread-safe via RwLock)
4. Convert Value to tagged representation
5. Retain if boxed
6. Return tagged value

**Value to Tagged Conversion:**
- Immediate values: encode directly
- Boxed values: get pointer, add to global heap registry, return tagged pointer
- Must handle all Value enum variants

#### Variable Definition (`evolve_var_def`)

```c
int64_t evolve_var_def(int64_t sym_id, int64_t value);
```

**Implementation:**
1. Lookup or create Var in current namespace
2. Convert tagged value to Value representation
3. Retain value (increment ref_count if boxed)
4. Store in Var (thread-safe via RwLock)
5. Update namespace registry
6. Return value (retained)

**Tagged to Value Conversion:**
- Extract tag
- For immediates: construct appropriate Value variant
- For boxed: lookup in heap registry, construct Value::* variant
- Must handle all tag types

### Namespace Management

#### Namespace Switch (`evolve_ns_switch`)

```c
int64_t evolve_ns_switch(int64_t sym_id);
```

**Implementation:**
1. Lookup namespace by name (from sym_id)
2. Create namespace if it doesn't exist
3. Set as current namespace in global registry
4. Return namespace value (tagged)

**Namespace Value:**
- Namespaces are boxed objects (type_tag=7)
- Structure contains namespace ID and metadata
- Used for namespace introspection

### Native Function Support

#### Native Function Call (`evolve_native_call`)

```c
int64_t evolve_native_call(int64_t sym_id, int32_t argc, const int64_t* argv);
```

**Implementation:**
1. Lookup native function by sym_id in global native function registry
2. Call native function with argc and argv
3. Native functions have signature: `(int32_t argc, const int64_t* argv) -> int64_t`
4. Return tagged result

**Native Function Registry:**
- Global hash map: `sym_id -> NativeFn`
- Populated at runtime initialization
- Includes all core functions (arithmetic, comparison, collection ops, etc.)

### Truthiness Check

#### Is Truthy (`evolve_is_truthy`)

```c
bool evolve_is_truthy(int64_t value);
```

**Implementation:**
1. Extract tag
2. If tag == 2 (Nil): return false
3. If tag == 3 (Bool): extract bool value, return it
4. Otherwise: return true

### Quote Support

#### Quote (`evolve_quote`)

```c
int64_t evolve_quote(const void* data);
```

**Implementation:**
- `data` points to serialized Value data
- Deserialize and create boxed quote object
- Return tagged pointer
- **Note:** Quote serialization format TBD (can use EDN or custom binary format)

### Error Handling

#### Panic (`evolve_panic`)

```c
void evolve_panic(const char* msg);
```

**Implementation:**
1. Print error message to stderr
2. Print stack trace (if available)
3. Call `abort()` to terminate process
4. **Future:** Can be replaced with condition system integration

### Runtime Initialization

The runtime must be initialized before any compiled code executes:

```c
void evolve_runtime_init(void);
```

**Initialization Steps:**
1. Initialize memory allocator
2. Create default namespace ("user")
3. Register all native functions
4. Initialize global variable registry
5. Initialize namespace registry
6. Set up error handlers

### Thread Safety

All runtime functions must be thread-safe:

* **Reference counting**: Use atomic operations
* **Global state**: Protect with mutexes or use lock-free data structures
* **Namespace registry**: Thread-safe access required
* **Var access**: Already thread-safe via RwLock in interpreter; runtime must maintain this

### Integration with Interpreter

The runtime should share data structures with the interpreter where possible:

* **Namespace Registry**: Can reuse `NamespaceRegistry` from interpreter
* **Native Functions**: Can reuse `NativeRegistry` from interpreter
* **Value Representation**: Runtime uses tagged pointers; interpreter uses `Value` enum
* **Conversion Layer**: Implement bidirectional conversion between tagged pointers and `Value`

### Implementation Phases

**Phase 1: Core Runtime (MVP)**
1. Memory management (alloc, retain, release)
2. Value boxing/unboxing
3. String operations
4. Basic vector operations
5. Function call dispatch
6. Global variable lookup/definition
7. Truthiness check
8. Panic

**Phase 2: Complete Collections**
1. Full vector operations (get, count, rest)
2. Map operations (new, get)
3. Set operations
4. List operations

**Phase 3: Advanced Features**
1. Closure execution
2. Native function dispatch
3. Namespace management
4. Quote support

**Phase 4: Optimization**
1. Custom allocator
2. Arena allocation for temporaries
3. Reference count batching
4. Profiling hooks

### Testing Strategy

1. **Unit Tests**: Test each runtime function in isolation
2. **Integration Tests**: Test compiled code calling runtime functions
3. **Memory Tests**: Verify no leaks, proper reference counting
4. **Concurrency Tests**: Test thread safety of all operations
5. **Compatibility Tests**: Ensure runtime matches interpreter behavior

### File Organization

Suggested structure:

```
src/runtime/
├── mod.rs              # Public API and initialization
├── memory.rs           # Allocation and reference counting
├── value.rs            # Value boxing/unboxing
├── string.rs           # String operations
├── vector.rs           # Vector operations
├── map.rs              # Map operations
├── set.rs              # Set operations
├── list.rs             # List operations
├── closure.rs          # Closure operations
├── dispatch.rs         # Function call dispatch
├── var.rs              # Global variable management
├── namespace.rs        # Namespace management
├── native.rs           # Native function registry
└── error.rs            # Error handling and panic
```

### C API Export

All runtime functions must be exported with C linkage:

```rust
#[no_mangle]
pub extern "C" fn evolve_alloc(size: i64) -> *mut u8 {
    // Implementation
}
```

This allows compiled LLVM code to link against the runtime library.
