use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use std::sync::Mutex;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct KeywId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NsId(pub u32);

struct Table {
    map: FxHashMap<String, u32>, // text -> id
    rev: Vec<String>,            // id -> text
}

impl Table {
    fn new() -> Self {
        Self { map: FxHashMap::default(), rev: Vec::new() }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.rev.len() as u32;
        self.rev.push(s.to_owned());
        self.map.insert(self.rev[id as usize].clone(), id);
        id
    }

    fn resolve(&self, id: u32) -> &str {
        &self.rev[id as usize]
    }
}

struct Interner {
    syms: Table,
    kws: Table,
    namespaces: Table,
}

impl Interner {
    fn new() -> Self {
        Self { syms: Table::new(), kws: Table::new(), namespaces: Table::new() }
    }

    // Symbols: store exactly as given (e.g., "user/map" or "map")
    // Symbols are always qualified if they contain '/'
    fn intern_sym(&mut self, s: &str) -> SymId {
        SymId(self.syms.intern(s))
    }

    fn sym_str(&self, id: SymId) -> &str {
        self.syms.resolve(id.0)
    }

    // Extract namespace and name from a symbol
    // Returns (namespace, name) where namespace is None if unqualified
    pub fn sym_split(&self, id: SymId) -> (Option<&str>, &str) {
        let full = self.sym_str(id);
        if let Some(pos) = full.rfind('/') {
            let (ns, name) = full.split_at(pos);
            (Some(ns), &name[1..]) // Skip the '/'
        } else {
            (None, full)
        }
    }

    // Keywords: store canonical text WITHOUT leading ':'
    // Input can be:
    // - ":foo/bar" -> stores "foo/bar"
    // - "::bar" -> caller should resolve to ":current-ns/bar" first
    // - ":bar" -> stores "bar" (unqualified)
    fn intern_kw(&mut self, s: &str) -> KeywId {
        let canon = s.strip_prefix(':').unwrap_or(s); // tolerate ":name" input

        // Handle double colon - this should be resolved by caller, but we handle it defensively
        let canon = if canon.starts_with(':') {
            // This is an error case - caller should have resolved ::bar before calling
            // But we'll treat it as unqualified for now
            &canon[1..]
        } else {
            canon
        };

        KeywId(self.kws.intern(canon))
    }

    fn kw_text(&self, id: KeywId) -> &str {
        self.kws.resolve(id.0)
    }

    fn kw_print(&self, id: KeywId) -> String {
        format!(":{}", self.kw_text(id))
    }

    // Extract namespace and name from a keyword
    // Returns (namespace, name) where namespace is None if unqualified
    pub fn kw_split(&self, id: KeywId) -> (Option<&str>, &str) {
        let full = self.kw_text(id);
        if let Some(pos) = full.rfind('/') {
            let (ns, name) = full.split_at(pos);
            (Some(ns), &name[1..]) // Skip the '/'
        } else {
            (None, full)
        }
    }

    fn intern_ns(&mut self, ns: &str) -> NsId {
        NsId(self.namespaces.intern(ns))
    }

    fn ns_str(&self, id: NsId) -> &str {
        self.namespaces.resolve(id.0)
    }
}

static INTERNER: Lazy<Mutex<Interner>> = Lazy::new(|| Mutex::new(Interner::new()));

pub fn intern_sym(s: &str) -> SymId {
    INTERNER.lock().unwrap().intern_sym(s)
}

pub fn sym_to_str(id: SymId) -> String {
    INTERNER.lock().unwrap().sym_str(id).to_owned()
}

pub fn sym_split(id: SymId) -> (Option<String>, String) {
    let interner = INTERNER.lock().unwrap();
    let (ns, name) = interner.sym_split(id);
    (ns.map(|s| s.to_string()), name.to_string())
}

pub fn intern_kw(s: &str) -> KeywId {
    INTERNER.lock().unwrap().intern_kw(s)
}

pub fn kw_to_str(id: KeywId) -> String {
    INTERNER.lock().unwrap().kw_text(id).to_owned()
}

pub fn kw_print(id: KeywId) -> String {
    INTERNER.lock().unwrap().kw_print(id)
}

pub fn kw_split(id: KeywId) -> (Option<String>, String) {
    let interner = INTERNER.lock().unwrap();
    let (ns, name) = interner.kw_split(id);
    (ns.map(|s| s.to_string()), name.to_string())
}

pub fn intern_ns(ns: &str) -> NsId {
    INTERNER.lock().unwrap().intern_ns(ns)
}

pub fn ns_to_str(id: NsId) -> String {
    INTERNER.lock().unwrap().ns_str(id).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_same_symbol_returns_same_id() {
        let id1 = intern_sym("foo");
        let id2 = intern_sym("foo");
        assert_eq!(id1, id2);
        assert_eq!(sym_to_str(id1), "foo");
    }

    #[test]
    fn intern_different_symbols_returns_different_ids() {
        let id1 = intern_sym("foo");
        let id2 = intern_sym("bar");
        assert_ne!(id1, id2);
    }

    #[test]
    fn sym_split_unqualified() {
        let id = intern_sym("foo");
        let (ns, name) = sym_split(id);
        assert_eq!(ns, None);
        assert_eq!(name, "foo");
    }

    #[test]
    fn sym_split_qualified() {
        let id = intern_sym("user/foo");
        let (ns, name) = sym_split(id);
        assert_eq!(ns, Some("user".to_string()));
        assert_eq!(name, "foo");
    }

    #[test]
    fn kw_intern_strips_colon() {
        let id = intern_kw(":foo");
        assert_eq!(kw_to_str(id), "foo");
        assert_eq!(kw_print(id), ":foo");
    }

    #[test]
    fn kw_intern_without_colon() {
        let id = intern_kw("foo");
        assert_eq!(kw_to_str(id), "foo");
        assert_eq!(kw_print(id), ":foo");
    }

    #[test]
    fn kw_split_qualified() {
        let id = intern_kw(":user/foo");
        let (ns, name) = kw_split(id);
        assert_eq!(ns, Some("user".to_string()));
        assert_eq!(name, "foo");
    }

    #[test]
    fn namespace_intern() {
        let id1 = intern_ns("user");
        let id2 = intern_ns("user");
        assert_eq!(id1, id2);
        assert_eq!(ns_to_str(id1), "user");
    }
}
