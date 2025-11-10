//! Persistent hash array mapped trie (HAMT) map with insertion-order tracking.
//!
//! The map stores key/value bindings inside a HAMT-backed radix tree while a
//! persistent vector keeps the sequence of insertions. Updates preserve the
//! original insertion index for a key, and removals mark the associated entry
//! as inactive so iteration still yields the surviving bindings in their
//! original order.

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::collections::Vector;
use crate::collections::vector::Iter as VectorIter;

const BRANCH_BITS: u32 = 5;
const BITMASK: u64 = ((1 << BRANCH_BITS) as u64) - 1;
const MAX_SHIFT: u32 = BRANCH_BITS * 7; // Supports up to 35 bits of the hash, plenty in practice.
const COLLISION_SPLIT_THRESHOLD: usize = 8;

fn hash_key<K: Hash>(key: &K) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

fn index_from_hash(hash: u64, shift: u32) -> usize {
    ((hash >> shift) & BITMASK) as usize
}

fn bit_for(index: usize) -> u32 {
    1 << index
}

fn index_in_bitmap(bitmap: u32, bit: u32) -> usize {
    (bitmap & (bit - 1)).count_ones() as usize
}

#[derive(Clone)]
struct Entry<K, V> {
    key: K,
    value: V,
    hash: u64,
    order_index: usize,
}

#[derive(Clone)]
struct OrderEntry<K, V> {
    key: K,
    value: V,
    alive: bool,
}

enum InsertAction {
    Added,
    Updated { order_index: usize },
}

struct RemoveResult<K, V> {
    node: Option<Arc<Node<K, V>>>,
    removed_index: Option<usize>,
}

#[derive(Clone)]
enum Node<K, V> {
    Leaf(Vec<Entry<K, V>>),
    Branch { bitmap: u32, children: Vec<Arc<Node<K, V>>> },
}

impl<K: Eq, V> Node<K, V> {
    fn get<'a>(
        node: &'a Arc<Node<K, V>>,
        shift: u32,
        hash: u64,
        key: &K,
    ) -> Option<&'a V> {
        match node.as_ref() {
            Node::Leaf(entries) => entries
                .iter()
                .find(|entry| entry.hash == hash && entry.key == *key)
                .map(|entry| &entry.value),
            Node::Branch { bitmap, children } => {
                let idx = index_from_hash(hash, shift);
                let bit = bit_for(idx);
                if bitmap & bit == 0 {
                    return None;
                }
                let child_pos = index_in_bitmap(*bitmap, bit);
                Node::get(&children[child_pos], shift + BRANCH_BITS, hash, key)
            }
        }
    }
}

impl<K: Eq + Clone, V: Clone> Node<K, V> {
    fn insert(
        node: &Arc<Node<K, V>>,
        shift: u32,
        hash: u64,
        key: K,
        value: V,
        order_index: usize,
    ) -> (Arc<Node<K, V>>, InsertAction) {
        match node.as_ref() {
            Node::Leaf(entries) => {
                if let Some(pos) = entries
                    .iter()
                    .position(|entry| entry.hash == hash && entry.key == key)
                {
                    let mut new_entries = entries.clone();
                    let order_index = new_entries[pos].order_index;
                    new_entries[pos].value = value;
                    (
                        Arc::new(Node::Leaf(new_entries)),
                        InsertAction::Updated { order_index },
                    )
                } else {
                    let mut new_entries = entries.clone();
                    new_entries.push(Entry { key, value, hash, order_index });
                    if Node::should_branch(&new_entries, shift)
                        || (new_entries.len() > COLLISION_SPLIT_THRESHOLD
                            && shift < MAX_SHIFT)
                    {
                        let node =
                            Node::build_branch_from_entries(new_entries, shift);
                        (node, InsertAction::Added)
                    } else {
                        (Arc::new(Node::Leaf(new_entries)), InsertAction::Added)
                    }
                }
            }
            Node::Branch { bitmap, children } => {
                let idx = index_from_hash(hash, shift);
                let bit = bit_for(idx);
                if bitmap & bit != 0 {
                    let child_pos = index_in_bitmap(*bitmap, bit);
                    let (new_child, action) = Node::insert(
                        &children[child_pos],
                        shift + BRANCH_BITS,
                        hash,
                        key,
                        value,
                        order_index,
                    );
                    let mut new_children = children.clone();
                    new_children[child_pos] = new_child;
                    (
                        Arc::new(Node::Branch {
                            bitmap: *bitmap,
                            children: new_children,
                        }),
                        action,
                    )
                } else {
                    let mut new_children = children.clone();
                    let pos = index_in_bitmap(*bitmap, bit);
                    new_children.insert(
                        pos,
                        Arc::new(Node::Leaf(vec![Entry {
                            key,
                            value,
                            hash,
                            order_index,
                        }])),
                    );
                    let new_bitmap = *bitmap | bit;
                    (
                        Arc::new(Node::Branch {
                            bitmap: new_bitmap,
                            children: new_children,
                        }),
                        InsertAction::Added,
                    )
                }
            }
        }
    }

    fn remove(
        node: &Arc<Node<K, V>>,
        shift: u32,
        hash: u64,
        key: &K,
    ) -> RemoveResult<K, V> {
        match node.as_ref() {
            Node::Leaf(entries) => {
                if let Some(pos) = entries
                    .iter()
                    .position(|entry| entry.hash == hash && entry.key == *key)
                {
                    let mut new_entries = entries.clone();
                    let removed = new_entries.remove(pos);
                    let node = if new_entries.is_empty() {
                        None
                    } else {
                        Some(Arc::new(Node::Leaf(new_entries)))
                    };
                    RemoveResult { node, removed_index: Some(removed.order_index) }
                } else {
                    RemoveResult { node: Some(node.clone()), removed_index: None }
                }
            }
            Node::Branch { bitmap, children } => {
                let idx = index_from_hash(hash, shift);
                let bit = bit_for(idx);
                if bitmap & bit == 0 {
                    return RemoveResult {
                        node: Some(node.clone()),
                        removed_index: None,
                    };
                }
                let child_pos = index_in_bitmap(*bitmap, bit);
                let result = Node::remove(
                    &children[child_pos],
                    shift + BRANCH_BITS,
                    hash,
                    key,
                );

                if result.removed_index.is_none() {
                    return RemoveResult {
                        node: Some(node.clone()),
                        removed_index: None,
                    };
                }

                let mut new_children = children.clone();
                let mut new_bitmap = *bitmap;

                if let Some(updated_child) = result.node.clone() {
                    new_children[child_pos] = updated_child;
                } else {
                    new_children.remove(child_pos);
                    new_bitmap &= !bit;
                }

                let node = match new_children.len() {
                    0 => None,
                    1 => Some(new_children.into_iter().next().unwrap()),
                    _ => Some(Arc::new(Node::Branch {
                        bitmap: new_bitmap,
                        children: new_children,
                    })),
                };

                RemoveResult { node, removed_index: result.removed_index }
            }
        }
    }

    fn build_branch_from_entries(
        entries: Vec<Entry<K, V>>,
        shift: u32,
    ) -> Arc<Node<K, V>> {
        if entries.len() <= 1 || shift >= MAX_SHIFT {
            return Arc::new(Node::Leaf(entries));
        }

        let mut buckets: BTreeMap<usize, Vec<Entry<K, V>>> = BTreeMap::new();
        for entry in entries {
            let idx = index_from_hash(entry.hash, shift);
            buckets.entry(idx).or_default().push(entry);
        }

        let mut bitmap = 0u32;
        let mut children = Vec::with_capacity(buckets.len());

        for (idx, bucket) in buckets.into_iter() {
            let bit = bit_for(idx);
            bitmap |= bit;
            let child = if bucket.len() > 1 {
                Node::build_branch_from_entries(bucket, shift + BRANCH_BITS)
            } else {
                Arc::new(Node::Leaf(bucket))
            };
            children.push(child);
        }

        Arc::new(Node::Branch { bitmap, children })
    }

    fn should_branch(entries: &[Entry<K, V>], shift: u32) -> bool {
        if entries.len() <= 1 || shift >= MAX_SHIFT {
            return false;
        }
        let first_idx = index_from_hash(entries[0].hash, shift);
        entries
            .iter()
            .skip(1)
            .any(|entry| index_from_hash(entry.hash, shift) != first_idx)
    }
}

#[derive(Clone)]
pub struct Map<K, V> {
    len: usize,
    root: Option<Arc<Node<K, V>>>,
    order: Vector<OrderEntry<K, V>>,
}

impl<K, V> Map<K, V> {
    /// Creates a new empty map.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map: Map<i32, &str> = Map::new();
    /// assert!(map.is_empty());
    /// assert_eq!(map.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { len: 0, root: None, order: Vector::new() }
    }

    /// Returns the number of key-value pairs in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map = Map::new().insert("a", 1).insert("b", 2);
    /// assert_eq!(map.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map contains no key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map: Map<i32, &str> = Map::new();
    /// assert!(map.is_empty());
    ///
    /// let map = map.insert(1, "one");
    /// assert!(!map.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over the map's key-value pairs in insertion order.
    ///
    /// The iterator yields tuples of `(&K, &V)` in the order that keys were
    /// first inserted into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map = Map::new()
    ///     .insert("a", 1)
    ///     .insert("b", 2)
    ///     .insert("c", 3);
    ///
    /// let items: Vec<_> = map.iter().collect();
    /// assert_eq!(items, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);
    ///
    /// // Iterator is double-ended
    /// let reversed: Vec<_> = map.iter().rev().collect();
    /// assert_eq!(reversed, vec![(&"c", &3), (&"b", &2), (&"a", &1)]);
    /// ```
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter { inner: self.order.iter() }
    }
}

impl<K: Eq + Hash, V> Map<K, V> {
    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to check for
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map = Map::new().insert("a", 1).insert("b", 2);
    ///
    /// assert!(map.contains_key(&"a"));
    /// assert!(!map.contains_key(&"c"));
    /// ```
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns a reference to the value corresponding to the key, or `None` if
    /// the key is not present in the map.
    ///
    /// Time complexity: `O(log n)` average case
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map = Map::new().insert("a", 1).insert("b", 2);
    ///
    /// assert_eq!(map.get(&"a"), Some(&1));
    /// assert_eq!(map.get(&"b"), Some(&2));
    /// assert_eq!(map.get(&"c"), None);
    /// ```
    pub fn get(&self, key: &K) -> Option<&V> {
        let root = self.root.as_ref()?;
        let hash = hash_key(key);
        Node::get(root, 0, hash, key)
    }
}

impl<K: Eq + Hash + Clone, V: Clone> Map<K, V> {
    /// Inserts a key-value pair into the map, returning a new map.
    ///
    /// If the key already exists, the value is updated but the key maintains its
    /// original insertion position in the iteration order. If the key is new, it
    /// is appended to the end of the iteration order.
    ///
    /// The original map remains unchanged due to structural sharing.
    ///
    /// Time complexity: `O(log n)` average case
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to associate with the key
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map1 = Map::new();
    /// let map2 = map1.insert("a", 1);
    /// let map3 = map2.insert("b", 2);
    ///
    /// assert_eq!(map1.len(), 0);
    /// assert_eq!(map2.len(), 1);
    /// assert_eq!(map3.len(), 2);
    /// assert_eq!(map3.get(&"a"), Some(&1));
    ///
    /// // Updating an existing key preserves insertion order
    /// let updated = map3.insert("a", 10);
    /// assert_eq!(updated.get(&"a"), Some(&10));
    /// let items: Vec<_> = updated.iter().collect();
    /// assert_eq!(items, vec![(&"a", &10), (&"b", &2)]);
    ///
    /// // Original map is unchanged
    /// assert_eq!(map3.get(&"a"), Some(&1));
    /// ```
    pub fn insert(&self, key: K, value: V) -> Self {
        let hash = hash_key(&key);
        let new_index = self.order.len();
        let (new_root, action) = match &self.root {
            Some(root) => {
                Node::insert(root, 0, hash, key.clone(), value.clone(), new_index)
            }
            None => (
                Arc::new(Node::Leaf(vec![Entry {
                    key: key.clone(),
                    value: value.clone(),
                    hash,
                    order_index: new_index,
                }])),
                InsertAction::Added,
            ),
        };

        let mut len = self.len;
        let mut order = self.order.clone();

        match action {
            InsertAction::Added => {
                let entry = OrderEntry { key, value, alive: true };
                order = order.push_back(entry);
                len += 1;
                Self { len, root: Some(new_root), order }
            }
            InsertAction::Updated { order_index } => {
                if let Some(existing) = self.order.get(order_index) {
                    let mut updated = existing.clone();
                    updated.value = value;
                    order = self
                        .order
                        .update(order_index, updated)
                        .expect("order index should be valid");
                }
                Self { len, root: Some(new_root), order }
            }
        }
    }

    /// Removes a key from the map, returning a new map without that key.
    ///
    /// If the key exists, it is removed and the map's length is decremented. The
    /// original map remains unchanged due to structural sharing.
    ///
    /// Time complexity: `O(log n)` average case
    ///
    /// # Arguments
    ///
    /// * `key` - The key to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Map;
    ///
    /// let map = Map::new()
    ///     .insert("a", 1)
    ///     .insert("b", 2)
    ///     .insert("c", 3);
    ///
    /// let removed = map.remove(&"b");
    /// assert_eq!(removed.len(), 2);
    /// assert_eq!(removed.get(&"b"), None);
    ///
    /// // Remaining keys maintain their order
    /// let items: Vec<_> = removed.iter().collect();
    /// assert_eq!(items, vec![(&"a", &1), (&"c", &3)]);
    ///
    /// // Original map is unchanged
    /// assert_eq!(map.len(), 3);
    /// assert_eq!(map.get(&"b"), Some(&2));
    /// ```
    pub fn remove(&self, key: &K) -> Self {
        let root = match &self.root {
            Some(root) => root,
            None => return self.clone(),
        };
        let hash = hash_key(key);
        let result = Node::remove(root, 0, hash, key);
        let removed_index = match result.removed_index {
            Some(index) => index,
            None => return self.clone(),
        };

        let mut order = self.order.clone();
        if let Some(entry) = self.order.get(removed_index) {
            let mut updated = entry.clone();
            updated.alive = false;
            order = self
                .order
                .update(removed_index, updated)
                .expect("order index should be valid");
        }

        let len = self.len - 1;

        Self { len, root: result.node, order }
    }
}

impl<K, V> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash, V: PartialEq> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.iter().all(|(key, value)| other.get(key).map_or(false, |v| v == value))
    }
}

impl<K: Eq + Hash, V: Eq> Eq for Map<K, V> {}

impl<K: fmt::Debug + Eq + Hash, V: fmt::Debug> fmt::Debug for Map<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (key, value) in self.iter() {
            map.entry(key, value);
        }
        map.finish()
    }
}

impl<K: Eq + Hash + Clone, V: Clone> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Map::new();
        for (key, value) in iter {
            map = map.insert(key, value);
        }
        map
    }
}

pub struct Iter<'a, K, V> {
    inner: VectorIter<'a, OrderEntry<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.inner.next() {
            if entry.alive {
                return Some((&entry.key, &entry.value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.inner.size_hint();
        (0, upper)
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.inner.next_back() {
            if entry.alive {
                return Some((&entry.key, &entry.value));
            }
        }
        None
    }
}

impl<'a, K, V> std::iter::FusedIterator for Iter<'a, K, V> {}

impl<'a, K, V> IntoIterator for &'a Map<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::Map;

    #[test]
    fn new_map_is_empty() {
        let map: Map<i32, i32> = Map::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn insert_preserves_order() {
        let map = Map::new().insert("a", 1).insert("b", 2).insert("c", 3);
        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);
    }

    #[test]
    fn update_keeps_original_position() {
        let map = Map::new().insert("a", 1).insert("b", 2);
        let updated = map.insert("a", 10);
        assert_eq!(updated.len(), 2);
        let items: Vec<_> = updated.iter().collect();
        assert_eq!(items, vec![(&"a", &10), (&"b", &2)]);
        assert_eq!(map.get(&"a"), Some(&1));
    }

    #[test]
    fn remove_marks_entry_inactive() {
        let map = Map::new().insert("a", 1).insert("b", 2).insert("c", 3);
        let removed = map.remove(&"b");
        assert_eq!(removed.len(), 2);
        assert_eq!(removed.get(&"b"), None);
        let items: Vec<_> = removed.iter().collect();
        assert_eq!(items, vec![(&"a", &1), (&"c", &3)]);
    }

    #[test]
    fn structural_sharing_after_insert() {
        let base = Map::new().insert(1, "one");
        let derived = base.insert(2, "two");
        assert_eq!(base.len(), 1);
        assert_eq!(derived.len(), 2);
        assert_eq!(base.get(&2), None);
    }

    #[test]
    fn reinserting_after_removal_adds_to_end() {
        let map = Map::new().insert("a", 1).insert("b", 2);
        let removed = map.remove(&"a");
        let reinserted = removed.insert("a", 42);
        let items: Vec<_> = reinserted.iter().collect();
        assert_eq!(items, vec![(&"b", &2), (&"a", &42)]);
    }
}
