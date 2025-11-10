//! Persistent set implemented on top of the persistent map.
//!
//! The set preserves insertion order by reusing the `Map`'s HAMT structure with
//! auxiliary vector. Elements are stored as keys with a `()` value, giving the
//! set efficient structural sharing, `O(log n)` lookup, and order-aware
//! iteration.

use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;

use crate::collections::map::{Iter as MapIter, Map};

#[derive(Clone)]
pub struct Set<T> {
    map: Map<T, ()>,
}

impl<T> Set<T> {
    /// Creates a new empty set.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Set;
    ///
    /// let set: Set<i32> = Set::new();
    /// assert!(set.is_empty());
    /// assert_eq!(set.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { map: Map::new() }
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Set;
    ///
    /// let set = Set::new().insert("a").insert("b");
    /// assert_eq!(set.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Set;
    ///
    /// let set: Set<i32> = Set::new();
    /// assert!(set.is_empty());
    ///
    /// let set = set.insert(1);
    /// assert!(!set.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns an iterator over the set's elements in insertion order.
    ///
    /// The iterator yields references to the elements in the order they were
    /// first inserted into the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Set;
    ///
    /// let set = Set::new().insert("a").insert("b").insert("c");
    ///
    /// let items: Vec<_> = set.iter().copied().collect();
    /// assert_eq!(items, vec!["a", "b", "c"]);
    ///
    /// // Iterator is double-ended
    /// let reversed: Vec<_> = set.iter().rev().copied().collect();
    /// assert_eq!(reversed, vec!["c", "b", "a"]);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { inner: self.map.iter() }
    }
}

impl<T: Eq + Hash> Set<T> {
    /// Returns `true` if the set contains the specified value.
    ///
    /// Time complexity: `O(log n)` average case
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check for
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Set;
    ///
    /// let set = Set::new().insert("a").insert("b");
    ///
    /// assert!(set.contains(&"a"));
    /// assert!(!set.contains(&"c"));
    /// ```
    pub fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }
}

impl<T: Eq + Hash + Clone> Set<T> {
    /// Adds a value to the set, returning a new set.
    ///
    /// If the value already exists in the set, the set is returned unchanged
    /// (no duplicate is added). The original set remains unchanged due to
    /// structural sharing.
    ///
    /// Time complexity: `O(log n)` average case
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Set;
    ///
    /// let set1 = Set::new();
    /// let set2 = set1.insert("a");
    /// let set3 = set2.insert("b");
    ///
    /// assert_eq!(set1.len(), 0);
    /// assert_eq!(set2.len(), 1);
    /// assert_eq!(set3.len(), 2);
    /// assert!(set3.contains(&"a"));
    ///
    /// // Duplicate inserts are ignored
    /// let set4 = set3.insert("a");
    /// assert_eq!(set4.len(), 2);
    ///
    /// // Original set is unchanged
    /// assert_eq!(set3.len(), 2);
    /// ```
    pub fn insert(&self, value: T) -> Self {
        Self { map: self.map.insert(value, ()) }
    }

    /// Removes a value from the set, returning a new set without that value.
    ///
    /// If the value exists, it is removed and the set's length is decremented.
    /// The original set remains unchanged due to structural sharing.
    ///
    /// Time complexity: `O(log n)` average case
    ///
    /// # Arguments
    ///
    /// * `value` - The value to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Set;
    ///
    /// let set = Set::new()
    ///     .insert("a")
    ///     .insert("b")
    ///     .insert("c");
    ///
    /// let removed = set.remove(&"b");
    /// assert_eq!(removed.len(), 2);
    /// assert!(!removed.contains(&"b"));
    ///
    /// // Remaining elements maintain their order
    /// let items: Vec<_> = removed.iter().copied().collect();
    /// assert_eq!(items, vec!["a", "c"]);
    ///
    /// // Original set is unchanged
    /// assert_eq!(set.len(), 3);
    /// assert!(set.contains(&"b"));
    /// ```
    pub fn remove(&self, value: &T) -> Self {
        Self { map: self.map.remove(value) }
    }
}

impl<T: Eq + Hash + Clone> Default for Set<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash> PartialEq for Set<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|value| other.contains(value))
    }
}

impl<T: Eq + Hash> Eq for Set<T> {}

impl<T: fmt::Debug + Eq + Hash> fmt::Debug for Set<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Eq + Hash + Clone> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Set::new();
        for item in iter {
            set = set.insert(item);
        }
        set
    }
}

pub struct Iter<'a, T> {
    inner: MapIter<'a, T, ()>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.inner.size_hint();
        (lower, upper)
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(key, _)| key)
    }
}

impl<'a, T> std::iter::FusedIterator for Iter<'a, T> {}

impl<'a, T> IntoIterator for &'a Set<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::Set;

    #[test]
    fn new_set_is_empty() {
        let set: Set<i32> = Set::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.contains(&1));
    }

    #[test]
    fn insert_preserves_order_and_uniqueness() {
        let set = Set::new().insert("a").insert("b").insert("a").insert("c");
        assert_eq!(set.len(), 3);
        let items: Vec<_> = set.iter().copied().collect();
        assert_eq!(items, vec!["a", "b", "c"]);
    }

    #[test]
    fn remove_excludes_value() {
        let set = Set::new().insert(1).insert(2).insert(3);
        let removed = set.remove(&2);
        assert_eq!(removed.len(), 2);
        assert!(!removed.contains(&2));
        assert_eq!(removed.iter().copied().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn structural_sharing_on_insert() {
        let base = Set::new().insert(1);
        let derived = base.insert(2);
        assert_eq!(base.len(), 1);
        assert!(base.contains(&1));
        assert!(!base.contains(&2));
        assert_eq!(derived.len(), 2);
        assert!(derived.contains(&1));
        assert!(derived.contains(&2));
    }

    #[test]
    fn iterator_is_double_ended() {
        let set = Set::new().insert(1).insert(2).insert(3);
        let forward: Vec<_> = set.iter().copied().collect();
        let reverse: Vec<_> = set.iter().rev().copied().collect();
        assert_eq!(forward, vec![1, 2, 3]);
        assert_eq!(reverse, vec![3, 2, 1]);
    }
}
