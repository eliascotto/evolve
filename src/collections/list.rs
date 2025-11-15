//! Persistent singly linked list with `O(1)` prepend and structural sharing.
//!
//! The list stores its nodes behind `Arc` pointers so that cloning a list is
//! inexpensive and results in shared structure instead of data duplication.
//! This makes it well suited for functional-style workloads where new variants
//! of an existing list are created frequently.

use std::fmt;
use std::iter::{FromIterator, FusedIterator};
use std::sync::Arc;
use std::vec::Vec;

#[derive(Clone)]
pub struct List<T> {
    head: Option<Arc<Node<T>>>,
    len: usize,
}

struct Node<T> {
    elem: T,
    next: Option<Arc<Node<T>>>,
}

impl<T> List<T> {
    /// Creates a new empty list.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list: List<i32> = List::new();
    /// assert!(list.is_empty());
    /// assert_eq!(list.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { head: None, len: 0 }
    }

    /// Returns the number of elements in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list = List::new().prepend(1).prepend(2).prepend(3);
    /// assert_eq!(list.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list: List<i32> = List::new();
    /// assert!(list.is_empty());
    ///
    /// let list = list.prepend(1);
    /// assert!(!list.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Prepends a value to the front of the list, returning a new list.
    ///
    /// This operation is `O(1)` and preserves the original list through structural
    /// sharing, allowing efficient creation of new versions without copying the
    /// entire data structure.
    ///
    /// Time complexity: `O(1)`
    ///
    /// # Arguments
    ///
    /// * `value` - The value to prepend to the list
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list1 = List::new();
    /// let list2 = list1.prepend(1);
    /// let list3 = list2.prepend(2);
    ///
    /// assert_eq!(list1.len(), 0);
    /// assert_eq!(list2.len(), 1);
    /// assert_eq!(list3.len(), 2);
    ///
    /// // Elements are in reverse order of prepend operations
    /// let values: Vec<_> = list3.iter().copied().collect();
    /// assert_eq!(values, vec![2, 1]);
    /// ```
    pub fn prepend(&self, value: T) -> Self {
        let new_node = Arc::new(Node { elem: value, next: self.head.clone() });

        Self { head: Some(new_node), len: self.len + 1 }
    }

    /// Returns a reference to the first element, or `None` if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list = List::new();
    /// assert_eq!(list.head(), None);
    ///
    /// let list = list.prepend(1).prepend(2).prepend(3);
    /// assert_eq!(list.head(), Some(&3));
    /// ```
    pub fn head(&self) -> Option<&T> {
        self.head.as_deref().map(|node| &node.elem)
    }

    /// Returns the list without the first element, or `None` if the list is empty.
    ///
    /// The original list remains unchanged due to structural sharing.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list = List::new().prepend(1).prepend(2).prepend(3);
    /// let tail = list.tail().unwrap();
    ///
    /// assert_eq!(tail.len(), 2);
    /// assert_eq!(tail.head(), Some(&2));
    ///
    /// // Original list is unchanged
    /// assert_eq!(list.len(), 3);
    /// assert_eq!(list.head(), Some(&3));
    /// ```
    pub fn tail(&self) -> Option<Self> {
        let node = self.head.as_ref()?;
        debug_assert!(self.len > 0);

        Some(Self { head: node.next.clone(), len: self.len - 1 })
    }

    /// Splits the list into its head element and the tail list.
    ///
    /// Returns `None` if the list is empty. Otherwise, returns a tuple containing
    /// a reference to the first element and a new list containing the remaining elements.
    ///
    /// The original list remains unchanged due to structural sharing.
    ///
    /// # Returns
    ///
    /// * `Some((head, tail))` if the list is not empty
    /// * `None` if the list is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list = List::new().prepend(10).prepend(20).prepend(30);
    /// let (head, tail) = list.split_first().unwrap();
    ///
    /// assert_eq!(*head, 30);
    /// assert_eq!(tail.len(), 2);
    /// assert_eq!(tail.head(), Some(&20));
    ///
    /// // Original list is unchanged
    /// assert_eq!(list.len(), 3);
    /// ```
    pub fn split_first(&self) -> Option<(&T, Self)> {
        let node = self.head.as_ref()?;
        debug_assert!(self.len > 0);

        let rest = Self { head: node.next.clone(), len: self.len - 1 };

        Some((&node.elem, rest))
    }

    /// Returns an iterator over the list's elements.
    ///
    /// The iterator yields references to the elements in order from head to tail.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::List;
    ///
    /// let list = List::new().prepend(1).prepend(2).prepend(3);
    ///
    /// let mut iter = list.iter();
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), None);
    ///
    /// // Can collect into a vector
    /// let values: Vec<_> = list.iter().copied().collect();
    /// assert_eq!(values, vec![3, 2, 1]);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { next: self.head.as_deref(), remaining: self.len }
    }
}

impl<T> Default for List<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq> PartialEq for List<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq> Eq for List<T> {}

impl<T: fmt::Debug> fmt::Debug for List<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> FromIterator<T> for List<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        let mut list = List::new();

        for item in items.into_iter().rev() {
            list = list.prepend(item);
        }

        list
    }
}

impl<'a, T> IntoIterator for &'a List<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct Iter<'a, T> {
    next: Option<&'a Node<T>>,
    remaining: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.next?;
        self.next = node.next.as_deref();
        self.remaining = self.remaining.saturating_sub(1);
        Some(&node.elem)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> std::iter::ExactSizeIterator for Iter<'a, T> {}
impl<'a, T> FusedIterator for Iter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::List;
    use crate::interner;
    use crate::reader::Span;
    use crate::value::{self, Value};
    use std::sync::Arc;

    #[test]
    fn new_list_is_empty() {
        let list: List<i32> = List::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.head().is_none());
    }

    #[test]
    fn prepend_adds_elements_to_front() {
        let list = List::new().prepend(1).prepend(2).prepend(3);

        assert_eq!(list.len(), 3);
        assert_eq!(list.head(), Some(&3));

        let values: Vec<_> = list.iter().copied().collect();
        assert_eq!(values, vec![3, 2, 1]);
    }

    #[test]
    fn tail_returns_remaining_list() {
        let list = List::new().prepend(1).prepend(2);
        let tail = list.tail().expect("tail should exist");

        assert_eq!(tail.len(), 1);
        assert_eq!(tail.head(), Some(&1));
        assert_eq!(tail.iter().copied().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn split_first_returns_head_and_tail() {
        let list = List::new().prepend(10).prepend(20);
        let (head, tail) = list.split_first().expect("should split");

        assert_eq!(*head, 20);
        assert_eq!(tail.iter().copied().collect::<Vec<_>>(), vec![10]);
    }

    #[test]
    fn list_is_persistent() {
        let base = List::new().prepend(2).prepend(1);
        let extended = base.prepend(0);

        assert_eq!(base.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(extended.iter().copied().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn from_iterator_preserves_order() {
        let list: List<_> = (1..=4).collect();
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn list_is_heterogenous() {
        let int_value = Value::Int { span: Span { start: 0, end: 0 }, value: 1 };
        let string_value = Value::String {
            span: Span { start: 0, end: 0 },
            value: Arc::from("hello"),
        };
        let symbol_value = value::symbol(
            interner::intern_sym("hello"),
            None,
            None,
            Span { start: 0, end: 0 },
        );
        let list = List::new()
            .prepend(int_value.clone())
            .prepend(string_value.clone())
            .prepend(symbol_value.clone());
        assert_eq!(list.head(), Some(&symbol_value));
        assert_eq!(list.tail().unwrap().head(), Some(&string_value));
        assert_eq!(list.tail().unwrap().tail().unwrap().head(), Some(&int_value));
        assert_eq!(
            list.tail().unwrap().tail().unwrap().tail().unwrap().head(),
            None
        );
    }
}
