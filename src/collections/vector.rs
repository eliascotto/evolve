//! Persistent vector implemented as an RRB-Tree.
//!
//! The implementation mirrors the behaviour of Clojure's persistent vectors but
//! also keeps track of relaxed size information so that subtrees can be
//! concatenated efficiently. In practice this provides `O(1)` amortised push,
//! `O(log₃₂ n)` random access/updates, and preserves previous versions through
//! structural sharing.

use std::fmt;
use std::iter::{DoubleEndedIterator, FromIterator, FusedIterator};
use std::ops::Index;
use std::sync::Arc;

const BRANCH_BITS: u32 = 5;
const BRANCH_FACTOR: usize = 1 << BRANCH_BITS; // 32

#[derive(Clone)]
enum Node<T> {
    Leaf(Vec<T>),
    Branch { children: Vec<Arc<Node<T>>>, sizes: Option<Vec<usize>> },
}

impl<T> Node<T> {
    fn leaf(values: Vec<T>) -> Self {
        debug_assert!(values.len() <= BRANCH_FACTOR);
        Node::Leaf(values)
    }

    fn branch(children: Vec<Arc<Node<T>>>) -> Self {
        debug_assert!(!children.is_empty());
        debug_assert!(children.len() <= BRANCH_FACTOR);
        let mut sizes = Vec::with_capacity(children.len());
        let mut total = 0usize;
        for child in &children {
            total += child.total_len();
            sizes.push(total);
        }
        Node::Branch { children, sizes: Some(sizes) }
    }

    fn total_len(&self) -> usize {
        match self {
            Node::Leaf(values) => values.len(),
            Node::Branch { sizes, children } => {
                if let Some(sizes) = sizes {
                    *sizes.last().unwrap()
                } else {
                    children.iter().map(|child| child.total_len()).sum()
                }
            }
        }
    }

    fn new_path(mut shift: u32, leaf: Arc<Node<T>>) -> Arc<Node<T>> {
        debug_assert!(matches!(leaf.as_ref(), Node::Leaf(_)));
        if shift == 0 {
            return leaf;
        }

        let mut node = leaf;
        while shift > 0 {
            node = Arc::new(Node::branch(vec![node]));
            shift -= BRANCH_BITS;
        }
        node
    }

    fn child_index_for(sizes: &[usize], index: usize) -> (usize, usize) {
        let mut i = 0;
        while sizes[i] <= index {
            i += 1;
        }
        let local = if i == 0 { index } else { index - sizes[i - 1] };
        (i, local)
    }

    fn push_leaf(
        node: &Arc<Node<T>>,
        shift: u32,
        leaf: Arc<Node<T>>,
    ) -> PushResult<T> {
        match node.as_ref() {
            Node::Leaf(_) => {
                unreachable!("push_leaf should never be called on a leaf node")
            }
            Node::Branch { children, .. } => {
                if children.is_empty() {
                    let child = if shift == BRANCH_BITS {
                        leaf
                    } else {
                        Node::new_path(shift - BRANCH_BITS, leaf)
                    };
                    let new_children = vec![child];
                    return PushResult {
                        node: Arc::new(Node::branch(new_children)),
                        overflow: None,
                    };
                }

                if shift == BRANCH_BITS {
                    if children.len() < BRANCH_FACTOR {
                        let mut new_children = children.clone();
                        new_children.push(leaf);
                        PushResult {
                            node: Arc::new(Node::branch(new_children)),
                            overflow: None,
                        }
                    } else {
                        PushResult {
                            node: node.clone(),
                            overflow: Some(Arc::new(Node::branch(vec![leaf]))),
                        }
                    }
                } else {
                    let mut new_children = children.clone();
                    let last_index = new_children.len() - 1;
                    let last_child = new_children[last_index].clone();
                    let child_result =
                        Node::push_leaf(&last_child, shift - BRANCH_BITS, leaf);

                    if let Some(extra) = child_result.overflow {
                        if new_children.len() < BRANCH_FACTOR {
                            new_children[last_index] = child_result.node;
                            new_children.push(extra);
                            PushResult {
                                node: Arc::new(Node::branch(new_children)),
                                overflow: None,
                            }
                        } else {
                            new_children[last_index] = child_result.node;
                            let overflow = Arc::new(Node::branch(vec![extra]));
                            PushResult {
                                node: Arc::new(Node::branch(new_children)),
                                overflow: Some(overflow),
                            }
                        }
                    } else {
                        new_children[last_index] = child_result.node;
                        PushResult {
                            node: Arc::new(Node::branch(new_children)),
                            overflow: None,
                        }
                    }
                }
            }
        }
    }

    fn pop_leaf(node: &Arc<Node<T>>, shift: u32) -> PopResult<T> {
        match node.as_ref() {
            Node::Leaf(_) => {
                unreachable!("pop_leaf should never be called on a leaf node")
            }
            Node::Branch { children, .. } => {
                debug_assert!(!children.is_empty());

                if shift == BRANCH_BITS {
                    let mut new_children = children.clone();
                    let leaf = new_children
                        .pop()
                        .expect("expected at least one child when popping a leaf");
                    let node = if new_children.is_empty() {
                        None
                    } else {
                        Some(Arc::new(Node::branch(new_children)))
                    };
                    PopResult { node, leaf }
                } else {
                    let mut new_children = children.clone();
                    let last_index = new_children
                        .len()
                        .checked_sub(1)
                        .expect("node should have at least one child");
                    let last_child = new_children[last_index].clone();
                    let result = Node::pop_leaf(&last_child, shift - BRANCH_BITS);

                    if let Some(updated_child) = result.node.clone() {
                        new_children[last_index] = updated_child;
                    } else {
                        new_children.pop();
                    }

                    let node = if new_children.is_empty() {
                        None
                    } else {
                        Some(Arc::new(Node::branch(new_children)))
                    };

                    PopResult { node, leaf: result.leaf }
                }
            }
        }
    }

    fn update(
        node: &Arc<Node<T>>,
        shift: u32,
        mut index: usize,
        value: T,
    ) -> Arc<Node<T>>
    where
        T: Clone,
    {
        match node.as_ref() {
            Node::Leaf(values) => {
                let mut new_values = values.clone();
                new_values[index] = value;
                Arc::new(Node::leaf(new_values))
            }
            Node::Branch { children, sizes } => {
                let sizes = sizes
                    .as_ref()
                    .expect("branch nodes should always have relaxed sizes");
                let (child_idx, local_index) =
                    Node::<T>::child_index_for(sizes, index);
                index = local_index;

                let mut new_children = children.clone();
                let updated_child = Node::update(
                    &children[child_idx],
                    shift - BRANCH_BITS,
                    index,
                    value,
                );
                new_children[child_idx] = updated_child;

                Arc::new(Node::branch(new_children))
            }
        }
    }
}

struct PushResult<T> {
    node: Arc<Node<T>>,
    overflow: Option<Arc<Node<T>>>,
}

struct PopResult<T> {
    node: Option<Arc<Node<T>>>,
    leaf: Arc<Node<T>>,
}

#[derive(Clone)]
pub struct Vector<T> {
    len: usize,
    shift: u32,
    root: Option<Arc<Node<T>>>,
    tail: Arc<Vec<T>>,
}

impl<T> Vector<T> {
    /// Creates a new empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let vec: Vector<i32> = Vector::new();
    /// assert!(vec.is_empty());
    /// ```
    pub fn new() -> Self {
        Self { len: 0, shift: 0, root: None, tail: Arc::new(Vec::new()) }
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let mut vec = Vector::new();
    /// vec = vec.push_back(1);
    /// vec = vec.push_back(2);
    /// assert_eq!(vec.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let vec: Vector<i32> = Vector::new();
    /// assert!(vec.is_empty());
    ///
    /// let vec = vec.push_back(1);
    /// assert!(!vec.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the element at the given index, or `None` if the
    /// index is out of bounds.
    ///
    /// Time complexity: `O(log₃₂ n)`
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to retrieve
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let mut vec = Vector::new();
    /// vec = vec.push_back(10);
    /// vec = vec.push_back(20);
    /// vec = vec.push_back(30);
    ///
    /// assert_eq!(vec.get(0), Some(&10));
    /// assert_eq!(vec.get(1), Some(&20));
    /// assert_eq!(vec.get(2), Some(&30));
    /// assert_eq!(vec.get(3), None);
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }

        let tail_offset = self.tail_offset();
        if index >= tail_offset {
            return self.tail.get(index - tail_offset);
        }

        let mut node = self.root.as_ref()?;
        let mut shift = self.shift;
        let mut idx = index;

        loop {
            match node.as_ref() {
                Node::Leaf(values) => return values.get(idx),
                Node::Branch { children, sizes } => {
                    let sizes = sizes
                        .as_ref()
                        .expect("branch nodes should have relaxed sizes");
                    let (child_idx, local_idx) =
                        Node::<T>::child_index_for(sizes, idx);
                    let child = &children[child_idx];

                    if shift == BRANCH_BITS {
                        match child.as_ref() {
                            Node::Leaf(values) => return values.get(local_idx),
                            Node::Branch { .. } => unreachable!(
                                "when shift == BRANCH_BITS, children must be leaves"
                            ),
                        }
                    } else {
                        idx = local_idx;
                        shift -= BRANCH_BITS;
                        node = child;
                    }
                }
            }
        }
    }

    /// Returns a reference to the last element, or `None` if the vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let mut vec = Vector::new();
    /// assert_eq!(vec.last(), None);
    ///
    /// vec = vec.push_back(1);
    /// vec = vec.push_back(2);
    /// vec = vec.push_back(3);
    /// assert_eq!(vec.last(), Some(&3));
    /// ```
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() { None } else { self.get(self.len - 1) }
    }

    /// Returns an iterator over the vector's elements.
    ///
    /// The iterator yields references to the elements in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let mut vec = Vector::new();
    /// vec = vec.push_back(1);
    /// vec = vec.push_back(2);
    /// vec = vec.push_back(3);
    ///
    /// let mut iter = vec.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    ///
    /// // Can also iterate in reverse
    /// let reversed: Vec<_> = vec.iter().rev().copied().collect();
    /// assert_eq!(reversed, vec![3, 2, 1]);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { vector: self, index: 0, end: self.len }
    }

    fn tail_offset(&self) -> usize {
        if self.len >= self.tail.len() { self.len - self.tail.len() } else { 0 }
    }
}

impl<T: Clone> Vector<T> {
    /// Appends an element to the back of the vector, returning a new vector.
    ///
    /// This operation preserves the original vector through structural sharing,
    /// allowing efficient creation of new versions without copying the entire
    /// data structure.
    ///
    /// Time complexity: `O(1)` amortized
    ///
    /// # Arguments
    ///
    /// * `value` - The value to append to the vector
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let vec1 = Vector::new();
    /// let vec2 = vec1.push_back(1);
    /// let vec3 = vec2.push_back(2);
    ///
    /// assert_eq!(vec1.len(), 0);
    /// assert_eq!(vec2.len(), 1);
    /// assert_eq!(vec3.len(), 2);
    /// assert_eq!(vec3.get(0), Some(&1));
    /// assert_eq!(vec3.get(1), Some(&2));
    /// ```
    pub fn push_back(&self, value: T) -> Self {
        let mut new_tail = self.tail.as_ref().clone();
        if new_tail.len() < BRANCH_FACTOR {
            new_tail.push(value);
            return Self {
                len: self.len + 1,
                shift: self.shift,
                root: self.root.clone(),
                tail: Arc::new(new_tail),
            };
        }

        let tail_node = Arc::new(Node::leaf(new_tail));
        let mut new_shift;
        let new_root = if let Some(root) = &self.root {
            let push_result = Node::push_leaf(root, self.shift, tail_node.clone());
            let mut node = push_result.node;
            new_shift = self.shift;

            if let Some(extra) = push_result.overflow {
                let mut children = Vec::with_capacity(2);
                children.push(node);
                children.push(extra);
                node = Arc::new(Node::branch(children));
                new_shift += BRANCH_BITS;
            }

            Some(node)
        } else {
            new_shift = BRANCH_BITS;
            Some(Arc::new(Node::branch(vec![tail_node.clone()])))
        };

        let mut fresh_tail = Vec::with_capacity(BRANCH_FACTOR);
        fresh_tail.push(value);

        Self {
            len: self.len + 1,
            shift: if new_root.is_some() { new_shift } else { 0 },
            root: new_root,
            tail: Arc::new(fresh_tail),
        }
    }

    /// Removes and returns the last element, along with a new vector without that element.
    ///
    /// Returns `None` if the vector is empty. Otherwise, returns a tuple containing
    /// the new vector (without the last element) and the removed element.
    ///
    /// The original vector remains unchanged due to structural sharing.
    ///
    /// # Returns
    ///
    /// * `Some((new_vector, removed_element))` if the vector is not empty
    /// * `None` if the vector is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let mut vec = Vector::new();
    /// vec = vec.push_back(1);
    /// vec = vec.push_back(2);
    /// vec = vec.push_back(3);
    ///
    /// let (vec_after_pop, removed) = vec.pop_back().unwrap();
    /// assert_eq!(removed, 3);
    /// assert_eq!(vec_after_pop.len(), 2);
    /// assert_eq!(vec_after_pop.get(1), Some(&2));
    ///
    /// // Original vector is unchanged
    /// assert_eq!(vec.len(), 3);
    /// assert_eq!(vec.get(2), Some(&3));
    /// ```
    pub fn pop_back(&self) -> Option<(Self, T)> {
        if self.is_empty() {
            return None;
        }

        let tail_len = self.tail.len();
        if tail_len > 1 {
            let mut new_tail = self.tail.as_ref().clone();
            let removed =
                new_tail.pop().expect("tail has at least one element when len > 0");
            return Some((
                Self {
                    len: self.len - 1,
                    shift: self.shift,
                    root: self.root.clone(),
                    tail: Arc::new(new_tail),
                },
                removed,
            ));
        }

        if self.root.is_none() {
            let removed = self
                .tail
                .last()
                .expect("tail must contain the only element")
                .clone();
            return Some((Self::new(), removed));
        }

        let root = self.root.as_ref().expect("root exists if tail len == 1");
        let pop_result = Node::pop_leaf(root, self.shift);
        let mut new_root = pop_result.node;
        let mut new_shift = self.shift;
        let removed =
            self.tail.last().expect("tail should contain the last element").clone();
        let tail_values = match pop_result.leaf.as_ref() {
            Node::Leaf(values) => values.clone(),
            _ => unreachable!("popped node must be a leaf"),
        };

        while new_shift > BRANCH_BITS {
            let collapse = if let Some(root_node) = &new_root {
                match root_node.as_ref() {
                    Node::Branch { children, .. } if children.len() == 1 => {
                        Some(children[0].clone())
                    }
                    _ => None,
                }
            } else {
                None
            };

            if let Some(child) = collapse {
                new_root = Some(child);
                new_shift -= BRANCH_BITS;
            } else {
                break;
            }
        }

        if let Some(root_node) = &new_root {
            if new_shift == BRANCH_BITS {
                if let Node::Branch { children, .. } = root_node.as_ref() {
                    if children.is_empty() {
                        new_root = None;
                        new_shift = 0;
                    }
                }
            }
        } else {
            new_shift = 0;
        }

        Some((
            Self {
                len: self.len - 1,
                shift: new_shift,
                root: new_root,
                tail: Arc::new(tail_values),
            },
            removed,
        ))
    }

    /// Updates the element at the given index, returning a new vector with the updated value.
    ///
    /// Returns `None` if the index is out of bounds. The original vector remains
    /// unchanged due to structural sharing.
    ///
    /// Time complexity: `O(log₃₂ n)`
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to update
    /// * `value` - The new value to store at the given index
    ///
    /// # Returns
    ///
    /// * `Some(new_vector)` if the index is valid
    /// * `None` if the index is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let mut vec = Vector::new();
    /// vec = vec.push_back(1);
    /// vec = vec.push_back(2);
    /// vec = vec.push_back(3);
    ///
    /// let updated = vec.update(1, 99).unwrap();
    /// assert_eq!(updated.get(1), Some(&99));
    ///
    /// // Original vector is unchanged
    /// assert_eq!(vec.get(1), Some(&2));
    ///
    /// // Out of bounds update returns None
    /// assert!(vec.update(10, 0).is_none());
    /// ```
    pub fn update(&self, index: usize, value: T) -> Option<Self> {
        if index >= self.len {
            return None;
        }

        let tail_offset = self.tail_offset();
        if index >= tail_offset {
            let mut new_tail = self.tail.as_ref().clone();
            new_tail[index - tail_offset] = value;
            return Some(Self {
                len: self.len,
                shift: self.shift,
                root: self.root.clone(),
                tail: Arc::new(new_tail),
            });
        }

        let root = self.root.as_ref()?;
        let updated_root = Node::update(root, self.shift, index, value);
        Some(Self {
            len: self.len,
            shift: self.shift,
            root: Some(updated_root),
            tail: self.tail.clone(),
        })
    }

    /// Concatenates this vector with another, returning a new vector containing
    /// all elements from both vectors.
    ///
    /// The elements from `other` are appended to the elements of `self`. Both
    /// original vectors remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector to append to this one
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::collections::Vector;
    ///
    /// let mut vec1 = Vector::new();
    /// vec1 = vec1.push_back(1);
    /// vec1 = vec1.push_back(2);
    ///
    /// let mut vec2 = Vector::new();
    /// vec2 = vec2.push_back(3);
    /// vec2 = vec2.push_back(4);
    ///
    /// let combined = vec1.concat(&vec2);
    /// assert_eq!(combined.len(), 4);
    /// assert_eq!(combined.get(0), Some(&1));
    /// assert_eq!(combined.get(1), Some(&2));
    /// assert_eq!(combined.get(2), Some(&3));
    /// assert_eq!(combined.get(3), Some(&4));
    ///
    /// // Original vectors are unchanged
    /// assert_eq!(vec1.len(), 2);
    /// assert_eq!(vec2.len(), 2);
    /// ```
    pub fn concat(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for item in other.iter() {
            result = result.push_back(item.clone());
        }
        result
    }
}

impl<T> Default for Vector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq> PartialEq for Vector<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq> Eq for Vector<T> {}

impl<T: fmt::Debug> fmt::Debug for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<T: Clone> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = Vector::new();
        for item in iter {
            vec = vec.push_back(item);
        }
        vec
    }
}

pub struct Iter<'a, T> {
    vector: &'a Vector<T>,
    index: usize,
    end: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }
        let result = self.vector.get(self.index);
        self.index += 1;
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }
        self.end -= 1;
        self.vector.get(self.end)
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}
impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<'a, T> IntoIterator for &'a Vector<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::Vector;

    #[test]
    fn new_vector_is_empty() {
        let vector: Vector<i32> = Vector::new();
        assert!(vector.is_empty());
        assert_eq!(vector.len(), 0);
        assert!(vector.get(0).is_none());
    }

    #[test]
    fn push_and_get_across_branches() {
        let mut vec = Vector::new();
        for i in 0..100 {
            vec = vec.push_back(i);
        }

        assert_eq!(vec.len(), 100);
        assert_eq!(vec.get(0), Some(&0));
        assert_eq!(vec.get(31), Some(&31));
        assert_eq!(vec.get(32), Some(&32));
        assert_eq!(vec.get(99), Some(&99));
        assert!(vec.get(100).is_none());
    }

    #[test]
    fn structural_sharing_on_push() {
        let base = Vector::new();
        let vec1 = base.push_back(1);
        let vec2 = vec1.push_back(2);
        let vec3 = vec2.push_back(3);

        assert_eq!(vec1.len(), 1);
        assert_eq!(vec1.get(0), Some(&1));

        assert_eq!(vec2.len(), 2);
        assert_eq!(vec2.get(1), Some(&2));

        assert_eq!(vec3.len(), 3);
        assert_eq!(vec3.get(2), Some(&3));
    }

    #[test]
    fn pop_back_returns_previous_state() {
        let mut vec = Vector::new();
        for i in 0..40 {
            vec = vec.push_back(i);
        }

        let (vec_after_pop, removed) = vec.pop_back().expect("pop should succeed");
        assert_eq!(removed, 39);
        assert_eq!(vec_after_pop.len(), 39);
        assert_eq!(vec_after_pop.get(38), Some(&38));

        // Ensure original vector untouched
        assert_eq!(vec.len(), 40);
        assert_eq!(vec.get(39), Some(&39));
    }

    #[test]
    fn update_replaces_element() {
        let mut vec = Vector::new();
        for i in 0..64 {
            vec = vec.push_back(i);
        }

        let updated = vec.update(50, 500).expect("index in range");
        assert_eq!(updated.get(50), Some(&500));
        assert_eq!(vec.get(50), Some(&50));
    }

    #[test]
    fn concat_appends_other_vector() {
        let mut left = Vector::new();
        let mut right = Vector::new();

        for i in 0..20 {
            left = left.push_back(i);
        }
        for i in 20..40 {
            right = right.push_back(i);
        }

        let combined = left.concat(&right);
        assert_eq!(combined.len(), 40);
        assert_eq!(combined.get(0), Some(&0));
        assert_eq!(combined.get(39), Some(&39));
        // Ensure original vectors remain unchanged.
        assert_eq!(left.len(), 20);
        assert_eq!(right.len(), 20);
    }

    #[test]
    fn iterator_traverses_sequentially() {
        let mut vec = Vector::new();
        for i in 0..50 {
            vec = vec.push_back(i);
        }

        let collected: Vec<_> = vec.iter().copied().collect();
        assert_eq!(collected, (0..50).collect::<Vec<_>>());

        let double_ended: Vec<_> = vec.iter().rev().copied().collect();
        assert_eq!(double_ended, (0..50).rev().collect::<Vec<_>>());
    }
}
