//! Provides [`IdSet`], a set type featuring hashless *O*(1) insertion and
//! removal.

use std::fmt;

/// An unordered collection of elements with *O*(1) insertion and removal.
///
/// `IdSet`s are pseudo-ordered collections of elements offering fast, *O*(1)
/// insertion and removal. Each `IdSet` is backed by a vector of optional
/// elements, which allows for fast, hashless access into a contiguous slice of
/// memory. In contrast, [`HashSet`][std::collections::HashSet]s from the
/// standard library also satisfy these properties in principle, but have worse
/// performance and less general applicability due to the need for hashing.
///
/// The trade-off is that elements in the store are generally no longer
/// accessible by value. Instead, elements are assigned integer ID values on
/// insertion, which are later used to index into the `IdSet`. `IdSet`s also
/// have worse memory efficiency when values are removed from the middle of the
/// range of assigned IDs.
///
/// # Example
/// ```
/// use tensor_net::id_set::IdSet;
///
/// let mut values: IdSet<char> = IdSet::new();
/// let id_a = values.insert('a');
/// let id_b = values.insert('b');
/// let id_c = values.insert('c');
/// assert_eq!((id_a, id_b, id_c), (0, 1, 2));
/// println!("{}", values); // [a, b, c]
/// println!("{:#}", values); // [0: a, 1: b, 2: c]
///
/// assert_eq!(values.get(0), Some(&'a'));
/// assert_eq!(values.remove(1), Some('b'));
///
/// let id_d = values.insert('d');
/// assert_eq!(id_d, 1);
/// ```
#[derive(Clone, Debug)]
pub struct IdSet<T> {
    data: Vec<Option<T>>,
    free: Vec<usize>,
    count: usize,
}

impl<T> PartialEq for IdSet<T>
where T: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(l, r)| l == r)
    }
}

impl<T> Eq for IdSet<T>
where T: Eq
{ }

impl<T> PartialOrd for IdSet<T>
where T: PartialOrd
{
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        let mut l_iter = self.iter();
        let mut r_iter = rhs.iter();
        loop {
            match (l_iter.next(), r_iter.next()) {
                (None, None) => { return Some(Ordering::Equal); },
                (None, Some(_)) => { return Some(Ordering::Less); },
                (Some(_), None) => { return Some(Ordering::Greater); },
                (Some(l), Some(r)) =>
                    match l.partial_cmp(r) {
                        Some(Ordering::Equal) => { continue; },
                        None => { return None; },
                        x => { return x; },
                    },
            }
        }
    }
}

impl<T> Ord for IdSet<T>
where T: Ord
{
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let mut l_iter = self.iter();
        let mut r_iter = rhs.iter();
        loop {
            match (l_iter.next(), r_iter.next()) {
                (None, None) => { return Ordering::Equal; },
                (None, Some(_)) => { return Ordering::Less; },
                (Some(_), None) => { return Ordering::Greater; },
                (Some(l), Some(r)) =>
                    match l.cmp(r) {
                        Ordering::Equal => { continue; },
                        x => { return x; },
                    },
            }
        }
    }
}

impl<T> Default for IdSet<T> {
    fn default() -> Self { Self::new() }
}

impl<T> IdSet<T> {
    /// Create a new, empty `IdSet`.
    pub fn new() -> Self {
        Self { data: Vec::new(), free: Vec::new(), count: 0 }
    }

    /// Create a new `IdSet` with placeholders for values at IDs `0 .. n`. As
    /// new values are inserted, IDs will be yielded starting from `0`.
    /// ```
    /// # use tensor_net::id_set::IdSet;
    /// // create a new IdSet with space pre-allocated for 10 items
    /// let mut values: IdSet<char> = IdSet::prealloc(10);
    ///
    /// // IDs are yielded in the same order they would for a new, empty IdSet
    /// assert_eq!(values.insert('a'), 0);
    /// assert_eq!(values.insert('b'), 1);
    /// // ...
    /// ```
    pub fn prealloc(n: usize) -> Self {
        let free: Vec<usize> = (0..n).rev().collect();
        let data: Vec<Option<T>> = free.iter().map(|_| None).collect();
        Self { data, free, count: 0 }
    }

    /// Return the minimum inhabited ID.
    pub fn min_id(&self) -> Option<usize> {
        self.data.iter().enumerate()
            .find(|(_, mb_item)| mb_item.is_some())
            .map(|(id, _)| id)
    }

    /// Return the maximum inhabited ID.
    pub fn max_id(&self) -> Option<usize> {
        self.data.iter().enumerate()
            .rfind(|(_, mb_item)| mb_item.is_some())
            .map(|(id, _)| id)
    }

    /// Return the minimum and maximum inhabited IDs.
    ///
    /// This is a synonym for `self.min_id().zip(self.max_id())`.
    pub fn id_bounds(&self) -> Option<(usize, usize)> {
        let k_min = self.min_id()?;
        let k_max = self.max_id()?;
        Some((k_min, k_max))
    }

    /// Like [`id_bounds`][Self::id_bounds], but returning the minimum and
    /// maximum IDs as a range.
    pub fn id_range(&self) -> Option<std::ops::RangeInclusive<usize>> {
        let k_min = self.min_id()?;
        let k_max = self.max_id()?;
        Some(k_min ..= k_max)
    }

    /// Return the number of elements.
    pub fn len(&self) -> usize { self.count }

    /// Return `true` if `self` contains no elements.
    pub fn is_empty(&self) -> bool { self.count == 0 }

    /// Return a fresh ID that will be associated with the next item to be
    /// inserted.
    pub fn next_id(&self) -> usize {
        if let Some(id) = self.free.last() {
            *id
        } else {
            self.data.len()
        }
    }

    /// Insert a new value and return its ID.
    pub fn insert(&mut self, val: T) -> usize {
        if let Some(id) = self.free.pop() {
            let _ = self.data[id].insert(val);
            self.count += 1;
            id
        } else {
            self.data.push(Some(val));
            self.count += 1;
            self.data.len() - 1
        }
    }

    /// Remove and return the value at `id`, if it exists.
    pub fn remove(&mut self, id: usize) -> Option<T> {
        let mb_val: Option<T> =
            self.data.get_mut(id)
            .and_then(|mb_item| mb_item.take());
        if mb_val.is_some() {
            self.free.push(id);
            self.count -= 1;
        }
        mb_val
    }

    /// Remove all items and placeholders at IDs greater than `len - 1`.
    pub fn truncate(&mut self, len: usize) {
        let to_drop: usize =
            self.data.iter().skip(len)
            .filter(|mb_item| mb_item.is_some())
            .count();
        self.data.truncate(len);
        self.free = self.free.drain(..).filter(|id| *id < len).collect();
        self.count -= to_drop;
    }

    /// Remove all placeholders at IDs greater than the maximum inhabited ID.
    pub fn shrink_to_fit(&mut self) {
        self.truncate(self.max_id().map(|imax| imax + 1).unwrap_or(0));
    }

    /// Remove all placeholders at IDs greater than `len - 1` ot the maximum
    /// inhabited ID, whichever is greater.
    pub fn shrink_to(&mut self, len: usize) {
        let size = self.max_id().map(|imax| imax + 1).unwrap_or(0).max(len);
        self.truncate(size);
    }

    /// Drop all items, replacing with placeholders.
    ///
    /// As new values are inserted after calling this method, IDs will be
    /// yielded starting at `0`.
    pub fn clear(&mut self) {
        self.data.iter_mut()
            .for_each(|mb_item| { *mb_item = None; });
        self.free = (0..self.data.len()).collect();
        self.count = 0;
    }

    /// Return a reference to the value at `id`, if it exists.
    pub fn get(&self, id: usize) -> Option<&T> {
        self.data.get(id).and_then(|mb_item| mb_item.as_ref())
    }

    /// Return a mutable reference to the value at `id`, if it exists.
    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        self.data.get_mut(id).and_then(|mb_item| mb_item.as_mut())
    }

    /// Return `true` if `self` contains a value at `id`.
    pub fn contains_id(&self, id: usize) -> bool {
        self.data.get(id).is_some_and(|mb_item| mb_item.is_some())
    }

    /// Return an iterator over references to all items.
    ///
    /// The iterator item type is `&T`.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { iter: self.data.iter(), len: self.count }
    }

    /// Return an iterator over mutable references to all items.
    ///
    /// The iterator item type is `&mut T`.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut { iter: self.data.iter_mut(), len: self.count }
    }

    /// Return a draining iterator over all items in a given ID range.
    ///
    /// As items are yielded from the iterator, they are removed from `self` and
    /// replaced with placeholders.
    ///
    /// The iterator item type is `T`.
    #[allow(clippy::iter_skip_zero)]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where R: std::ops::RangeBounds<usize>
    {
        use std::ops::Bound;
        let id_min =
            match range.start_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) =>
                    if *i == usize::MAX {
                        return Drain {
                            iter: self.data.iter_mut().skip(0).take(0),
                            len: 0,
                        };
                    } else {
                        *i + 1
                    },
                Bound::Unbounded => 0,
            };
        let id_max =
            match range.end_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) =>
                    if *i == 0 {
                        return Drain {
                            iter: self.data.iter_mut().skip(0).take(0),
                            len: 0,
                        };
                    } else {
                        (*i - 1).min(self.data.len() - 1)
                    },
                Bound::Unbounded => self.data.len() - 1,
            };
        if id_min > id_max {
            return Drain {
                iter: self.data.iter_mut().skip(0).take(0),
                len: 0,
            };
        }
        let len =
            self.data[id_min..=id_max].iter()
            .filter(|mb_item| mb_item.is_some())
            .count();
        let iter =
            self.data.iter_mut()
            .skip(id_min)
            .take(id_max + 1 - id_min);
        Drain { iter, len }
    }

    /// Return an iterator over references to all items with their IDs.
    ///
    /// The iterator item type is `(usize, &T)`.
    pub fn iter_id(&self) -> IterId<'_, T> {
        IterId { iter: self.data.iter().enumerate(), len: self.count }
    }

    /// Return an iterator over mutable references to all items with their IDs.
    ///
    /// The iterator item type is `(usize, &mut T)`.
    pub fn iter_mut_id(&mut self) -> IterMutId<'_, T> {
        IterMutId { iter: self.data.iter_mut().enumerate(), len: self.count }
    }

    /// Return an iterator over all items with their IDs, consuming `self`.
    ///
    /// The iterator item type is `(usize, T)`.
    pub fn into_iter_id(self) -> IntoIterId<T> {
        IntoIterId { iter: self.data.into_iter().enumerate(), len: self.count }
    }

    /// Return a draining iterator over all items with IDs in a given ID range.
    ///
    /// As items are yielded from the iterator, they are removed from `self` and
    /// replaced with placeholders.
    ///
    /// The iterator item type is `(usize, T)`.
    #[allow(clippy::iter_skip_zero)]
    pub fn drain_id<R>(&mut self, range: R) -> DrainId<'_, T>
    where R: std::ops::RangeBounds<usize>
    {
        use std::ops::Bound;
        let id_min =
            match range.start_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) =>
                    if *i == usize::MAX {
                        let iter =
                            self.data.iter_mut()
                            .enumerate()
                            .skip(0)
                            .take(0);
                        return DrainId { iter, len: 0 };
                    } else {
                        *i + 1
                    },
                Bound::Unbounded => 0,
            };
        let id_max =
            match range.end_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) =>
                    if *i == 0 {
                        let iter =
                            self.data.iter_mut()
                            .enumerate()
                            .skip(0)
                            .take(0);
                        return DrainId { iter, len: 0 };
                    } else {
                        (*i - 1).min(self.data.len() - 1)
                    },
                Bound::Unbounded => self.data.len() - 1,
            };
        if id_min > id_max {
            return DrainId {
                iter: self.data.iter_mut().enumerate().skip(0).take(0),
                len: 0,
            };
        }
        let len =
            self.data[id_min..=id_max].iter()
            .filter(|mb_item| mb_item.is_some())
            .count();
        let iter =
            self.data.iter_mut()
            .enumerate()
            .skip(id_min)
            .take(id_max + 1 - id_min);
        DrainId { iter, len }
    }
}

impl<T> IdSet<T>
where T: PartialEq
{
    /// Return `true` if `self` contains a value equal to `target`.
    pub fn contains(&self, target: &T) -> bool {
        self.data.iter()
            .any(|mb_item| mb_item.as_ref().is_some_and(|item| item == target))
    }

    /// Return the smallest ID assigned to a value equal to `target`, if one
    /// exists.
    pub fn find(&self, target: &T) -> Option<usize> {
        self.data.iter().enumerate()
            .find_map(|(id, mb_item)| {
                mb_item.as_ref()
                    .is_some_and(|item| item == target).then_some(id)
            })
    }
}

/// Iterator over references to `IdSet` elements.
///
/// The iterator item type is `&'a T`.
#[derive(Clone, Debug)]
pub struct Iter<'a, T> {
    iter: std::slice::Iter<'a, Option<T>>,
    len: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|mb_item| mb_item.is_some())
            .and_then(|mb_item| { self.len -= 1; mb_item.as_ref() })
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|mb_item| mb_item.is_some())
            .and_then(|mb_item| { self.len -= 1; mb_item.as_ref() })
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for Iter<'_, T> { }

/// Iterator over mutable references to `IdSet` elements.
///
/// The iterator item type is `&'a mut T`.
#[derive(Debug)]
pub struct IterMut<'a, T> {
    iter: std::slice::IterMut<'a, Option<T>>,
    len: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|mb_item| mb_item.is_some())
            .and_then(|mb_item| { self.len -= 1; mb_item.as_mut() })
    }
}

impl<T> DoubleEndedIterator for IterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|mb_item| mb_item.is_some())
            .and_then(|mb_item| { self.len -= 1; mb_item.as_mut() })
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for IterMut<'_, T> { }

/// Iterator over `IdSet` elements.
///
/// The iterator item type is `T`.
#[derive(Debug)]
pub struct IntoIter<T> {
    iter: std::vec::IntoIter<Option<T>>,
    len: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|mb_item| mb_item.is_some())
            .and_then(|mb_item| { self.len -= 1; mb_item })
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|mb_item| mb_item.is_some())
            .and_then(|mb_item| { self.len -= 1; mb_item })
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for IntoIter<T> { }

/// Draining iterator for `IdSet`s.
///
/// The iterator item type is `T`.
#[derive(Debug)]
pub struct Drain<'a, T> {
    iter: std::iter::Take<std::iter::Skip<std::slice::IterMut<'a, Option<T>>>>,
    len: usize,
}

impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|mb_item| mb_item.is_some())
            .and_then(|mb_item| {
                self.len -= 1;
                mb_item.take()
            })
    }
}

impl<T> DoubleEndedIterator for Drain<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|mb_item| mb_item.is_some())
            .and_then(|mb_item| {
                self.len -= 1;
                mb_item.take()
            })
    }
}

impl<T> ExactSizeIterator for Drain<'_, T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for Drain<'_, T> { }

/// Iterator over references to `IdSet` elements with IDs.
///
/// The iterator item type is `(usize, &'a T)`.
#[derive(Clone, Debug)]
pub struct IterId<'a, T> {
    iter: std::iter::Enumerate<std::slice::Iter<'a, Option<T>>>,
    len: usize,
}

impl<'a, T> Iterator for IterId<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -= 1;
                mb_item.as_ref().map(|item| (id, item))
            })
    }
}

impl<T> DoubleEndedIterator for IterId<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -= 1;
                mb_item.as_ref().map(|item| (id, item))
            })
    }
}

impl<T> ExactSizeIterator for IterId<'_, T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for IterId<'_, T> { }

/// Iterator over mutable references to `IdSet` elements with IDs.
///
/// The iterator item type is `(usize, &'a mut T)`.
#[derive(Debug)]
pub struct IterMutId<'a, T> {
    iter: std::iter::Enumerate<std::slice::IterMut<'a, Option<T>>>,
    len: usize,
}

impl<'a, T> Iterator for IterMutId<'a, T> {
    type Item = (usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -= 1;
                mb_item.as_mut().map(|item| (id, item))
            })
    }
}

impl<T> DoubleEndedIterator for IterMutId<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -= 1;
                mb_item.as_mut().map(|item| (id, item))
            })
    }
}

impl<T> ExactSizeIterator for IterMutId<'_, T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for IterMutId<'_, T> { }

/// Iterator over `IdSet` elements with IDs.
///
/// The iterator item type is `(usize, T)`.
#[derive(Debug)]
pub struct IntoIterId<T> {
    iter: std::iter::Enumerate<std::vec::IntoIter<Option<T>>>,
    len: usize,
}

impl<T> Iterator for IntoIterId<T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -= 1;
                mb_item.map(|item| (id, item))
            })
    }
}

impl<T> DoubleEndedIterator for IntoIterId<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -=1;
                mb_item.map(|item| (id, item))
            })
    }
}

impl<T> ExactSizeIterator for IntoIterId<T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for IntoIterId<T> { }

/// Draining iterator for `IdSet`s, with IDs.
///
/// The Iterator item type is `(usize, T)`.
#[derive(Debug)]
pub struct DrainId<'a, T> {
    iter: std::iter::Take<std::iter::Skip<std::iter::Enumerate<std::slice::IterMut<'a, Option<T>>>>>,
    len: usize,
}

impl<T> Iterator for DrainId<'_, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -= 1;
                mb_item.take().map(|item| (id, item))
            })
    }
}

impl<T> DoubleEndedIterator for DrainId<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|(_, mb_item)| mb_item.is_some())
            .and_then(|(id, mb_item)| {
                self.len -= 1;
                mb_item.take().map(|item| (id, item))
            })
    }
}

impl<T> ExactSizeIterator for DrainId<'_, T> {
    fn len(&self) -> usize { self.len }
}

impl<T> std::iter::FusedIterator for DrainId<'_, T> { }

impl<'a, T> IntoIterator for &'a IdSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, T> IntoIterator for &'a mut IdSet<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

impl<T> IntoIterator for IdSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { iter: self.data.into_iter(), len: self.count }
    }
}

impl<T> FromIterator<T> for IdSet<T> {
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = T>
    {
        let data: Vec<Option<T>> = iter.into_iter().map(Some).collect();
        let free: Vec<usize> = Vec::new();
        let count = data.len();
        Self { data, free, count }
    }
}

macro_rules! impl_fmt {
    ( $trait:path ) => {
        impl<T> $trait for IdSet<T>
        where T: $trait
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let n = self.len();
                write!(f, "[")?;
                if f.alternate() {
                    for (k, (id, val)) in self.iter_id().enumerate() {
                        write!(f, "{}: ", id)?;
                        val.fmt(f)?;
                        if k < n - 1 { write!(f, ", ")?; }
                    }
                } else {
                    for (k, val) in self.iter().enumerate() {
                        val.fmt(f)?;
                        if k < n - 1 { write!(f, ", ")?; }
                    }
                }
                write!(f, "]")?;
                Ok(())
            }
        }
    }
}
impl_fmt!(fmt::Display);
impl_fmt!(fmt::LowerExp);
impl_fmt!(fmt::UpperExp);
impl_fmt!(fmt::Octal);
impl_fmt!(fmt::LowerHex);
impl_fmt!(fmt::UpperHex);
impl_fmt!(fmt::Binary);
impl_fmt!(fmt::Pointer);

