#![allow(rustdoc::broken_intra_doc_links)]

//! An N-dimensional array of data with shape determined by a set of numerical
//! indices.
//!
//! A [`Tensor`] is multi-linear algebraic object that can be seen as the
//! generalization of linear or bi-linear objects such as vectors and matrices.
//! Like vectors and matrices, tensors represent collections of objects that are
//! accessed by supplying a number of index values (i.e. 1 for a vector, 2 for a
//! matrix), but allow for more than 2.
//!
//! <blockquote>
//!   <p style="font-size:20px">
//!     <i>T</i><sub><i>a</i><sub>1</sub>,...,<i>a</i><sub><i>N</i></sub></sub>
//!   </p>
//! </blockquote>
//!
//! Linear operations are generalized as well. The usual matrix-matrix,
//! matrix-vector, vector-matrix, and vector-vector "dot" products are
//! generalized to the tensor contraction over a subset of two tensors' indices,
//! where the result is calculated by summing over the values of those indices
//! and leaving all others untouched.
//!
//! <blockquote>
//!   <p style="font-size:20px">
//!     <i>C</i><sub>
//!       <i>a</i><sub>1</sub>,...,<i>a</i><sub><i>N</i></sub>,
//!       <i>b</i><sub>1</sub>,...,<i>b</i><sub><i>M</i></sub>
//!     </sub>
//!       = Σ<sub><i>α</i><sub>1</sub>,...,<i>α</i><sub><i>D</i></sub></sub> [
//!         <i>A</i><sub>
//!           <i>a</i><sub>1</sub>,...,<i>a</i><sub><i>N</i></sub>,
//!           <i>α</i><sub>1</sub>,...,<i>α</i><sub><i>D</i></sub>
//!         </sub>
//!         × <i>B</i><sub>
//!           <i>b</i><sub>1</sub>,...,<i>b</i><sub><i>M</i></sub>,
//!           <i>α</i><sub>1</sub>,...,<i>α</i><sub><i>D</i></sub>
//!         </sub>
//!       ]
//!   </p>
//! </blockquote>
//!
//! Likewise, the Kronecker product is generalized to the tensor product.
//!
//! <blockquote>
//!   <p style="font-size:20px">
//!     <i>C</i><sub>
//!       <i>a</i><sub>1</sub>,...,<i>a</i><sub><i>N</i></sub>,
//!       <i>b</i><sub>1</sub>,...,<i>b</i><sub><i>M</i></sub>
//!     </sub>
//!       = <i>A</i><sub>
//!         <i>a</i><sub>1</sub>,...,<i>a</i><sub><i>N</i></sub>
//!       </sub>
//!       × <i>B</i><sub>
//!         <i>b</i><sub>1</sub>,...,<i>b</i><sub><i>M</i></sub>
//!       </sub>
//!   </p>
//! </blockquote>
//!
//! ```
//! use tensor_net::tensor::{ Idx, Tensor }; // see the Idx trait
//!
//! #[derive(Copy, Clone, Debug, PartialEq, Eq)]
//! enum Index { A, B, C }
//!
//! impl Idx for Index {
//!     fn dim(&self) -> usize {
//!         match self {
//!             Self::A => 3,
//!             Self::B => 4,
//!             Self::C => 5,
//!         }
//!     }
//! }
//!
//! impl std::fmt::Display for Index {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         write!(f, "{:?}", self)
//!     }
//! }
//!
//! let a = Tensor::new([Index::A, Index::B], |_| 1.0).unwrap();
//! println!("{}", a);
//! // [[1, 1, 1, 1],
//! //  [1, 1, 1, 1],
//! //  [1, 1, 1, 1]] { A, B }
//!
//! let b = Tensor::new([Index::B, Index::C], |_| 2.0).unwrap();
//! println!("{}", b);
//! // [[2, 2, 2, 2, 2],
//! //  [2, 2, 2, 2, 2],
//! //  [2, 2, 2, 2, 2],
//! //  [2, 2, 2, 2, 2]] { B, C }
//!
//! let c = a.contract(b).unwrap(); // C_{a,c} = A_{a,b} B_{b,c}
//! println!("{}", c);
//! // [[8, 8, 8, 8, 8],
//! //  [8, 8, 8, 8, 8],
//! //  [8, 8, 8, 8, 8]] { A, C }
//! ```

use std::{ collections::VecDeque, fmt };
use nalgebra::{ self as na, ComplexField };
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    /// Returned when attempting to create a new tensor with duplicate indices.
    #[error("error in tensor creation: duplicate indices")]
    DuplicateIndices,

    /// Returned when attempting to create a new tensor with at least one index
    /// that has zero dimension.
    #[error("error in tensor creation: encountered a zero-dimensional index")]
    ZeroDimIndex,

    /// Returned when attempting to create a new tensor from a pre-existing
    /// collection of elements and the provided indices have non-matching total
    /// dimension.
    #[error("error in tensor creation: non-matching indices and number of elements")]
    IncompatibleNumComplexFields,

    /// Returned when a tensor add is attempted between two tensors with
    /// incompatible indices.
    #[error("error in tensor add: non-matching indices")]
    IncompatibleIndicesAdd,

    /// Returned when a tensor sub is attempted between two tensors with
    /// incompatible indices.
    #[error("error in tensor sub: non-matching indices")]
    IncompatibleIndicesSub,
}
use TensorError::*;
pub type TensorResult<T> = Result<T, TensorError>;

/// Describes a tensor index.
///
/// Generally, a tensor network will to distinguish between the available
/// indices over which its tensors can be contracted. There are two ways to
/// implement this. The first is by using a "static" representation, where all
/// available indices are encoded directly in the structure of the type
/// implementing this trait:
/// ```ignore
/// #[derive(Clone, Debug, PartialEq, Eq)]
/// enum Index { A, B, C, /* ... */ }
///
/// impl Idx for Index {
///     fn dim(&self) -> usize {
///         match self {
///             Self::A => /* dimension for A */,
///             Self::B => /* dimension for B */,
///             Self::C => /* dimension for C */,
///             // ...
///         }
///     }
/// }
/// ```
/// This representation is safer because it encodes relevant information in
/// degrees of freedom that are immutable at runtime (i.e. the total number of
/// possible distinct indices is fixed), but cannot be used in a tensor network
/// that dynamically adds or removes arbitrary tensors and indices.
///
/// The second is by using a "dynamic" representation, where the available
/// indices and their dimensions are left to be encoded as arbitrary fields of
/// the type:
/// ```ignore
/// #[derive(Clone, Debug, PartialEq, Eq)]
/// struct Index(String, usize);
///
/// impl Idx for Index {
///     fn dim(&self) -> usize { self.1 }
///
///     fn label(&self) -> String { self.0.clone() }
/// }
/// ```
/// This representation allows for dynamic addition of arbitrary indices to a
/// tensor network, but comes with a memory overhead and is somewhat less safe
/// than static representations because there are more runtime-mutable degrees
/// of freedom.
///
/// Ultimately, static representations should be preferred where possible.
pub trait Idx: Clone + PartialEq + std::fmt::Debug {
    /// Return the number of values the index can take.
    ///
    /// This value must never be zero.
    fn dim(&self) -> usize;

    /// Return an identifying label for the index. This method is used only for
    /// printing purposes.
    ///
    /// The default implementation renders `self` using `Debug`.
    fn label(&self) -> String { format!("{self:?}") }
}

/// A dynamically dimensioned tensor index type.
///
/// This type should be used only when a tensor network needs to be dynamically
/// modifiable, otherwise static implementations should be preferred (see
/// [`Idx`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynIdx(
    /// Identifier label.
    pub std::sync::Arc<str>,
    /// Index dimension.
    pub usize,
);

impl Idx for DynIdx {
    fn dim(&self) -> usize { self.1 }
}

impl<T> From<(T, usize)> for DynIdx
where std::sync::Arc<str>: From<T>
{
    fn from(x: (T, usize)) -> Self {
        let (label, dim) = x;
        Self(label.into(), dim)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ReshapeDir {
    // move indices from row to column
    LR(usize),
    // move indices from column to row
    RL(usize),
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Idxs<T> {
    rowdim: usize,
    row: VecDeque<T>,
    coldim: usize,
    col: VecDeque<T>,
}

impl<T> Idxs<T> {
    fn len(&self) -> usize { self.row.len() + self.col.len() }

    fn is_empty(&self) -> bool { self.row.is_empty() && self.col.is_empty() }

    fn total_dim(&self) -> usize { self.rowdim * self.coldim }

    fn shape(&self) -> (usize, usize) { (self.rowdim, self.coldim) }

    fn iter(&self) -> IdxsIter<'_, T> {
        IdxsIter { source: self.row.iter().chain(self.col.iter()) }
    }

    fn iter_mut(&mut self) -> IdxsIterMut<'_, T> {
        IdxsIterMut { source: self.row.iter_mut().chain(self.col.iter_mut()) }
    }

    fn transpose(&mut self) { std::mem::swap(&mut self.row, &mut self.col); }
}

impl<T> Idxs<T>
where T: PartialEq
{
    fn contains(&self, target: &T) -> bool {
        self.row.contains(target) || self.col.contains(target)
    }

    fn all_match(&self, other: &Self) -> bool {
        self.len() == other.len()
        && self.row.iter().chain(self.col.iter()).all(|idx| other.contains(idx))
    }

    // return a reference to the first common index, where the reference is to
    // the index in `self`
    fn find_common(&self, other: &Self) -> Option<&T> {
        self.row.iter().chain(self.col.iter())
            .find(|idx| other.contains(idx))
    }

    // find the reshape needed to move a target index to the right-most spot on
    // the row side
    //
    // returns `None` if the target index is in the right-most spot on the
    // column side
    fn find_reshape(&self, target: &T) -> Option<ReshapeDir> {
        if self.col.back().is_some_and(|idx| idx == target) {
            return None;
        }
        let find_row =
            self.row.iter().rev().enumerate()
            .find_map(|(r, idx)| (idx == target).then_some(r));
        if let Some(r) = find_row {
            return Some(ReshapeDir::LR(r));
        }
        let find_col =
            self.col.iter().enumerate()
            .find_map(|(c, idx)| (idx == target).then_some(c));
        if let Some(c) = find_col {
            return Some(ReshapeDir::RL(c + 1));
        }
        None
    }
}

impl<T> Idxs<T>
where T: Idx
{
    // pop the rightmost index on the column side
    fn pop(&mut self) -> Option<T> {
        if let Some(last) = self.col.pop_back() {
            self.coldim /= last.dim();
            Some(last)
        } else {
            None
        }
    }

    // push a new index onto the right-most position on the column side
    fn push(&mut self, idx: T) {
        self.coldim *= idx.dim();
        self.col.push_back(idx);
    }

    // move the `n` right-most row indices to the column side from the left
    //
    // returns the shape of the matrix after the move
    fn reshape_right(&mut self, n: usize) -> (usize, usize) {
        for _ in 0..n {
            if let Some(idx) = self.row.pop_back() {
                let dim = idx.dim();
                self.rowdim /= dim;
                self.coldim *= dim;
                self.col.push_front(idx);
            } else {
                return self.shape();
            }
        }
        self.shape()
    }

    // move the `n` left-most column indices to the row side from the right
    //
    // returns the shape of the matrix after the move
    fn reshape_left(&mut self, n: usize) -> (usize, usize) {
        for _ in 0..n {
            if let Some(idx) = self.col.pop_front() {
                let dim = idx.dim();
                self.coldim /= dim;
                self.rowdim *= dim;
                self.row.push_back(idx);
            } else {
                return self.shape();
            }
        }
        self.shape()
    }

    fn reshape(&mut self, reshape: ReshapeDir) -> (usize, usize) {
        match reshape {
            ReshapeDir::LR(n) => self.reshape_right(n),
            ReshapeDir::RL(n) => self.reshape_left(n),
        }
    }

    // move all indices to the row side
    fn make_vector(&mut self) {
        self.col.drain(..)
            .for_each(|idx| { self.row.push_back(idx); });
        self.rowdim *= self.coldim;
    }

    #[allow(dead_code)]
    fn outer_prod(mut self, mut rhs: Self) -> Self {
        self.make_vector();
        rhs.make_vector();
        self.col = rhs.row;
        self.coldim = rhs.rowdim;
        self
    }

    fn contract(mut self, rhs: Self) -> Self {
        self.coldim = rhs.coldim;
        self.col = rhs.col;
        self
    }

    // find the reshape that would make the row and column dimensions
    // approximately equal
    //
    // a longer row dimension is preferred to a longer column dimension when a
    // square reshaping can't be found
    fn find_reshape_square(&self) -> Option<ReshapeDir> {
        let square_sidelen: usize =
            (self.total_dim() as f64).sqrt().round() as usize;
        let square_crossing =
            self.iter()
            .scan(1, |acc, idx| { *acc *= idx.dim(); Some(*acc) })
            .enumerate()
            .find_map(|(k, rowdim)| (rowdim >= square_sidelen).then_some(k));
        if let Some(k_cross) = square_crossing {
            // square_crossing being Some implies that the total number of
            // indices is greater than 0
            let nrow = self.row.len();
            if nrow > k_cross + 1 {
                Some(ReshapeDir::LR(nrow - 1 - k_cross))
            } else {
                Some(ReshapeDir::RL(k_cross + 1 - nrow))
            }
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct IdxsIter<'a, T> {
    source: std::iter::Chain<
        std::collections::vec_deque::Iter<'a, T>,
        std::collections::vec_deque::Iter<'a, T>,
    >
}

impl<'a, T> Iterator for IdxsIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> { self.source.next() }
}

impl<T> DoubleEndedIterator for IdxsIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> { self.source.next_back() }
}

impl<T> std::iter::FusedIterator for IdxsIter<'_, T> { }

#[derive(Debug)]
struct IdxsIterMut<'a, T> {
    source: std::iter::Chain<
        std::collections::vec_deque::IterMut<'a, T>,
        std::collections::vec_deque::IterMut<'a, T>,
    >
}

impl<'a, T> Iterator for IdxsIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> { self.source.next() }
}

impl<T> DoubleEndedIterator for IdxsIterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> { self.source.next_back() }
}

impl<T> std::iter::FusedIterator for IdxsIterMut<'_, T> { }

#[derive(Debug)]
struct IdxsIntoIter<T> {
    source: std::iter::Chain<
        std::collections::vec_deque::IntoIter<T>,
        std::collections::vec_deque::IntoIter<T>,
    >
}

impl<T> Iterator for IdxsIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> { self.source.next() }
}

impl<T> DoubleEndedIterator for IdxsIntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> { self.source.next_back() }
}

impl<T> std::iter::FusedIterator for IdxsIntoIter<T> { }

impl<'a, T> IntoIterator for &'a Idxs<T> {
    type Item = &'a T;
    type IntoIter = IdxsIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, T> IntoIterator for &'a mut Idxs<T> {
    type Item = &'a mut T;
    type IntoIter = IdxsIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

impl<T> IntoIterator for Idxs<T> {
    type Item = T;
    type IntoIter = IdxsIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IdxsIntoIter { source: self.row.into_iter().chain(self.col) }
    }
}

#[derive(Clone, Debug)]
enum IndicesData<'a, T> {
    Scalar,
    Tensor(IdxsIter<'a, T>),
}

#[derive(Clone, Debug)]
pub struct Indices<'a, T>(IndicesData<'a, T>);

impl<'a, T> Iterator for Indices<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IndicesData::Scalar => None,
            IndicesData::Tensor(idxs) => idxs.next(),
        }
    }
}

impl<T> DoubleEndedIterator for Indices<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IndicesData::Scalar => None,
            IndicesData::Tensor(idxs) => idxs.next_back(),
        }
    }
}

impl<T> std::iter::FusedIterator for Indices<'_, T> { }

#[derive(Debug)]
enum IndicesMutData<'a, T> {
    Scalar,
    Tensor(IdxsIterMut<'a, T>),
}

#[derive(Debug)]
pub struct IndicesMut<'a, T>(IndicesMutData<'a, T>);

impl<'a, T> Iterator for IndicesMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IndicesMutData::Scalar => None,
            IndicesMutData::Tensor(idxs) => idxs.next(),
        }
    }
}

impl<T> DoubleEndedIterator for IndicesMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IndicesMutData::Scalar => None,
            IndicesMutData::Tensor(idxs) => idxs.next_back(),
        }
    }
}

impl<T> std::iter::FusedIterator for IndicesMut<'_, T> { }

// reorders the indices in two tensors to move all common indices to the column
// side (in the same order) and all non-common indices to the row side; no
// guarantees about the ordering of indices on either of the resulting row sides
fn sort_idx_common<T, A>(
    mut idxs_a: Idxs<T>,
    mut a: na::DMatrix<A>,
    mut idxs_b: Idxs<T>,
    mut b: na::DMatrix<A>,
) -> (Idxs<T>, na::DMatrix<A>, Idxs<T>, na::DMatrix<A>)
where
    T: Idx,
    A: na::Scalar,
{
    use na::{ Const, Dyn };
    let n = a.len();
    a = a.reshape_generic(Dyn(n), Dyn(1));
    let n = b.len();
    b = b.reshape_generic(Dyn(n), Dyn(1));
    let mut sorted_dim: usize = 1;
    let mut nonsorted_dim: usize;
    let mut idxbuf_a: Vec<T> = Vec::new();
    let mut idxbuf_b: Vec<T> = Vec::new();
    while let Some(com) = idxs_a.find_common(&idxs_b) {
        let d = com.dim();
        let mb_reshape_a = idxs_a.find_reshape(com);
        let mb_reshape_b = idxs_b.find_reshape(com);

        if let Some(reshape) = mb_reshape_a {
            let (m, n) = idxs_a.reshape(reshape);
            for mut nonsorted in a.column_iter_mut() {
                let idx_adj =
                    nonsorted.clone_owned()
                    .reshape_generic(Dyn(m), Dyn(n))
                    .transpose()
                    .reshape_generic(Dyn(m * n), Const::<1>);
                nonsorted.copy_from(&idx_adj);
            }
            nonsorted_dim = m * n;
            idxs_a.transpose();
        } else {
            nonsorted_dim = idxs_a.total_dim();
        }
        a = a.reshape_generic(Dyn(nonsorted_dim / d), Dyn(d * sorted_dim));

        if let Some(reshape) = mb_reshape_b {
            let (m, n) = idxs_b.reshape(reshape);
            for mut nonsorted in b.column_iter_mut() {
                let idx_adj =
                    nonsorted.clone_owned()
                    .reshape_generic(Dyn(m), Dyn(n))
                    .transpose()
                    .reshape_generic(Dyn(m * n), Const::<1>);
                nonsorted.copy_from(&idx_adj);
            }
            nonsorted_dim = m * n;
            idxs_b.transpose();
        } else {
            nonsorted_dim = idxs_b.total_dim();
        }
        b = b.reshape_generic(Dyn(nonsorted_dim / d), Dyn(d * sorted_dim));

        sorted_dim *= d;
        idxbuf_a.push(idxs_a.pop().unwrap());
        idxbuf_b.push(idxs_b.pop().unwrap());
    }
    idxs_a.make_vector();
    idxbuf_a.into_iter().rev()
        .for_each(|idx| { idxs_a.push(idx); });
    idxs_b.make_vector();
    idxbuf_b.into_iter().rev()
        .for_each(|idx| { idxs_b.push(idx); });
    (idxs_a, a, idxs_b, b)
}

fn contract<T, A>(
    idxs_a: Idxs<T>,
    a: na::DMatrix<A>,
    idxs_b: Idxs<T>,
    b: na::DMatrix<A>,
) -> (Idxs<T>, na::DMatrix<A>)
where
    T: Idx,
    A: ComplexField,
{
    let (idxs_a, a, mut idxs_b, b) = sort_idx_common(idxs_a, a, idxs_b, b);
    idxs_b.transpose();
    let idxs_c = idxs_a.contract(idxs_b);
    let c = a * b.transpose();
    (idxs_c, c)
}

#[derive(Clone, Debug)]
enum TensorData<T, A> {
    Scalar(A),
    Tensor(Idxs<T>, na::DMatrix<A>),
}

impl<T, A> PartialEq for TensorData<T, A>
where
    T: PartialEq,
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => a == b,
            (Self::Tensor(idxs_a, a), Self::Tensor(idxs_b, b)) =>
                idxs_a.iter().zip(idxs_b.iter())
                    .all(|(idx_a, idx_b)| idx_a == idx_b)
                && a.iter().zip(b.iter()).all(|(ak, bk)| ak == bk),
            _ => false
        }
    }
}

impl<T, A> Eq for TensorData<T, A>
where
    T: Eq,
    A: Eq,
{ }

impl<T, A> TensorData<T, A>
where
    T: Idx,
    A: ComplexField,
{
    fn new<I, F>(indices: I, mut elems: F) -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        let indices: VecDeque<T> = indices.into_iter().collect();
        if !is_unique(&indices) { return Err(DuplicateIndices); }
        if indices.is_empty() {
            let scalar: A = elems(&[]);
            Ok(Self::Scalar(scalar))
        } else {
            let strides: Vec<usize> =
                indices.iter()
                .scan(1, |acc, idx| { *acc *= idx.dim(); Some(*acc) })
                .collect();
            let len = *strides.last().unwrap();
            if len == 0 { return Err(ZeroDimIndex); }
            let elem_iter =
                (0..len)
                .scan(
                    vec![0; indices.len()],
                    |idxs, k| { ndinc(idxs, &strides, k); Some(elems(idxs)) },
                );
            let data: na::DMatrix<A> =
                na::DMatrix::from_iterator(len, 1, elem_iter);
            let idxs = Idxs {
                rowdim: len, row: indices, coldim: 1, col: VecDeque::new() };
            let mut new = Self::Tensor(idxs, data);
            new.reshape_square();
            Ok(new)
        }
    }

    unsafe fn new_unchecked<I, F>(indices: I, mut elems: F) -> Self
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        let indices: VecDeque<T> = indices.into_iter().collect();
        if indices.is_empty() {
            let scalar: A = elems(&[]);
            Self::Scalar(scalar)
        } else {
            let strides: Vec<usize> =
                indices.iter()
                .scan(1, |acc, idx| { *acc *= idx.dim(); Some(*acc) })
                .collect();
            let len = *strides.last().unwrap();
            let elem_iter =
                (0..len)
                .scan(
                    vec![0; indices.len()],
                    |idxs, k| { ndinc(idxs, &strides, k); Some(elems(idxs)) },
                );
            let data: na::DMatrix<A> =
                na::DMatrix::from_iterator(len, 1, elem_iter);
            let idxs = Idxs {
                rowdim: len, row: indices, coldim: 1, col: VecDeque::new() };
            let mut new = Self::Tensor(idxs, data);
            new.reshape_square();
            new
        }
    }

    fn new_scalar(val: A) -> Self { Self::Scalar(val) }

    fn from_elems<I>(indices: I, elems: na::DVector<A>) -> TensorResult<Self>
    where I: IntoIterator<Item = T>
    {
        let indices: VecDeque<T> = indices.into_iter().collect();
        if !is_unique(&indices) { return Err(DuplicateIndices); }
        let total_dim: usize = indices.iter().map(|idx| idx.dim()).product();
        if total_dim == 0 { return Err(ZeroDimIndex); }
        if total_dim != elems.len() { return Err(IncompatibleNumComplexFields); }
        if indices.is_empty() {
            Ok(Self::Scalar(elems[0].clone()))
        } else {
            let idxs = Idxs {
                rowdim: total_dim,
                row: indices,
                coldim: 1,
                col: VecDeque::new()
            };
            let data = elems.reshape_generic(na::Dyn(total_dim), na::Dyn(1));
            let mut new = Self::Tensor(idxs, data);
            new.reshape_square();
            Ok(new)
        }
    }

    unsafe fn from_elems_unchecked<I>(indices: I, elems: na::DVector<A>) -> Self
    where I: IntoIterator<Item = T>
    {
        let indices: VecDeque<T> = indices.into_iter().collect();
        let total_dim: usize = indices.iter().map(|idx| idx.dim()).product();
        if indices.is_empty() {
            Self::Scalar(elems[0].clone())
        } else {
            let idxs = Idxs {
                rowdim: total_dim,
                row: indices,
                coldim: 1,
                col: VecDeque::new()
            };
            let data = elems.reshape_generic(na::Dyn(total_dim), na::Dyn(1));
            let mut new = Self::Tensor(idxs, data);
            new.reshape_square();
            new
        }
    }

    fn is_scalar(&self) -> bool { matches!(self, Self::Scalar(_)) }

    fn is_tensor(&self) -> bool { matches!(self, Self::Tensor(..)) }

    fn has_index(&self, index: &T) -> bool {
        match self {
            Self::Scalar(_) => false,
            Self::Tensor(idxs, _) => idxs.contains(index),
        }
    }

    fn rank(&self) -> usize {
        match self {
            Self::Scalar(_) => 0,
            Self::Tensor(idxs, _) => idxs.len(),
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Scalar(_) => Vec::new(),
            Self::Tensor(idxs, _) => idxs.iter().map(|idx| idx.dim()).collect(),
        }
    }

    fn indices(&self) -> Indices<'_, T> {
        match self {
            Self::Scalar(_) => Indices(IndicesData::Scalar),
            Self::Tensor(idxs, _) => Indices(IndicesData::Tensor(idxs.iter())),
        }
    }

    unsafe fn indices_mut(&mut self) -> IndicesMut<'_, T> {
        match self {
            Self::Scalar(_) => IndicesMut(IndicesMutData::Scalar),
            Self::Tensor(idxs, _) =>
                IndicesMut(IndicesMutData::Tensor(idxs.iter_mut())),
        }
    }

    fn reshape_square(&mut self) {
        match self {
            Self::Scalar(_) => { },
            Self::Tensor(idxs, data) => {
                if let Some(reshape) = idxs.find_reshape_square() {
                    let (m, n) = idxs.reshape(reshape);
                    let mut temp = na::dmatrix!();
                    std::mem::swap(data, &mut temp);
                    temp = temp.reshape_generic(na::Dyn(m), na::Dyn(n));
                    std::mem::swap(data, &mut temp);
                }
            },
        }
    }

    unsafe fn map_indices<F, U>(self, f: F) -> TensorData<U, A>
    where
        F: FnMut(T) -> U,
        U: Idx,
    {
        match self {
            Self::Scalar(a) => TensorData::Scalar(a),
            Self::Tensor(idxs, data) => {
                let indices: VecDeque<U> = idxs.into_iter().map(f).collect();
                let len: usize = indices.iter().map(|idx| idx.dim()).product();
                let idxs = Idxs {
                    rowdim: len, row: indices, coldim: 1, col: VecDeque::new() };
                let data = data.reshape_generic(na::Dyn(len), na::Dyn(1));
                TensorData::Tensor(idxs, data)
            },
        }
    }

    fn map<F, B>(&self, mut f: F) -> TensorData<T, B>
    where
        F: FnMut(&[usize], &A) -> B,
        B: ComplexField,
    {
        match self {
            Self::Scalar(a) => TensorData::Scalar(f(&[], a)),
            Self::Tensor(idxs, data) => {
                let strides: Vec<usize> =
                    idxs.iter()
                    .scan(1, |acc, idx| { *acc *= idx.dim(); Some(*acc) })
                    .collect();
                let len = *strides.last().unwrap();
                let elem_iter =
                    data.iter().enumerate()
                    .scan(
                        vec![0; idxs.len()],
                        |idxs, (k, a)| {
                            ndinc(idxs, &strides, k);
                            Some(f(idxs, a))
                        },
                    );
                let data: na::DMatrix<B> =
                    na::DMatrix::from_iterator(len, 1, elem_iter);
                let mut idxs = idxs.clone();
                idxs.make_vector();
                TensorData::Tensor(idxs, data)
            },
        }
    }

    fn map_inplace<F>(&mut self, mut f: F)
    where F: FnMut(&[usize], &A) -> A
    {
        match self {
            Self::Scalar(a) => { *a = f(&[], &*a); },
            Self::Tensor(idxs, data) => {
                let strides: Vec<usize> =
                    idxs.iter()
                    .scan(1, |acc, idx| { *acc *= idx.dim(); Some(*acc) })
                    .collect();
                data.iter_mut().enumerate()
                    .scan(
                        vec![0; idxs.len()],
                        |idxs, (k, a)| {
                            ndinc(idxs, &strides, k);
                            let a_new = f(idxs, &*a);
                            Some((a, a_new))
                        }
                    )
                    .for_each(|(a, a_new)| { *a = a_new; });
            },
        }
    }

    fn multiply(self, other: Self) -> Self {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => {
                Self::Scalar(a * b)
            },
            (Self::Scalar(a), Self::Tensor(idxs, mut b)) => {
                b.iter_mut().for_each(|bk| { *bk *= a.clone(); });
                Self::Tensor(idxs, b)
            },
            (Self::Tensor(idxs, mut a), Self::Scalar(b)) => {
                a.iter_mut().for_each(|ak| { *ak *= b.clone(); });
                Self::Tensor(idxs, a)
            },
            (Self::Tensor(idxs_a, a), Self::Tensor(idxs_b, b)) => {
                let (idxs_c, c) = contract(idxs_a, a, idxs_b, b);
                if idxs_c.is_empty() {
                    Self::Scalar(c[0].clone())
                } else {
                    Self::Tensor(idxs_c, c)
                }
            },
        }
    }

    fn add_checked(self, other: Self) -> TensorResult<Self> {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => {
                Ok(Self::Scalar(a + b))
            },
            (Self::Scalar(_), Self::Tensor(..)) => {
                Err(IncompatibleIndicesAdd)
            },
            (Self::Tensor(..), Self::Scalar(_)) => {
                Err(IncompatibleIndicesAdd)
            },
            (Self::Tensor(idxs_a, a), Self::Tensor(idxs_b, b)) => {
                if !idxs_a.all_match(&idxs_b) {
                    return Err(IncompatibleIndicesAdd);
                }
                let (idxs_a, a, _, b) = sort_idx_common(idxs_a, a, idxs_b, b);
                Ok(Self::Tensor(idxs_a, a + b))
            },
        }
    }

    fn sub_checked(self, other: Self) -> TensorResult<Self> {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => {
                Ok(Self::Scalar(a - b))
            },
            (Self::Scalar(_), Self::Tensor(..)) => {
                Err(IncompatibleIndicesAdd)
            },
            (Self::Tensor(..), Self::Scalar(_)) => {
                Err(IncompatibleIndicesAdd)
            },
            (Self::Tensor(idxs_a, a), Self::Tensor(idxs_b, b)) => {
                if !idxs_a.all_match(&idxs_b) {
                    return Err(IncompatibleIndicesAdd);
                }
                let (idxs_a, a, _, b) = sort_idx_common(idxs_a, a, idxs_b, b);
                Ok(Self::Tensor(idxs_a, a - b))
            },
        }
    }

    fn conj(&self) -> Self {
        match self {
            Self::Scalar(a) => Self::Scalar(a.clone().conjugate()),
            Self::Tensor(idxs, a) => Self::Tensor(idxs.clone(), a.conjugate()),
        }
    }

    fn into_conj(self) -> Self {
        match self {
            Self::Scalar(a) => Self::Scalar(a.conjugate()),
            Self::Tensor(idxs, mut a) => {
                a.conjugate_mut();
                Self::Tensor(idxs, a)
            },
        }
    }

    fn conj_mut(&mut self) {
        match self {
            Self::Scalar(a) => { *a = a.clone().conjugate(); },
            Self::Tensor(_, a) => { a.conjugate_mut(); },
        }
    }
}

impl<T, A> std::ops::Mul<TensorData<T, A>> for TensorData<T, A>
where
    T: Idx,
    A: ComplexField,
{
    type Output = TensorData<T, A>;

    fn mul(self, other: TensorData<T, A>) -> Self::Output {
        self.multiply(other)
    }
}

impl<T, A> std::ops::Add<TensorData<T, A>> for TensorData<T, A>
where
    T: Idx,
    A: ComplexField,
{
    type Output = TensorData<T, A>;

    fn add(self, other: TensorData<T, A>) -> Self::Output {
        match self.add_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

impl<T, A> std::ops::Sub<TensorData<T, A>> for TensorData<T, A>
where
    T: Idx,
    A: ComplexField,
{
    type Output = TensorData<T, A>;

    fn sub(self, other: TensorData<T, A>) -> Self::Output {
        match self.sub_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

// mostly copied from nalgebra source; modified slightly for reduced whitespace
macro_rules! impl_fmt {
    (
        $trait: path,
        $fmt_str_without_precision: expr,
        $fmt_str_with_precision: expr
    ) => {
        impl<T, A> $trait for TensorData<T, A>
        where
            T: fmt::Display,
            A: na::Scalar + $trait,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fn val_width<U>(val: &U, f: &mut fmt::Formatter<'_>) -> usize
                where U: na::Scalar + $trait
                {
                    match f.precision() {
                        Some(prec) =>
                            format!($fmt_str_with_precision, val, prec)
                            .chars()
                            .count(),
                        None =>
                            format!($fmt_str_without_precision, val)
                            .chars()
                            .count(),
                    }
                }

                match self {
                    Self::Scalar(a) => {
                        writeln!(f, "{{ }}")?;
                        if let Some(prec) = f.precision() {
                            write!(f, $fmt_str_with_precision, a, prec)?;
                        } else {
                            write!(f, $fmt_str_without_precision, a)?;
                        }
                    },
                    Self::Tensor(idxs, a) => {
                        write!(f, "{{ <")?;
                        let n_idxs = idxs.row.len();
                        for (k, idx) in idxs.row.iter().enumerate() {
                            idx.fmt(f)?;
                            if k < n_idxs - 1 { write!(f, ", ")?; }
                        }
                        write!(f, ">, <")?;
                        let n_idxs = idxs.col.len();
                        for (k, idx) in idxs.col.iter().enumerate() {
                            idx.fmt(f)?;
                            if k < n_idxs - 1 { write!(f, ", ")?; }
                        }
                        writeln!(f, "> }}")?;

                        let (nrows, ncols) = a.shape();
                        if nrows == 0 || ncols == 0 { return write!(f, "[ ]"); }

                        let max_length: usize =
                            a.iter()
                            .fold(0, |acc, ak| acc.max(val_width(ak, f)));
                        let max_length_with_space = max_length + 1;

                        writeln!(
                            f,
                            "┌ {:>width$} ┐",
                            "",
                            width = max_length_with_space * ncols - 1
                        )?;
                        for i in 0..nrows {
                            write!(f, "│")?;
                            for j in 0..ncols {
                                let number_length =
                                    val_width(&a[(i, j)], f) + 1;
                                let pad = max_length_with_space - number_length;
                                write!(f, " {:>thepad$}", "", thepad = pad)?;
                                match f.precision() {
                                    Some(prec) => {
                                        write!(f, $fmt_str_with_precision,
                                            a[(i, j)], prec)?;
                                    },
                                    None => {
                                        write!(f, $fmt_str_without_precision,
                                            a[(i, j)])?;
                                    },
                                }
                            }
                            writeln!(f, " │")?;
                        }
                        writeln!(
                            f,
                            "└ {:>width$} ┘",
                            "",
                            width = max_length_with_space * ncols - 1
                        )?;
                    },
                }
                Ok(())
            }
        }
    };
}
impl_fmt!(fmt::Display, "{}", "{:.1$}");
impl_fmt!(fmt::LowerExp, "{:e}", "{:.1$e}");
impl_fmt!(fmt::UpperExp, "{:E}", "{:.1$E}");
impl_fmt!(fmt::Octal, "{:o}", "{:1$o}");
impl_fmt!(fmt::LowerHex, "{:x}", "{:1$x}");
impl_fmt!(fmt::UpperHex, "{:X}", "{:1$X}");
impl_fmt!(fmt::Binary, "{:b}", "{:.1$b}");
impl_fmt!(fmt::Pointer, "{:p}", "{:.1$p}");

fn is_unique<'a, I, T>(elems: I) -> bool
where
    I: IntoIterator<Item = &'a T> + Copy,
    T: PartialEq + 'a,
{
    elems.into_iter().enumerate()
        .all(|(k, e0)| elems.into_iter().skip(k + 1).all(|e1| e0 != e1))
}

fn modinc(n: &mut usize, m: usize) -> bool {
    *n = (*n + 1) % m;
    *n == 0
}

fn ndinc(idxs: &mut [usize], strides: &[usize], k: usize) {
    if k > 0 {
        for (i, s) in idxs.iter_mut().zip(strides) {
            if !modinc(i, *s) { break; }
        }
    }
}

/// Basic implementation of an abstract tensor object.
///
/// A `Tensor<T, A>` consists of some number of numerical quantities of type `A`
/// and a series of *unique* indices belonging to a type `T` that implements
/// [`Idx`].
///
/// This implementation distinguished between rank 0 (scalar) and rank > 0
/// (array) quantities for some small operational benefits, but otherwise lives
/// in an abstraction layer one step above concrete array representations,
/// meaning that individual elements of the tensor usually will not be
/// accessible. No parallelism or GPU computation is offered.
///
/// Note that equality between `Tensor`s is defined with some attention
/// dependence on inner structure. For two `Tensor`s to be equal, their indices
/// must all be equal and *in the same order*, in addition to naive pairwise
/// equality between tensor elements.
///
/// `Tensor`s implement addition and subtraction between tensors with equal sets
/// of indices (index order does not matter here), and multiplication is
/// implemented to greedily contract over all matching indices, resorting to an
/// ordinary tensor product when two tensors have no matching indices.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tensor<T, A>(TensorData<T, A>);

impl<T, A> From<TensorData<T, A>> for Tensor<T, A> {
    fn from(data: TensorData<T, A>) -> Self { Self(data) }
}

impl<T, A> Tensor<T, A>
where
    T: Idx,
    A: ComplexField,
{
    /// Create a new tensor using a function over index values.
    pub fn new<I, F>(indices: I, elems: F) -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        TensorData::new(indices, elems).map(Self::from)
    }

    /// Create a new tensor using a function over index values.
    ///
    /// # Safety
    /// This function does not check for duplicate indices, which will cause
    /// unrecoverable errors in contraction and book-keeping within tensor
    /// networks. A call to this function is safe iff all indices are unique
    /// with dimension greater than zero.
    pub unsafe fn new_unchecked<I, F>(indices: I, elems: F) -> Self
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        TensorData::new_unchecked(indices, elems).into()
    }

    /// Create a new rank-0 (scalar) tensor.
    pub fn new_scalar(val: A) -> Self { TensorData::new_scalar(val).into() }

    /// Create a new tensor from a pre-existing vector of elements.
    ///
    /// Fails if duplicate indices are provided, the length of the vector does
    /// not match the total dimension of the indices, or an index with zero
    /// dimension is encountered.
    pub fn from_elems<I>(indices: I, elems: na::DVector<A>)
        -> TensorResult<Self>
    where I: IntoIterator<Item = T>
    {
        TensorData::from_elems(indices, elems).map(Self::from)
    }

    /// Create a new tensor from a pre-existing vector of elements.
    ///
    /// # Safety
    /// This function does not check for duplicate indices or that the
    /// length of the input vector matches the total dimension of the indices.
    /// Failure to meet these conditions will cause uncrecoverable errors in all
    /// arithmetic operations and book-keeping within tensor networks. A call to
    /// this function is safe iff all indices are unique with dimension greater
    /// than zero, and the length of the input vector is equal to the product of
    /// all index dimensions.
    pub unsafe fn from_elems_unchecked<I>(indices: I, elems: na::DVector<A>)
        -> Self
    where I: IntoIterator<Item = T>
    {
        TensorData::from_elems_unchecked(indices, elems).into()
    }

    /// Return `true` if `self` has rank 0.
    pub fn is_scalar(&self) -> bool { self.0.is_scalar() }

    /// Return `true` if `self` has rank greater than 0.
    pub fn is_tensor(&self) -> bool { self.0.is_scalar() }

    /// Return `true` if `self` has the given index.
    pub fn has_index(&self, index: &T) -> bool { self.0.has_index(index) }

    /// Return the rank (i.e. the number of indices) of `self`.
    pub fn rank(&self) -> usize { self.0.rank() }

    /// Return the shape (dimensions of each index) of `self` in a vector.
    ///
    /// If `self` is a scalar, the returned vector is empty.
    pub fn shape(&self) -> Vec<usize> { self.0.shape() }

    /// Reshape the underlying data array so that it can be represented as an
    /// approximately square matrix. This has no bearing on arithmetic
    /// operations, but can be nice if you want print the tensor data.
    ///
    /// This is a no-move, no-copy operation, and a no-op on scalars.
    pub fn reshape_square(&mut self) { self.0.reshape_square(); }

    /// Return an iterator over all indices.
    ///
    /// If `self` is a scalar, the iterator is empty.
    pub fn indices(&self) -> Indices<'_, T> { self.0.indices() }

    /// Return an iterator over mutable references to all indices.
    ///
    /// If `self` is a scalar, the iterator is empty.
    ///
    /// # Safety
    /// Mutating an index in such a way that its dimension changes or it becomes
    /// equal to another index in `self` or in an incommensurate way with
    /// another index in a tensor network will cause unrecoverable errors in all
    /// arithmetic operations and book-keeping. A call to this function is safe
    /// iff mutations to the index do not change the index's dimension and still
    /// allow the index to be identitfied with its tensor(s) within a network.
    pub unsafe fn indices_mut(&mut self) -> IndicesMut<'_, T> {
        self.0.indices_mut()
    }

    /// Apply a mapping function to all indices.
    ///
    /// If `self` is a scalar, no operation is performed.
    ///
    /// # Safety
    /// Mutating an index in such a way that its dimension changes or it becomes
    /// equal to another index in `self` or in an incommensurate way with
    /// another index in a tensor network will cause unrecoverable errors in all
    /// arithmetic operations and book-keeping. A call to this function is safe
    /// iff mutations to the index do not change the index's dimension and still
    /// allow the index to be identitfied with its tensor(s) within a network.
    pub unsafe fn map_indices<F, U>(self, f: F) -> Tensor<U, A>
    where
        F: FnMut(T) -> U,
        U: Idx,
    {
        self.0.map_indices(f).into()
    }

    /// Apply a mapping function to the (indexed) elements of `self`, returning
    /// a new tensor.
    pub fn map<F, B>(&self, f: F) -> Tensor<T, B>
    where
        F: FnMut(&[usize], &A) -> B,
        B: ComplexField,
    {
        self.0.map(f).into()
    }

    /// Apply a mapping function to the (indexed) elements of `self` in place,
    /// reassigning elements to the outputs of the function.
    pub fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(&[usize], &A) -> A
    {
        self.0.map_inplace(f);
    }

    /// Multiply `self` with `other`, consuming both. All common indices are
    /// contracted. If no common indices exist, this is equivalent to an
    /// ordinary tensor product.
    ///
    /// This operation results in a new `Tensor` whose indices comprise all
    /// non-common indices belonging to `self` and `other`. All indices
    /// originally belonging to `self` are placed before those from `other`, but
    /// the ordering of indices within these groups is not preserved.
    ///
    /// This operation is used by the `*` operator.
    pub fn multiply(self, other: Self) -> Self {
        self.0.multiply(other.0).into()
    }

    /// Compute the sum of `self` and `other`, consuming both.
    ///
    /// Fails if either tensor holds an index not held by the other.
    ///
    /// This operation is used by the `+` operator.
    pub fn add_checked(self, other: Self) -> TensorResult<Self> {
        self.0.add_checked(other.0).map(Self::from)
    }

    /// Compute the difference of `self` and `other`, consuming both.
    ///
    /// Fails if either tensor holds an index not held by the other.
    ///
    /// This operation is used by the `-` operator.
    pub fn sub_checked(self, other: Self) -> TensorResult<Self> {
        self.0.sub_checked(other.0).map(Self::from)
    }

    /// Return a new tensor containing the element-wise complex conjugation of
    /// `self`.
    pub fn conj(&self) -> Self { self.0.conj().into() }

    /// Compute the element-wise complex conjugation of `self`.
    pub fn into_conj(self) -> Self { self.0.into_conj().into() }

    /// Conjugate `self` element-wise in place.
    pub fn conj_mut(&mut self) { self.0.conj_mut() }
}

impl<T, A> std::ops::Mul<Tensor<T, A>> for Tensor<T, A>
where
    T: Idx,
    A: ComplexField,
{
    type Output = Tensor<T, A>;

    fn mul(self, other: Tensor<T, A>) -> Self::Output {
        self.multiply(other)
    }
}

impl<T, A> std::ops::Add<Tensor<T, A>> for Tensor<T, A>
where
    T: Idx,
    A: ComplexField,
{
    type Output = Tensor<T, A>;

    fn add(self, other: Tensor<T, A>) -> Self::Output {
        match self.add_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

impl<T, A> std::ops::Sub<Tensor<T, A>> for Tensor<T, A>
where
    T: Idx,
    A: ComplexField,
{
    type Output = Tensor<T, A>;

    fn sub(self, other: Tensor<T, A>) -> Self::Output {
        match self.sub_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

macro_rules! impl_fmt_derived {
    ( $trait: path ) => {
        impl<T, A> $trait for Tensor<T, A>
        where
            T: fmt::Display,
            A: na::Scalar + $trait,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    }
}
impl_fmt_derived!(fmt::Display);
impl_fmt_derived!(fmt::LowerExp);
impl_fmt_derived!(fmt::UpperExp);
impl_fmt_derived!(fmt::Octal);
impl_fmt_derived!(fmt::LowerHex);
impl_fmt_derived!(fmt::UpperHex);
impl_fmt_derived!(fmt::Binary);
impl_fmt_derived!(fmt::Pointer);

