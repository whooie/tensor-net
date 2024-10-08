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
//! #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
//!
//!     fn label(&self) -> String { format!("{self:?}") }
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

use std::{
    fmt,
    hash::Hash,
    ops::{ Add, Sub, Mul, Range },
};
use itertools::Itertools;
use ndarray::{ self as nd, Dimension };
use num_complex::ComplexFloat;
use num_traits::NumAssign;
use rustc_hash::FxHashSet as HashSet;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    /// Returned when attempting to create a new tensor with duplicate indices.
    #[error("error in tensor creation: duplicate indices")]
    DuplicateIndices,

    /// Returned when attempting to create a new tensor from a pre-existing
    /// array and the provided indices have non-matching total dimension.
    #[error("error in tensor creation: non-matching indices and array shape")]
    IncompatibleShape,

    /// Returned when a contraction is attempted between two tensors (rank > 0)
    /// with no common indices.
    #[error("error in tensor contraction: no matching indices")]
    NoMatchingIndices,

    /// Returned when a tensor product is attempted between two tensors (rank >
    /// 0) with common indices.
    #[error("error in tensor product: matching indices")]
    MatchingIndices,

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
/// Generally, a tensor network will want to distinguish between the available
/// indices over which its tensors can be contracted. There are two ways to
/// implement this. The first is by using a "static" representation, where all
/// available indices are encoded in the structure of the type implementing this
/// trait:
/// ```ignore
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
///
///     fn label(&self) -> String { format!("{self:?}") }
/// }
/// ```
/// This representation is safer because it encodes relevant information in
/// degrees of freedom that are immutable at runtime, but cannot be used in a
/// tensor network that dynamically adds or removes arbitrary tensors and
/// indices.
///
/// The second is by using a "dynamic" representation, where the available
/// indices and their dimensions are left to be encoded as arbitrary fields of
/// the type:
/// ```ignore
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// struct Index(String, usize);
///
/// impl Idx for Index {
///     fn dim(&self) -> usize { self.1 }
///
///     fn label(&self) -> String { format!("{self:?}") }
/// }
/// ```
/// This representation allows for dynamic mutation of a tensor network, but
/// comes with a memory overhead and is somewhat less safe than static
/// representations because there are runtime-mutable degrees of freedom.
///
/// Ultimately, static representations should be preferred where possible.
pub trait Idx: Clone + Eq + Hash + std::fmt::Debug {
    /// Return the number of values the index can take.
    ///
    /// This value must never be zero.
    fn dim(&self) -> usize;

    /// Return an iterator over all possible index values. The default
    /// implementation returns `0..self.dim()`.
    fn iter(&self) -> Range<usize> { 0..self.dim() }

    /// Return an identifying label for the index. This method is used only for
    /// printing purposes.
    fn label(&self) -> String;
}

/// A dynamically dimensioned tensor index type.
///
/// This type should be used only when a tensor network needs to be dynamically
/// modifiable, otherwise static implementations of tensor indices should be
/// preferred (see [`Idx`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynIdx(
    /// Identifier label.
    pub String,
    /// Index dimension.
    pub usize,
);

impl Idx for DynIdx {
    fn dim(&self) -> usize { self.1 }

    fn label(&self) -> String { format!("{self:?}") }
}

impl<T> From<(T, usize)> for DynIdx
where T: ToString
{
    fn from(x: (T, usize)) -> Self {
        let (label, dim) = x;
        Self(label.to_string(), dim)
    }
}

/// An element of a [`Tensor`].
pub trait Elem: Copy + Clone + NumAssign + 'static { }

impl<T> Elem for T
where T: Copy + Clone + NumAssign + 'static
{ }

#[derive(Clone, PartialEq, Eq, Debug)]
enum TensorData<T, A> {
    Scalar(A),
    Tensor(Vec<T>, nd::ArrayD<A>),
}

impl<T, A> TensorData<T, A>
where
    T: Idx,
    A: Elem,
{
    fn new<I, F>(indices: I, mut elems: F) -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        let indices: Vec<T> = indices.into_iter().collect();
        let unique_check: HashSet<&T> = indices.iter().collect();
        if unique_check.len() != indices.len() {
            return Err(DuplicateIndices);
        }
        if indices.is_empty() {
            let scalar: A = elems(&[]);
            Ok(Self::Scalar(scalar))
        } else {
            let shape: Vec<usize>
                = indices.iter().map(|idx| idx.dim()).collect();
            let data: nd::ArrayD<A>
                = nd::ArrayD::from_shape_fn(
                    shape,
                    |idxs| elems(idxs.as_array_view().as_slice().unwrap())
                );
            Ok(Self::Tensor(indices, data))
        }
    }

    unsafe fn new_unchecked<I, F>(indices: I, mut elems: F) -> Self
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        let indices: Vec<T> = indices.into_iter().collect();
        if indices.is_empty() {
            let scalar: A = elems(&[]);
            Self::Scalar(scalar)
        } else {
            let shape: Vec<usize>
                = indices.iter().map(|idx| idx.dim()).collect();
            let data: nd::ArrayD<A>
                = nd::ArrayD::from_shape_fn(
                    shape,
                    |idxs| elems(idxs.as_array_view().as_slice().unwrap())
                );
            Self::Tensor(indices, data)
        }
    }

    fn new_scalar(val: A) -> Self { Self::Scalar(val) }

    fn from_array<I, D>(indices: I, array: nd::Array<A, D>)
        -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        D: nd::Dimension,
    {
        let indices: Vec<T> = indices.into_iter().collect();
        let unique_check: HashSet<&T> = indices.iter().collect();
        if unique_check.len() != indices.len() {
            return Err(DuplicateIndices);
        }
        let shape = array.shape();
        if indices.len() != shape.len()
            || indices.iter().zip(shape).any(|(idx, dim)| idx.dim() != *dim)
        {
            return Err(IncompatibleShape);
        }
        if indices.is_empty() {
            Ok(Self::Scalar(array.into_iter().next().unwrap()))
        } else {
            Ok(Self::Tensor(indices, array.into_dyn()))
        }
    }

    unsafe fn from_array_unchecked<I, D>(indices: I, array: nd::Array<A, D>)
        -> Self
    where
        I: IntoIterator<Item = T>,
        D: nd::Dimension,
    {
        let indices: Vec<T> = indices.into_iter().collect();
        if indices.is_empty() {
            Self::Scalar(array.into_iter().next().unwrap())
        } else {
            Self::Tensor(indices, array.into_dyn())
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

    fn index_pos(&self, index: &T) -> Option<usize> {
        match self {
            Self::Scalar(_) => None,
            Self::Tensor(idxs, _) => {
                idxs.iter()
                    .enumerate()
                    .find_map(|(k, idx)| (idx == index).then_some(k))
            },
        }
    }

    unsafe fn get_index_mut(&mut self, index: &T) -> Option<&mut T> {
        match self {
            Self::Scalar(_) => None,
            Self::Tensor(idxs, _) => idxs.iter_mut().find(|idx| idx == &index),
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
            Self::Tensor(idxs, _)
                => idxs.iter().map(|idx| idx.dim()).collect(),
        }
    }

    fn indices(&self) -> Indices<'_, T> {
        match self {
            Self::Scalar(_) => Indices(IndicesData::Scalar),
            Self::Tensor(idxs, _)
                => Indices(IndicesData::Tensor(idxs.iter())),
        }
    }

    unsafe fn indices_mut(&mut self) -> IndicesMut<'_, T> {
        match self {
            Self::Scalar(_) => IndicesMut(IndicesMutData::Scalar),
            Self::Tensor(idxs, _)
                => IndicesMut(IndicesMutData::Tensor(idxs.iter_mut())),
        }
    }

    unsafe fn map_indices<F, U>(self, map: F) -> TensorData<U, A>
    where
        F: Fn(T) -> U,
        U: Idx,
    {
        match self {
            Self::Scalar(a) => TensorData::Scalar(a),
            Self::Tensor(idxs, data) => {
                let idxs = idxs.into_iter().map(map).collect();
                TensorData::Tensor(idxs, data)
            },
        }
    }

    fn map<F, B>(&self, mut f: F) -> TensorData<T, B>
    where
        F: FnMut(&[usize], &A) -> B,
        B: Elem,
    {
        match self {
            Self::Scalar(a) => TensorData::Scalar(f(&[], a)),
            Self::Tensor(idxs, data) => TensorData::Tensor(
                idxs.clone(),
                data.indexed_iter()
                    .map(|(ix, a)| f(ix.as_array_view().as_slice().unwrap(), a))
                    .collect::<nd::Array1<B>>()
                    .into_dyn()
                    .into_shape(data.raw_dim())
                    .unwrap()
            ),
        }
    }

    fn map_inplace<F>(&mut self, mut f: F)
    where F: FnMut(&[usize], &A) -> A
    {
        match self {
            Self::Scalar(a) => { *a = f(&[], &*a); },
            Self::Tensor(_, data) => {
                data.indexed_iter_mut()
                    .for_each(|(ix, a)| {
                        *a = f(ix.as_array_view().as_slice().unwrap(), &*a);
                    })
            },
        }
    }

    fn swap_indices(&mut self, a: &T, b: &T) {
        if let Self::Tensor(idxs, data) = self {
            let pos_a
                = idxs.iter()
                .enumerate()
                .find_map(|(k, idx)| (idx == a).then_some(k));
            let pos_b
                = idxs.iter()
                .enumerate()
                .find_map(|(k, idx)| (idx == b).then_some(k));
            if let Some((k_a, k_b)) = pos_a.zip(pos_b) {
                idxs.swap(k_a, k_b);
                data.swap_axes(k_a, k_b);
            }
        }
    }

    fn swap_indices_pos(&mut self, a: usize, b: usize) {
        if let Self::Tensor(idxs, data) = self {
            if a >= idxs.len() || b >= idxs.len() { return; }
            idxs.swap(a, b);
            data.swap_axes(a, b);
        }
    }

    fn do_contract(
        idx_common: Vec<T>,
        mut idxs_a: Vec<T>,
        mut a: nd::ArrayD<A>,
        mut idxs_b: Vec<T>,
        mut b: nd::ArrayD<A>,
    ) -> Self
    {
        // swap common indices and corresponding axes to the rightmost positions
        // in a and the leftmost in b
        let n_idx_a = idxs_a.len();
        let n_common = idx_common.len();
        let mut k_source: usize;
        for (k_target, idx) in idx_common.iter().enumerate() {
            k_source
                = idxs_a.iter()
                .enumerate()
                .find_map(|(k_source, idx_source)| {
                    (idx_source == idx).then_some(k_source)
                })
                .unwrap();
            idxs_a.swap(k_source, n_idx_a - n_common + k_target);
            a.swap_axes(k_source, n_idx_a - n_common + k_target);

            k_source
                = idxs_b.iter()
                .enumerate()
                .find_map(|(k_source, idx_source)| {
                    (idx_source == idx).then_some(k_source)
                })
                .unwrap();
            idxs_b.swap(k_source, k_target);
            b.swap_axes(k_source, k_target);
        }

        // reshape to fuse all common and non-common indices together so we can
        // use matmul and then reshape the result to un-fuse
        let dim_noncomm_a
            = idxs_a.iter().take(n_idx_a - n_common).map(Idx::dim).product();
        let dim_comm
            = idx_common.iter().map(Idx::dim).product();
        let dim_noncomm_b
            = idxs_b.iter().skip(n_common).map(Idx::dim).product();
        let a: nd::CowArray<A, nd::Ix2>
            = a.as_standard_layout()
            .into_shape((dim_noncomm_a, dim_comm))
            .unwrap();
        let b: nd::CowArray<A, nd::Ix2>
            = b.as_standard_layout()
            .into_shape((dim_comm, dim_noncomm_b))
            .unwrap();
        let c: nd::Array2<A> = a.dot(&b);
        let new_shape: Vec<usize>
            = idxs_a.iter().take(n_idx_a - n_common)
            .chain(idxs_b.iter().skip(n_common))
            .map(Idx::dim)
            .collect();
        if new_shape.is_empty() {
            let c: A = c.into_iter().next().unwrap();
            TensorData::Scalar(c)
        } else {
            let new_idxs: Vec<T>
                = idxs_a.into_iter().take(n_idx_a - n_common)
                .chain(idxs_b.into_iter().skip(n_common))
                .collect();
            let c = c.into_shape(new_shape).unwrap();
            TensorData::Tensor(new_idxs, c)
        }
    }

    fn contract(self, other: Self) -> TensorResult<Self> {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                Ok(TensorData::Scalar(a * b))
            },
            (TensorData::Scalar(a), TensorData::Tensor(idxs, b)) => {
                let mul: nd::ArrayD<A> = b.mapv(|bk| a * bk);
                Ok(TensorData::Tensor(idxs, mul))
            },
            (TensorData::Tensor(idxs, a), TensorData::Scalar(b)) => {
                let mul: nd::ArrayD<A> = a.mapv(|ak| ak * b);
                Ok(TensorData::Tensor(idxs, mul))
            },
            (TensorData::Tensor(idxs_a, a), TensorData::Tensor(idxs_b, b)) => {
                // find common indices, error if none
                let idx_common: Vec<T>
                    = idxs_a.iter()
                    .filter(|idx| idxs_b.contains(idx))
                    .cloned()
                    .collect();
                (!idx_common.is_empty()).then_some(())
                    .ok_or(NoMatchingIndices)?;
                Ok(Self::do_contract(idx_common, idxs_a, a, idxs_b, b))
            },
        }
    }

    fn do_tensor_prod(
        mut idxs_a: Vec<T>,
        a: nd::ArrayD<A>,
        mut idxs_b: Vec<T>,
        b: nd::ArrayD<A>,
    ) -> Self
    {
        idxs_a.append(&mut idxs_b);
        let new_shape: Vec<usize> = idxs_a.iter().map(Idx::dim).collect();
        let c: nd::ArrayD<A>
            = a.iter()
            .cartesian_product(b.iter())
            .map(|(ak, bk)| *ak * *bk)
            .collect::<nd::Array1<A>>()
            .into_shape(new_shape)
            .unwrap();
        TensorData::Tensor(idxs_a, c)
    }

    fn tensor_prod(self, other: Self) -> TensorResult<Self> {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                Ok(TensorData::Scalar(a * b))
            },
            (TensorData::Scalar(a), TensorData::Tensor(idxs, b)) => {
                let mul: nd::ArrayD<A> = b.mapv(|bk| a * bk);
                Ok(TensorData::Tensor(idxs, mul))
            },
            (TensorData::Tensor(idxs, a), TensorData::Scalar(b)) => {
                let mul: nd::ArrayD<A> = a.mapv(|ak| ak * b);
                Ok(TensorData::Tensor(idxs, mul))
            },
            (TensorData::Tensor(idxs_a, a), TensorData::Tensor(idxs_b, b)) => {
                // find common indices, error if some
                idxs_a.iter()
                    .all(|idx| !idxs_b.contains(idx))
                    .then_some(())
                    .ok_or(MatchingIndices)?;
                Ok(Self::do_tensor_prod(idxs_a, a, idxs_b, b))
            },
        }
    }

    fn multiply(self, other: Self) -> Self {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                TensorData::Scalar(a * b)
            },
            (TensorData::Scalar(a), TensorData::Tensor(idxs, b)) => {
                let mul: nd::ArrayD<A> = b.mapv(|bk| a * bk);
                TensorData::Tensor(idxs, mul)
            },
            (TensorData::Tensor(idxs, a), TensorData::Scalar(b)) => {
                let mul: nd::ArrayD<A> = a.mapv(|ak| ak * b);
                TensorData::Tensor(idxs, mul)
            },
            (TensorData::Tensor(idxs_a, a), TensorData::Tensor(idxs_b, b)) => {
                let idx_common: Vec<T>
                    = idxs_a.iter()
                    .filter(|idx| idxs_b.contains(idx))
                    .cloned()
                    .collect();
                if idx_common.is_empty() {
                    Self::do_tensor_prod(idxs_a, a, idxs_b, b)
                } else {
                    Self::do_contract(idx_common, idxs_a, a, idxs_b, b)
                }
            },
        }
    }

    fn add_checked(self, other: Self) -> TensorResult<Self> {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                Ok(TensorData::Scalar(a + b))
            },
            (TensorData::Scalar(_), TensorData::Tensor(..)) => {
                Err(IncompatibleIndicesAdd)
            },
            (TensorData::Tensor(..), TensorData::Scalar(_)) => {
                Err(IncompatibleIndicesAdd)
            },
            (
                TensorData::Tensor(idxs_a, mut a),
                TensorData::Tensor(mut idxs_b, mut b),
            ) => {
                if idxs_a.len() != idxs_b.len()
                    || idxs_a.iter().any(|idx| !idxs_b.contains(idx))
                    || idxs_b.iter().any(|idx| !idxs_a.contains(idx))
                {
                    return Err(IncompatibleIndicesAdd);
                }
                let mut k_b: usize;
                for (k_a, idx_a) in idxs_a.iter().enumerate() {
                    k_b
                        = idxs_b.iter()
                        .enumerate()
                        .skip(k_a)
                        .find_map(|(k_b, idx_b)| {
                            (idx_a == idx_b).then_some(k_b)
                        })
                        .unwrap();
                    idxs_b.swap(k_a, k_b);
                    b.swap_axes(k_a, k_b);
                }
                a += &b;
                Ok(TensorData::Tensor(idxs_a, a))
            },
        }
    }

    fn sub_checked(self, other: Self) -> TensorResult<Self> {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                Ok(TensorData::Scalar(a - b))
            },
            (TensorData::Scalar(_), TensorData::Tensor(..)) => {
                Err(IncompatibleIndicesSub)
            },
            (TensorData::Tensor(..), TensorData::Scalar(_)) => {
                Err(IncompatibleIndicesSub)
            },
            (
                TensorData::Tensor(idxs_a, mut a),
                TensorData::Tensor(mut idxs_b, mut b),
            ) => {
                if idxs_a.len() != idxs_b.len()
                    || idxs_a.iter().any(|idx| !idxs_b.contains(idx))
                    || idxs_b.iter().any(|idx| !idxs_a.contains(idx))
                {
                    return Err(IncompatibleIndicesSub);
                }
                let mut k_b: usize;
                for (k_a, idx_a) in idxs_a.iter().enumerate() {
                    k_b
                        = idxs_b.iter()
                        .enumerate()
                        .skip(k_a)
                        .find_map(|(k_b, idx_b)| {
                            (idx_a == idx_b).then_some(k_b)
                        })
                        .unwrap();
                    idxs_b.swap(k_a, k_b);
                    b.swap_axes(k_a, k_b);
                }
                a -= &b;
                Ok(TensorData::Tensor(idxs_a, a))
            },
        }
    }

    fn into_flat(self) -> (Vec<T>, nd::Array1<A>) {
        match self {
            Self::Scalar(a) => (Vec::new(), nd::array![a]),
            Self::Tensor(idxs, data) => (idxs, data.into_iter().collect()),
        }
    }

    fn into_raw(self) -> (Vec<T>, nd::ArrayD<A>) {
        match self {
            Self::Scalar(a) => (Vec::new(), nd::array![a].into_dyn()),
            Self::Tensor(idxs, data) => (idxs, data),
        }
    }
}

impl<T, A> TensorData<T, A>
where T: Idx + Ord
{
    fn sort_indices(&mut self) {
        if let Self::Tensor(idxs, data) = self {
            let mut n = idxs.len();
            let mut swapped = true;
            while swapped {
                swapped = false;
                for i in 1..=n - 1 {
                    if idxs[i - 1] > idxs[i] {
                        idxs.swap(i - 1, i);
                        data.swap_axes(i - 1, i);
                        swapped = true;
                    }
                }
                n -= 1;
            }
        }
    }
}

impl<T, A> TensorData<T, A>
where
    T: Idx,
    A: ComplexFloat,
{
    fn conj(&self) -> Self {
        match self {
            Self::Scalar(z) => Self::Scalar(z.conj()),
            Self::Tensor(idxs, z) => {
                Self::Tensor(idxs.clone(), z.mapv(|zk| zk.conj()))
            },
        }
    }
}

impl<T, A> Mul<TensorData<T, A>> for TensorData<T, A>
where
    T: Idx,
    A: Elem,
{
    type Output = TensorData<T, A>;

    fn mul(self, other: TensorData<T, A>) -> Self::Output {
        self.multiply(other)
    }
}

impl<T, A> Add<TensorData<T, A>> for TensorData<T, A>
where
    T: Idx,
    A: Elem,
{
    type Output = TensorData<T, A>;

    fn add(self, other: TensorData<T, A>) -> Self::Output {
        match self.add_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

impl<T, A> Sub<TensorData<T, A>> for TensorData<T, A>
where
    T: Idx,
    A: Elem,
{
    type Output = TensorData<T, A>;

    fn sub(self, other: TensorData<T, A>) -> Self::Output {
        match self.sub_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

impl<T, A> fmt::Display for TensorData<T, A>
where
    T: fmt::Display,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(a) => {
                a.fmt(f)?;
                write!(f, " {{ }}")?;
            },
            Self::Tensor(idxs, a) => {
                a.fmt(f)?;
                write!(f, " {{ ")?;
                let n_idxs = idxs.len();
                for (k, idx) in idxs.iter().enumerate() {
                    idx.fmt(f)?;
                    if k < n_idxs - 1 { write!(f, ", ")?; }
                }
                write!(f, " }}")?;
            },
        }
        Ok(())
    }
}

/// Basic implementation of an abstract tensor object.
///
/// A `Tensor<T, A>` consists of some number of numerical quantities of type `A`
/// and a series of indices belonging to a type `T` that implements [`Idx`].
///
/// This implementation distinguishes between rank 0 (scalar) and rank > 0
/// (array) quantities for some small operational benefits, but otherwise lives
/// in an abstraction layer one step above concrete array representations. No
/// parallelism or GPU computation is offered.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tensor<T, A>(TensorData<T, A>);

impl<T, A> From<TensorData<T, A>> for Tensor<T, A> {
    fn from(data: TensorData<T, A>) -> Self { Self(data) }
}

impl<T, A> Tensor<T, A>
where
    T: Idx,
    A: Elem,
{
    /// Create a new tensor using a function over given indices.
    pub fn new<I, F>(indices: I, elems: F) -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        TensorData::new(indices, elems).map(Self::from)
    }

    /// Create a new tensor using a function over given indices.
    ///
    /// # Safety
    /// This function does not check for duplicate indices, which will cause
    /// unrecoverable errors in contraction and book-keeping within tensor
    /// networks. A call to this function is safe iff all indices are unique.
    pub unsafe fn new_unchecked<I, F>(indices: I, elems: F) -> Self
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        TensorData::new_unchecked(indices, elems).into()
    }

    /// Create a new rank-0 (scalar) tensor.
    pub fn new_scalar(val: A) -> Self { TensorData::new_scalar(val).into() }

    /// Create a new tensor from an n-dimensional array.
    ///
    /// Fails if duplicate indices are provided or the dimensions of the array
    /// do not match those of the indices.
    pub fn from_array<I, D>(indices: I, array: nd::Array<A, D>)
        -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        D: nd::Dimension,
    {
        TensorData::from_array(indices, array).map(Self::from)
    }

    /// Create a new tensor from an n-dimensional array.
    ///
    /// # Safety
    /// This function does not check for duplicate indices or that the
    /// dimensions of the array match those of the indices. Failure to meed
    /// these conditions will cause unrecoverable errors in all arithmetic
    /// operations and book-keeping within tensor networks. A call to thsi
    /// function is safe iff all indices are unique and each is aligned to an
    /// axis of size equal to its dimension in the data array.
    pub unsafe fn from_array_unchecked<I, D>(indices: I, array: nd::Array<A, D>)
        -> Self
    where
        I: IntoIterator<Item = T>,
        D: nd::Dimension,
    {
        TensorData::from_array_unchecked(indices, array).into()
    }

    /// Return `true` if `self has rank 0.
    ///
    /// This is equivalent to `self.rank() == 0` and `self.shape().is_empty()`.
    pub fn is_scalar(&self) -> bool { self.0.is_scalar() }

    /// Return `true` if `self` has the given index.
    pub fn has_index(&self, index: &T) -> bool { self.0.has_index(index) }

    /// Return the position of an index if contained in `self`.
    pub fn index_pos(&self, index: &T) -> Option<usize> {
        self.0.index_pos(index)
    }

    /// Return a mutable reference to an index if contained in `self`.
    ///
    /// # Safety
    /// Mutating an index in such a way that its dimension changes will cause
    /// unrecoverable errors in all arithmetic operations and book-keeping
    /// within tensor networks. A call to this function is safe iff mutations to
    /// the index do not change the index's dimension and still allow the index
    /// to be identified with its tensor(s) within a network.
    pub unsafe fn get_index_mut(&mut self, index: &T) -> Option<&mut T> {
        self.0.get_index_mut(index)
    }

    /// Return the rank of `self`.
    pub fn rank(&self) -> usize { self.0.rank() }

    /// Return the shape (dimensions of each index) of `self` in a vector.
    ///
    /// If `self` is a scalar, the returned vector is empty.
    pub fn shape(&self) -> Vec<usize> { self.0.shape() }

    /// Return an iterator over all indices.
    ///
    /// If `self` is a scalar, the iterator is empty.
    pub fn indices(&self) -> Indices<'_, T> { self.0.indices() }

    /// Return an iterator over mutable references to all indices.
    ///
    /// If `self` is a scalar, the iterator is empty.
    ///
    /// # Safety
    /// Mutating an index in such a way that its dimension changes will cause
    /// unrecoverable errors in all arithmetic operations and book-keeping
    /// within tensor networks. A call to this function is safe iff mutations to
    /// indices do not change the indices' dimensions and still allow each to be
    /// identified with its tensor(s) within a network.
    pub unsafe fn indices_mut(&mut self) -> IndicesMut<'_, T> {
        self.0.indices_mut()
    }

    /// Apply a mapping function to all indices.
    ///
    /// If `self` is a scalar, no operation is performed.
    ///
    /// # Safety
    /// Mutating an index in such a way that its dimension changes will cause
    /// unrecoverable errors in all arithmetic operations and book-keeping
    /// within tensor networks. A call to this function is safe iff
    /// transformations to indices do not change the indices' dimensions and
    /// still allow each to be identified with its tensor(s) within a network.
    pub unsafe fn map_indices<F, U>(self, map: F) -> Tensor<U, A>
    where
        F: Fn(T) -> U,
        U: Idx,
    {
        self.0.map_indices(map).into()
    }

    /// Apply a mapping function to the (indexed) elements of `self`, returning
    /// a new array.
    pub fn map<F, B>(&self, f: F) -> Tensor<T, B>
    where
        F: FnMut(&[usize], &A) -> B,
        B: Elem,
    {
        self.0.map(f).into()
    }

    /// Apply a mapping function to the (indexed) elements of `self` in place,
    /// reassigning to the output of the function.
    pub fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(&[usize], &A) -> A
    {
        self.0.map_inplace(f);
    }

    /// Swap two indices in the underlying representation of `self`.
    ///
    /// Note that this operation has no bearing over the computational
    /// complexity of tensor contractions. If either given index does not exist
    /// within `self`, no swap is performed.
    pub fn swap_indices(&mut self, a: &T, b: &T) { self.0.swap_indices(a, b); }

    /// Swap two index positions in the underlying representation of `self`.
    ///
    /// Note that this operation has no bearing over the computational
    /// complexity of tensor contractions. If either given index does not exist
    /// within `self`, no swap is performed.
    pub fn swap_indices_pos(&mut self, a: usize, b: usize) {
        self.0.swap_indices_pos(a, b);
    }

    /// Contract `self` with `other` over all common indices, consuming both.
    ///
    /// This operation results in a new `Tensor` whose indices comprise all
    /// non-common indices belonging to `self` and `other`. All indices
    /// originally belonging `self` are placed before those from `other`, but
    /// the ordering of indices within these groups is not preserved.
    ///
    /// Fails if no common indices exist.
    pub fn contract(self, other: Self) -> TensorResult<Self> {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.contract(rhs).map(|data| data.into())
    }

    /// Compute the tensor product of `self` and `other`, consuming both.
    ///
    /// This operation results in a new `Tensor` whose indices comprise all
    /// those belonging to `self` and `other`. Index ordering is preserved, with
    /// all indices originally belonging to `self` are placed before those from
    /// `other`.
    ///
    /// Fails if common indices exist.
    pub fn tensor_prod(self, other: Self) -> TensorResult<Self> {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.tensor_prod(rhs).map(|data| data.into())
    }

    /// Compute the generalized product of `self` and `other`, consuming both.
    ///
    /// This product is an ordinary tensor [contraction][Self::contract] if
    /// common indices exist (or ordinary multiplication if either is a scalar),
    /// otherwise a [tensor product][Self::tensor_prod].
    pub fn multiply(self, other: Self) -> Self {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.multiply(rhs).into()
    }

    /// Compute the sum of `self` and `other`, consuming both.
    ///
    /// Fails if either tensor holds an index not held by the other.
    pub fn add_checked(self, other: Self) -> TensorResult<Self> {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.add_checked(rhs).map(|data| data.into())
    }

    /// Compute the difference of `self` and `other`, consuming both.
    ///
    /// Fails if either tensor holds an index not held by the other.
    pub fn sub_checked(self, other: Self) -> TensorResult<Self> {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.sub_checked(rhs).map(|data| data.into())
    }

    /// Flatten `self` an ordinary 1D array and a list of indices.
    ///
    /// If `self` is a scalar, the list is empty.
    pub fn into_flat(self) -> (Vec<T>, nd::Array1<A>) {
        self.0.into_flat()
    }

    /// Unwrap `self` into a list of indices and an ordinary N-dimensional
    /// array.
    ///
    /// If `self` is a scalar, the list is empty.
    pub fn into_raw(self) -> (Vec<T>, nd::ArrayD<A>) {
        self.0.into_raw()
    }
}

impl<T, A> Tensor<T, A>
where T: Idx + Ord
{
    /// Apply an ordering to the set of indices in the underlying representation
    /// of `self`.
    ///
    /// Note that this operation has no bearing over the computational
    /// complexity of tensor contractions.
    pub fn sort_indices(&mut self) { self.0.sort_indices() }
}

impl<T, A> Tensor<T, A>
where
    T: Idx,
    A: ComplexFloat,
{
    /// Return a new tensor containing the element-wise conjugation of `self`.
    pub fn conj(&self) -> Self { self.0.conj().into() }
}

impl<T, A> Mul<Tensor<T, A>> for Tensor<T, A>
where
    T: Idx,
    A: Elem,
{
    type Output = Tensor<T, A>;

    fn mul(self, other: Tensor<T, A>) -> Self::Output {
        (self.0 * other.0).into()
    }
}

impl<T, A> Add<Tensor<T, A>> for Tensor<T, A>
where
    T: Idx,
    A: Elem,
{
    type Output = Tensor<T, A>;

    fn add(self, other: Tensor<T, A>) -> Self::Output {
        (self.0 + other.0).into()
    }
}

impl<T, A> Sub<Tensor<T, A>> for Tensor<T, A>
where
    T: Idx,
    A: Elem,
{
    type Output = Tensor<T, A>;

    fn sub(self, other: Tensor<T, A>) -> Self::Output {
        (self.0 - other.0).into()
    }
}

impl<T, A> fmt::Display for Tensor<T, A>
where
    T: fmt::Display,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

enum IndicesData<'a, T> {
    Scalar,
    Tensor(std::slice::Iter<'a, T>),
}

/// Iterator type over the indices of a given [`Tensor`].
///
/// The iterator item type is `&T`.
pub struct Indices<'a, T>(IndicesData<'a, T>);

impl<'a, T> Iterator for Indices<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IndicesData::Scalar => None,
            IndicesData::Tensor(iter) => iter.next(),
        }
    }
}

enum IndicesMutData<'a, T> {
    Scalar,
    Tensor(std::slice::IterMut<'a, T>),
}

/// Iterator type over mutable references to the indices of a given [`Tensor`].
///
/// The iterator item type is `&mut T`.
pub struct IndicesMut<'a, T>(IndicesMutData<'a, T>);

impl<'a, T> Iterator for IndicesMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IndicesMutData::Scalar => None,
            IndicesMutData::Tensor(iter) => iter.next(),
        }
    }
}

