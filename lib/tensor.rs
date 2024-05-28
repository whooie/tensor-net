//! An N-dimensional array of data with shape determined by a set of numerical
//! indices.
//!
//! A [`Tensor`] is multi-linear algebraic object that can be seen as the
//! generalization of linear or bi-linear objects such as vectors and matrices.
//! Like vectors and matrices, tensors represent collections of objects that are
//! accessed by supplying a number of index values (i.e. 1 for a vector, 2 for a
//! matrix), but allow for more than 2.
//!
//! *T*<sub>*a*<sub>1</sub>,...,*a*<sub>*N*</sub></sub>
//!
//! Linear operations are generalized as well. The usual matrix-matrix,
//! matrix-vector, vector-matrix, and vector-vector "dot" products are
//! generalized to the tensor contraction over a subset of two tensors' indices,
//! where the result is calculated by summing over the values of those indices
//! and leaving all others untouched.
//!
//! *C*<sub>*a*<sub>1</sub>,...,*a*<sub>*N*</sub>,*b*<sub>1</sub>,...,*b*<sub>*M*</sub></sub>
//!   = Σ<sub>{*α*<sub>1</sub>,...,*α*<sub>*D*</sub>}</sub>
//!     *A*<sub>*a*<sub>1</sub>,...,*a*<sub>*N*</sub>,*α*<sub>1</sub>,...,*α*<sub>*D*</sub></sub>
//!     × *B*<sub>*b*<sub>1</sub>,...,*b*<sub>*M*</sub>,*α*<sub>1</sub>,...,*α*<sub>*D*</sub></sub>
//!
//! Likewise, the Kronecker product is generalized to the tensor product.
//!
//! *C*<sub>*a*<sub>1</sub>,...,*a*<sub>*N*</sub>,*b*<sub>1</sub>,...,*b*<sub>*M*</sub></sub>
//!   = *A*<sub>*a*<sub>1</sub>,...,*a*<sub>*N*</sub></sub>
//!   × *B*<sub>*b*<sub>1</sub>,...,*b*<sub>*M*</sub></sub>

use std::{
    fmt,
    hash::Hash,
    iter::Sum,
    ops::{ Deref, Add, Sub, Mul, Range },
};
use itertools::Itertools;
use ndarray::{ self as nd, Dimension };
use num_complex::Complex64 as C64;
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
///     fn label(&self) -> String { format!("{:?}", self) }
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
///     fn label(&self) -> String { self.0.clone() }
/// }
/// ```
/// This representation allows for dynamic mutation of a tensor network, but
/// comes with a memory overhead and is somewhat less safe than static
/// representations because there are runtime-mutable degrees of freedom.
///
/// Ultimately, static representations should be preferred where possible.
pub trait Idx: Clone + Eq + Hash {
    /// Return the number of values the index can take.
    fn dim(&self) -> usize;

    /// Return an identifying label for the index. This method is used only for
    /// printing purposes.
    fn label(&self) -> String;

    /// Return an iterator over all possible index values. The default
    /// implementation returns `0..self.dim()`.
    fn iter(&self) -> Range<usize> { 0..self.dim() }
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

    fn label(&self) -> String { self.0.clone() }
}

impl<T> From<(T, usize)> for DynIdx
where T: ToString
{
    fn from(x: (T, usize)) -> Self {
        let (label, dim) = x;
        Self(label.to_string(), dim)
    }
}

/// Basic implementation of an abstract tensor quantity.
///
/// A `Tensor<T, A>` consists of some number of quantities of type `A` and a
/// series of indices belonging to a type `T` that implements [`Idx`].
///
/// This implementation distinguishes between rank 0 (scalar) and rank > 0
/// (array) quantities for some small operational benefits, but otherwise lives
/// in an abstraction layer one step above concrete array representations. No
/// parallelism or GPU computation is offered.
#[derive(Clone, PartialEq, Eq)]
pub struct Tensor<T, A>(TensorData<T, A>);

impl<T, A> fmt::Debug for Tensor<T, A>
where
    T: fmt::Debug,
    A: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(")?;
        match &self.0 {
            TensorData::Scalar(_) => {
                self.0.fmt(f)?;
            },
            TensorData::Tensor(..) => {
                writeln!(f)?;
                self.0.fmt(f)?;
                writeln!(f)?;
            },
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl<T, A> fmt::Display for Tensor<T, A>
where
    T: Idx,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, PartialEq, Eq)]
enum TensorData<T, A> {
    Scalar(A),
    Tensor(Vec<T>, nd::ArrayD<A>),
}

impl<T, A> fmt::Debug for TensorData<T, A>
where
    T: fmt::Debug,
    A: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(a) => {
                a.fmt(f)?;
                write!(f, ", type=scalar, rank=0, indices=[]")?;
            },
            Self::Tensor(idxs, a) => {
                a.fmt(f)?;
                write!(
                    f,
                    ",\ntype=tensor, rank={}, indices={:?}",
                    idxs.len(),
                    idxs,
                )?;
            },
        }
        Ok(())
    }
}

impl<T, A> fmt::Display for TensorData<T, A>
where
    T: Idx,
    A: fmt::Display,
{
    #[allow(unused_variables)]
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
                    write!(f, "{}", idx.label())?;
                    if k < n_idxs - 1 { write!(f, ", ")?; }
                }
                write!(f, " }}")?;
            },
        }
        Ok(())
    }
}

/// Iterator type over the indices of a given [`Tensor`].
///
/// The iterator item type is `&T`.
pub struct Indices<'a, T>(IndicesData<'a, T>);

enum IndicesData<'a, T> {
    Scalar,
    Tensor(std::slice::Iter<'a, T>),
}

impl<'a, T> Iterator for Indices<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IndicesData::Scalar => None,
            IndicesData::Tensor(iter) => iter.next(),
        }
    }
}

impl<T, A> TensorData<T, A>
where T: Idx
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
                    |idxs| elems(idxs.as_array_view().as_slice().unwrap()))
                ;
            Ok(Self::Tensor(indices, data))
        }
    }

    fn new_unchecked<I, F>(indices: I, mut elems: F) -> Self
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
                    |idxs| elems(idxs.as_array_view().as_slice().unwrap()))
                ;
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
        Ok(Self::Tensor(indices, array.into_dyn()))
    }

    fn is_scalar(&self) -> bool { matches!(self, Self::Scalar(_)) }

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

    fn do_contract<B, C>(
        idx_common: Vec<T>,
        mut idxs_a: Vec<T>,
        mut a: nd::ArrayD<A>,
        mut idxs_b: Vec<T>,
        mut b: nd::ArrayD<B>,
    ) -> TensorData<T, C>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
        C: Sum,
    {
        // swap common indices and corresponding axes to the leftmost
        // positions to make summing easier
        let mut k_source: usize;
        for (k_target, idx) in idx_common.iter().enumerate() {
            k_source
                = idxs_a.iter()
                .enumerate()
                .find_map(|(k_source, idx_source)| {
                    (idx_source == idx).then_some(k_source)
                })
                .unwrap();
            idxs_a.swap(k_target, k_source);
            a.swap_axes(k_target, k_source);

            k_source
                = idxs_b.iter()
                .enumerate()
                .find_map(|(k_source, idx_source)| {
                    (idx_source == idx).then_some(k_source)
                })
                .unwrap();
            idxs_b.swap(k_target, k_source);
            b.swap_axes(k_target, k_source);
        }

        // construct the new tensor component-wise
        let n_ix: usize = idx_common.len();
        let n_idx_a: usize = idxs_a.len() - n_ix;
        let n_idx_b: usize = idxs_b.len() - n_ix;
        let new_shape: Vec<usize>
            = idxs_a.iter().skip(n_ix)
            .chain(idxs_b.iter().skip(n_ix))
            .map(|idx| idx.dim())
            .collect();
        if new_shape.is_empty() {
            let c: C
                = idx_common.iter()
                .map(|idx| idx.iter())
                .multi_cartesian_product()
                .map(|k_sum| {
                    a[k_sum.deref()].clone() * b[k_sum.deref()].clone()
                })
                .sum();
            TensorData::Scalar(c)
        } else {
            let mut k_res: Vec<usize> = vec![0; n_idx_a + n_idx_b];
            let mut k_a: Vec<usize> = vec![0; n_ix + n_idx_a];
            let mut k_b: Vec<usize> = vec![0; n_ix + n_idx_b];
            let new_data: nd::ArrayD<C>
                = nd::ArrayD::from_shape_fn(
                    new_shape,
                    |k_res_dim| {
                        k_res.iter_mut()
                            .zip(k_res_dim.as_array_view())
                            .for_each(|(k_res_ref, k_res_dim_ref)| {
                                *k_res_ref = *k_res_dim_ref;
                            });
                        k_a.iter_mut()
                            .skip(n_ix)
                            .zip(k_res.iter().take(n_idx_a))
                            .for_each(|(k_a_ref, k_res_ref)| {
                                *k_a_ref = *k_res_ref;
                            });
                        k_b.iter_mut()
                            .skip(n_ix)
                            .zip(k_res.iter().skip(n_idx_a).take(n_idx_b))
                            .for_each(|(k_b_ref, k_res_ref)| {
                                *k_b_ref = *k_res_ref;
                            });
                        idx_common.iter()
                            .map(|idx| idx.iter())
                            .multi_cartesian_product()
                            .map(|k_sum| {
                                k_a.iter_mut()
                                    .take(n_ix)
                                    .zip(k_sum.iter())
                                    .for_each(|(k_a_ref, k_sum_ref)| {
                                        *k_a_ref = *k_sum_ref;
                                    });
                                k_b.iter_mut()
                                    .take(n_ix)
                                    .zip(k_sum.iter())
                                    .for_each(|(k_b_ref, k_sum_ref)| {
                                        *k_b_ref = *k_sum_ref;
                                    });
                                a[k_a.deref()].clone()
                                    * b[k_b.deref()].clone()
                            })
                            .sum::<C>()
                    }
                );
            let new_idxs: Vec<T>
                = idxs_a.into_iter()
                .skip(n_ix)
                .chain(idxs_b.into_iter().skip(n_ix))
                .collect();
            TensorData::Tensor(new_idxs, new_data)
        }
    }

    fn contract<B, C>(self, other: TensorData<T, B>)
        -> TensorResult<TensorData<T, C>>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
        C: Sum,
    {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                Ok(TensorData::Scalar(a * b))
            },
            (TensorData::Scalar(a), TensorData::Tensor(idxs, b)) => {
                let mul: nd::ArrayD<C> = b.mapv(|bk| a.clone() * bk);
                Ok(TensorData::Tensor(idxs, mul))
            },
            (TensorData::Tensor(idxs, a), TensorData::Scalar(b)) => {
                let mul: nd::ArrayD<C> = a.mapv(|ak| ak * b.clone());
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

    fn do_tensor_prod<B, C>(
        mut idxs_a: Vec<T>,
        a: nd::ArrayD<A>,
        mut idxs_b: Vec<T>,
        b: nd::ArrayD<B>,
    ) -> TensorData<T, C>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
    {
        let n_idxs_a = idxs_a.len();
        idxs_a.append(&mut idxs_b);
        TensorData::<T, C>::new_unchecked(
            idxs_a,
            |k_all| {
                a[&k_all[..n_idxs_a]].clone() * b[&k_all[n_idxs_a..]].clone()
            }
        )
    }

    fn tensor_prod<B, C>(self, other: TensorData<T, B>)
        -> TensorResult<TensorData<T, C>>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
    {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                Ok(TensorData::Scalar(a * b))
            },
            (TensorData::Scalar(a), TensorData::Tensor(idxs, b)) => {
                let mul: nd::ArrayD<C> = b.mapv(|bk| a.clone() * bk);
                Ok(TensorData::Tensor(idxs, mul))
            },
            (TensorData::Tensor(idxs, a), TensorData::Scalar(b)) => {
                let mul: nd::ArrayD<C> = a.mapv(|ak| ak * b.clone());
                Ok(TensorData::Tensor(idxs, mul))
            },
            (TensorData::Tensor(idxs_a, a), TensorData::Tensor(idxs_b, b)) => {
                // find common indices, error if some
                let idx_common: Vec<T>
                    = idxs_a.iter()
                    .filter(|idx| idxs_b.contains(idx))
                    .cloned()
                    .collect();
                idx_common.is_empty()
                    .then_some(())
                    .ok_or(MatchingIndices)?;
                Ok(Self::do_tensor_prod(idxs_a, a, idxs_b, b))
            },
        }
    }

    fn multiply<B, C>(self, other: TensorData<T, B>) -> TensorData<T, C>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
        C: Sum,
    {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => {
                TensorData::Scalar(a * b)
            },
            (TensorData::Scalar(a), TensorData::Tensor(idxs, b)) => {
                let mul: nd::ArrayD<C> = b.mapv(|bk| a.clone() * bk);
                TensorData::Tensor(idxs, mul)
            },
            (TensorData::Tensor(idxs, a), TensorData::Scalar(b)) => {
                let mul: nd::ArrayD<C> = a.mapv(|ak| ak * b.clone());
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

    fn add_checked<B, C>(self, other: TensorData<T, B>)
        -> TensorResult<TensorData<T, C>>
    where A: Add<B, Output = C>
    {
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
                TensorData::Tensor(idxs_a, a),
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
                let new_shape: Vec<usize>
                    = idxs_a.iter().map(|idx| idx.dim()).collect();
                let new_data: nd::ArrayD<C>
                    = unsafe {
                        nd::ArrayD::from_shape_vec_unchecked(
                            new_shape,
                            a.into_iter()
                                .zip(b)
                                .map(|(ak, bk)| ak + bk)
                                .collect()
                        )
                    };
                Ok(TensorData::Tensor(idxs_a, new_data))
            },
        }
    }

    fn sub_checked<B, C>(self, other: TensorData<T, B>)
        -> TensorResult<TensorData<T, C>>
    where A: Sub<B, Output = C>
    {
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
                TensorData::Tensor(idxs_a, a),
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
                let new_shape: Vec<usize>
                    = idxs_a.iter().map(|idx| idx.dim()).collect();
                let new_data: nd::ArrayD<C>
                    = unsafe {
                        nd::ArrayD::from_shape_vec_unchecked(
                            new_shape,
                            a.into_iter()
                                .zip(b)
                                .map(|(ak, bk)| ak - bk)
                                .collect()
                        )
                    };
                Ok(TensorData::Tensor(idxs_a, new_data))
            },
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

impl<T> TensorData<T, C64>
where T: Idx
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

impl<T, A, B, C> Mul<TensorData<T, B>> for TensorData<T, A>
where
    T: Idx,
    A: Clone + Mul<B, Output = C>,
    B: Clone,
    C: Sum,
{
    type Output = TensorData<T, C>;

    fn mul(self, other: TensorData<T, B>) -> Self::Output {
        self.multiply(other)
    }
}

impl<T, A, B, C> Add<TensorData<T, B>> for TensorData<T, A>
where
    T: Idx,
    A: Add<B, Output = C>,
{
    type Output = TensorData<T, C>;

    fn add(self, other: TensorData<T, B>) -> Self::Output {
        match self.add_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

impl<T, A, B, C> Sub<TensorData<T, B>> for TensorData<T, A>
where
    T: Idx,
    A: Sub<B, Output = C>,
{
    type Output = TensorData<T, C>;

    fn sub(self, other: TensorData<T, B>) -> Self::Output {
        match self.sub_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

impl<T, A> From<TensorData<T, A>> for Tensor<T, A> {
    fn from(data: TensorData<T, A>) -> Self { Self(data) }
}

impl<T, A> Tensor<T, A>
where T: Idx
{
    /// Create a new tensor using a function over given indices.
    pub fn new<I, F>(indices: I, elems: F) -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        TensorData::new(indices, elems).map(Self::from)
    }

    /// Create a new rank-0 (scalar) tensor.
    pub fn new_scalar(val: A) -> Self { TensorData::new_scalar(val).into() }

    /// Create a new tensor from an n-dimensional array.
    ///
    /// Fails if the dimensions of the array do not match the indices provided.
    pub fn from_array<I, D>(indices: I, array: nd::Array<A, D>)
        -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
        D: nd::Dimension,
    {
        TensorData::from_array(indices, array).map(Self::from)
    }

    /// Return `true` if `self` has rank 0.
    ///
    /// This is equivalent to `self.rank() == 0` and `self.shape().is_empty()`.
    pub fn is_scalar(&self) -> bool { self.0.is_scalar() }

    /// Return `true` if `self` has the given index.
    pub fn has_index(&self, index: &T) -> bool { self.0.has_index(index) }

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

    /// Swap two indices in the underlying representation of `self`.
    ///
    /// Note that this operation has no bearing over the computational
    /// complexity of tensor contractions. If either given index does not exist
    /// within `self`, no swap is performed.
    pub fn swap_indices(&mut self, a: &T, b: &T) { self.0.swap_indices(a, b); }

    /// Contract `self` with `other` over all common indices, consuming both.
    ///
    /// This operation results in a new `Tensor` whose indices comprise all
    /// non-common indices belonging to `self` and `other`. All indices
    /// originally belonging `self` are placed before those from `other`, but
    /// the ordering of indices within these groups is not preserved.
    ///
    /// Fails if no common indices exist.
    pub fn contract<B, C>(self, other: Tensor<T, B>)
        -> TensorResult<Tensor<T, C>>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
        C: Sum,
    {
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
    pub fn tensor_prod<B, C>(self, other: Tensor<T, B>)
        -> TensorResult<Tensor<T, C>>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
        C: Sum,
    {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.tensor_prod(rhs).map(|data| data.into())
    }

    /// Compute the generalized product of `self` and `other`, consuming both.
    ///
    /// This product is an ordinary tensor [contraction][Self::contract] if
    /// common indices exist (or ordinary multiplication if either is a scalar),
    /// otherwise a [tensor product][Self::tensor_prod].
    pub fn multiply<B, C>(self, other: Tensor<T, B>) -> Tensor<T, C>
    where
        A: Clone + Mul<B, Output = C>,
        B: Clone,
        C: Sum,
    {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.multiply(rhs).into()
    }

    /// Compute the sum of `self` and `other`, consuming both.
    ///
    /// Fails if either tensor holds an index not held by the other.
    pub fn add_checked<B, C>(self, other: Tensor<T, B>)
        -> TensorResult<Tensor<T, C>>
    where A: Add<B, Output = C>
    {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.add_checked(rhs).map(|data| data.into())
    }

    /// Compute the difference of `self` and `other`, consuming both.
    ///
    /// Fails if either tensor holds an index not held by the other.
    pub fn sub_checked<B, C>(self, other: Tensor<T, B>)
        -> TensorResult<Tensor<T, C>>
    where A: Sub<B, Output = C>
    {
        let (Tensor(lhs), Tensor(rhs)) = (self, other);
        lhs.sub_checked(rhs).map(|data| data.into())
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
    fn sort_indices(&mut self) { self.0.sort_indices() }
}

impl<T> Tensor<T, C64>
where T: Idx
{
    /// Return a new tensor containing the element-wise conjugation of `self`.
    pub fn conj(&self) -> Self { self.0.conj().into() }
}

impl<T, A, B, C> Mul<Tensor<T, B>> for Tensor<T, A>
where
    T: Idx,
    A: Clone + Mul<B, Output = C>,
    B: Clone,
    C: Sum,
{
    type Output = Tensor<T, C>;

    fn mul(self, other: Tensor<T, B>) -> Self::Output {
        self.multiply(other)
    }
}

impl<T, A, B, C> Add<Tensor<T, B>> for Tensor<T, A>
where
    T: Idx,
    A: Add<B, Output = C>,
{
    type Output = Tensor<T, C>;

    fn add(self, other: Tensor<T, B>) -> Self::Output {
        match self.add_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

impl<T, A, B, C> Sub<Tensor<T, B>> for Tensor<T, A>
where
    T: Idx,
    A: Sub<B, Output = C>,
{
    type Output = Tensor<T, C>;

    fn sub(self, other: Tensor<T, B>) -> Self::Output {
        match self.sub_checked(other) {
            Ok(res) => res,
            Err(err) => panic!("{}", err),
        }
    }
}

