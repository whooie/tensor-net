use std::{
    fmt,
    iter::Sum,
    ops::{ Deref, Mul, Range },
};
use itertools::Itertools;
use ndarray::{ self as nd, Dimension };
use num_complex::Complex64 as C64;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    /// Returned when a contraction is attempted between two tensors (rank > 0)
    /// with no common indices.
    #[error("error in tensor contraction: no matching indices")]
    NoMatchingIndices,
}
pub type TensorResult<T> = Result<T, TensorError>;

/// Describes a tensor index.
///
/// Generally, a tensor network will want to distinguish between the available
/// indices over which its tensors can be contracted. There are two ways to
/// implement this. The first is by using a "static" representation, where all
/// available indices are encoded in the structure of the type implementing this
/// trait:
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
/// #[derive(Clone, Debug, PartialEq, Eq)]
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
pub trait Idx: Clone + Eq {
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

/// Basic implementation of an abstract tensor quantity.
///
/// A `Tensor<T, A>` consists of some number of quantities of type `A` and a
/// series of indices belonging to a type `T` that implements [`Idx`]. The
/// individual elements of a tensor can be accessed directly, but since this
/// library optimizes for use in full tensor networks, this operation is
/// inefficient and discouraged.
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
    fn new<I, F>(indices: I, mut elems: F) -> Self
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
            (
                TensorData::Tensor(mut idxs_a, mut a),
                TensorData::Tensor(mut idxs_b, mut b),
            ) => {
                // find common indices, error if none
                let idx_common: Vec<T>
                    = idxs_a.iter()
                    .filter(|idx| idxs_b.contains(idx))
                    .cloned()
                    .collect();
                (!idx_common.is_empty()).then_some(())
                    .ok_or(TensorError::NoMatchingIndices)?;

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
                    Ok(TensorData::Scalar(c))
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
                    Ok(TensorData::Tensor(new_idxs, new_data))
                }
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

impl<T, A> From<TensorData<T, A>> for Tensor<T, A> {
    fn from(data: TensorData<T, A>) -> Self { Self(data) }
}

impl<T, A> Tensor<T, A>
where T: Idx
{
    /// Create a new tensor using a function over given indices.
    pub fn new<I, F>(indices: I, elems: F) -> Self
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&[usize]) -> A,
    {
        TensorData::new(indices, elems).into()
    }

    /// Create a new rank-0 (scalar) tensor.
    pub fn new_scalar(val: A) -> Self { TensorData::new_scalar(val).into() }

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

