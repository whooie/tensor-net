#![allow(rustdoc::private_intra_doc_links)]

//! Structures allowing (rank-2) [`DMatrix`][na::DMatrix] structures to be used
//! as rank-3 tensors, particularly for use as Γ tensors in a matrix product
//! state.
//!
//! The base mathematical structure here is a rank-3 tensor with indices `u`,
//! `s`, and `v`,
//! ```text
//!  u       v
//! ---- Γ ----
//!      |
//!      | s
//! ```
//! where `s` is the physical index and `u` and `v` are internal bond indices
//! associated with `Γ`'s neighbors in the matrix product state (MPS). In
//! particular, we want to work with the tensor `Γ_usv` (this specific index
//! order is natural to work with, given the well-known cascaded SVD algorithm
//! to convert an ordinary state vector into a MPS).
//!
//! Since `nalgebra` doesn't provide any rank-3 structures, we model `Γ_usv`
//! using an ordinary matrix that can be in one of two forms (see [`GData`]):
//! `Γ_<us>v`, where `s` is fused with `u`; or `Γ_u<sv>`, where `s` is fused
//! with `v`. We can use the fact that `nalgebra` provides a no-copy/no-move
//! [reshaping method][`na::DMatrix::reshape_generic`] to implement the
//! conversion between these two forms cheaply.

use std::{ mem, ops::Add };
use nalgebra as na;
use num_traits::{ Float, Zero };
use crate::ComplexScalar;

/// Basic wrapper enum around a bare `DMatrix`, with variants to keep track of
/// which bond index the physical index is fused to.
///
/// You'll likely want to deal with this data via [`Gamma`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GData<T> {
    /// The physical index is fused with the left bond index.
    LFused(na::DMatrix<T>),
    /// The physical index is fused with the right bond index.
    RFused(na::DMatrix<T>),
}

impl<T> GData<T> {
    /// Return `true` if `self` is `LFused`.
    pub fn is_lfused(&self) -> bool { matches!(self, Self::LFused(..)) }

    /// Return `true` if `self` is `RFused`.
    pub fn is_rfused(&self) -> bool { matches!(self, Self::RFused(..)) }

    /// Discard fusing information and return just the bare matrix.
    pub fn unwrap(self) -> na::DMatrix<T> {
        match self {
            Self::LFused(mat) => mat,
            Self::RFused(mat) => mat,
        }
    }

    /// Return a reference to the underlying matrix.
    pub fn get(&self) -> &na::DMatrix<T> {
        match self {
            Self::LFused(mat) => mat,
            Self::RFused(mat) => mat,
        }
    }

    /// Return a mutable reference to the underlying matrx.
    ///
    /// # Safety
    /// The underlying matrix must not be resized or reshaped.
    pub unsafe fn get_mut(&mut self) -> &mut na::DMatrix<T> {
        match self {
            Self::LFused(mat) => mat,
            Self::RFused(mat) => mat,
        }
    }
}

impl<T: na::ComplexField> GData<T> {
    /// Return the conjugate-transpose of `self`. The fusion direction is
    /// reversed.
    pub fn adjoint(&self) -> Self {
        match self {
            Self::LFused(mat) => Self::RFused(mat.adjoint()),
            Self::RFused(mat) => Self::LFused(mat.adjoint()),
        }
    }
}

/// A single (rank-3) Γ tensor in a matrix product state, represented
/// diagrammatically as
/// ```text
///  u       v
/// ---- Γ ----
///      |
///      | s
/// ```
/// where `s` is the physical index and `u` and `v` are internal bond indices
/// associated with `Γ`'s neighbors. This structure is intended to keep track of
/// which bond index the physical one is fused to, as well as limit mutable
/// access to the underlying matrix.
///
/// Since [`DMatrix`][na::DMatrix] provides methods to reshape or resize the
/// underlying matrix, all methods returning mutable references to it are marked
/// `unsafe`. Any functionality requiring mutation of this data should be
/// implemented as safe methods, lazily converting between left- and right-fused
/// forms as necessary.
///
/// Although the methods `data_mut` and `mat_mut` are provided to give mutable
/// access to the underlying data for completeness, `make_lfused` and
/// `make_rfused` should be preferred since they force a particular tensor
/// representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gamma<T> {
    mdim: usize,
    data: GData<T>,
}

impl<T> Gamma<T> {
    /// Create a new, left-fused Γ tensor, with physical dimension `mdim`.
    ///
    /// *Panics if `mdim` does not evenly divide the number of rows in `data`.*
    pub fn new_lfused(mdim: usize, data: na::DMatrix<T>) -> Self {
        if data.nrows() % mdim != 0 { panic!("inconsistent dimensions"); }
        Self { mdim, data: GData::LFused(data) }
    }

    /// Create a new, right-fused Γ tensor, with physical dimension `mdim`.
    ///
    /// *Panics if `mdim` does not evenly divide the number of columns in
    /// `data`.*
    pub fn new_rfused(mdim: usize, data: na::DMatrix<T>) -> Self {
        if data.ncols() % mdim != 0 { panic!("inconsistent dimensions"); }
        Self { mdim, data: GData::RFused(data) }
    }

    /// Return `true` if `self` is left-fused.
    pub fn is_lfused(&self) -> bool {
        matches!(self.data, GData::LFused(..))
    }

    /// Return `true` if `self` is right-fused.
    pub fn is_rfused(&self) -> bool {
        matches!(self.data, GData::RFused(..))
    }

    /// Return the dimensions of each index.
    pub fn dims(&self) -> (usize, usize, usize) {
        match &self.data {
            GData::LFused(mat) => {
                let (ms, n) = mat.shape();
                (ms / self.mdim, self.mdim, n)
            },
            GData::RFused(mat) => {
                let (m, sn) = mat.shape();
                (m, self.mdim, sn / self.mdim)
            },
        }
    }

    /// Return a reference to the inner data.
    pub fn data(&self) -> &GData<T> { &self.data }

    /// Return a mutable reference to the inner data.
    ///
    /// # Safety
    /// The underlying matrix must not be resized or reshaped.
    pub unsafe fn data_mut(&mut self) -> &mut GData<T> { &mut self.data }

    /// Return a reference to the bare underlying matrix.
    pub fn mat(&self) -> &na::DMatrix<T> { self.data.get() }

    /// Return a mutable reference to the bare underlying matrix.
    ///
    /// # Safety
    /// The matrix must not be resized or reshaped.
    pub unsafe fn mat_mut(&mut self) -> &mut na::DMatrix<T> {
        self.data.get_mut()
    }
}

impl<T: Clone + na::Scalar> Gamma<T> {
    /// Fuse the physical index with the left bond index, returning a mutable
    /// reference to the underlying matrix after.
    ///
    /// # Safety
    /// The matrix must not be resized or reshaped.
    #[allow(clippy::mem_replace_with_uninit)]
    pub unsafe fn make_lfused(&mut self) -> &mut na::DMatrix<T> {
        if self.is_lfused() { return unsafe { self.mat_mut() }; }
        let (m, s, n) = self.dims();
        let owned = unsafe { mem::replace(&mut self.data, mem::zeroed()) };
        let GData::RFused(mat) = owned else { unreachable!() };
        let lfused = mat.reshape_generic(na::Dyn(m * s), na::Dyn(n));
        self.data = GData::LFused(lfused);
        unsafe { self.mat_mut() }
    }

    /// Fuse the physical index with the right bond index, returning a mutable
    /// reference to the underlying matrix after.
    ///
    /// # Safety
    /// The matrix must not be resized or reshaped.
    #[allow(clippy::mem_replace_with_uninit)]
    pub unsafe fn make_rfused(&mut self) -> &mut na::DMatrix<T> {
        if self.is_rfused() { return unsafe { self.mat_mut() }; }
        let (m, s, n) = self.dims();
        let owned = unsafe { mem::replace(&mut self.data, mem::zeroed()) };
        let GData::LFused(mat) = owned else { unreachable!() };
        let rfused = mat.reshape_generic(na::Dyn(m), na::Dyn(s * n));
        self.data = GData::RFused(rfused);
        unsafe { self.mat_mut() }
    }

    /// Return the bare matrix in left-fused form.
    pub fn unwrap_lfused(mut self) -> na::DMatrix<T> {
        unsafe { self.make_lfused(); }
        self.data.unwrap()
    }

    /// Return the bare matrix in right-fused form.
    pub fn unwrap_rfused(mut self) -> na::DMatrix<T> {
        unsafe { self.make_rfused(); }
        self.data.unwrap()
    }
}

impl<T: na::ComplexField> Gamma<T> {
    /// Apply an operator to the physical index in place.
    ///
    /// `op` should be a square matrix, and will be used to left-multiply the
    /// local state vector equivalent; i.e. its columns should correspond to its
    /// "input" index.
    pub fn apply_op(&mut self, op: &na::DMatrix<T>) {
        let (m, s, n) = self.dims();
        if m == 1 {
            let mut owned: na::DMatrix<T> = na::dmatrix!();
            let mat = unsafe { self.make_lfused() };
            mem::swap(mat, &mut owned);
            *mat = op * owned;
        } else if n == 1 {
            let mut owned: na::DMatrix<T> = na::dmatrix!();
            let mat = unsafe { self.make_rfused() };
            mem::swap(mat, &mut owned);
            *mat = owned * op.transpose();
        } else {
            let op_tr = op.transpose();
            let data = unsafe { self.make_lfused() };
            data.column_iter_mut()
                .for_each(|mut col| {
                    let col_mat: na::DMatrix<T> =
                        col.clone_owned()
                        .reshape_generic(na::Dyn(m), na::Dyn(s));
                    let mul = col_mat * &op_tr;
                    let flat =
                        mul.reshape_generic(na::Dyn(m * s), na::Const::<1>);
                    col.copy_from(&flat);
                });
        }
    }

    /// Like `scale_left`, but using SIMD.
    pub fn scale_left<I>(&mut self, weights: I)
    where I: IntoIterator<Item = T::RealField>
    {
        let mat = unsafe { self.make_rfused() };
        mat.row_iter_mut().zip(weights)
            .for_each(|(mut row, w)| { row.scale_mut(w); });
    }

    /// Like `scale_right`, but using SIMD.
    pub fn scale_right<I>(&mut self, weights: I)
    where I: IntoIterator<Item = T::RealField>
    {
        let mat = unsafe { self.make_lfused() };
        mat.column_iter_mut().zip(weights)
            .for_each(|(mut col, w)| { col.scale_mut(w); });
    }

    /// Return the conjugate-transpose of `self`. The fusion direction is
    /// reversed.
    pub fn adjoint(&self) -> Self {
        Gamma { mdim: self.mdim, data: self.data.adjoint() }
    }

    /// Contract two Γ tensors on a shared bond index, with singular values in
    /// between.
    ///
    /// This returns a new Γ tensor with the fusion of both original physical
    /// indices considered as a new, single physical index. The dimensions of
    /// the original indices are returned alongside the new tensor for later
    /// decomposition.
    pub fn contract_bond<I>(mut self, svals: I, mut rhs: Self)
        -> (Self, (usize, usize))
    where I: IntoIterator<Item = T::RealField>
    {
        // nalgebra matrices are column-major, so we should apply the singular
        // values to the right bond index of self; this forces self into
        // left-fused form
        let m = self.dims().0;
        self.scale_right(svals);
        let Self { mdim: lphys, data: GData::LFused(lmat) } = self
            else { unreachable!() };
        let n = rhs.dims().2;
        unsafe { rhs.make_rfused() };
        let Self { mdim: rphys, data: GData::RFused(rmat) } = rhs
            else { unreachable!() };
        let new_phys = self.mdim * rhs.mdim;
        let new_mat =
            (lmat * rmat).reshape_generic(na::Dyn(m * new_phys), na::Dyn(n));
        let new = Self::new_lfused(new_phys, new_mat);
        (new, (lphys, rphys))
    }
}

impl<T: ComplexScalar> Gamma<T> {
    /// Factorize a Γ tensor on two physical indices back into a pair of Γ
    /// tensors with a set of singular values in between. This is the inverse
    /// operation to [`contract_bond`][Self::contract_bond].
    ///
    /// *Panics if `lphys` and `rphys` do not multiply to equal `self.mdim`.
    pub fn factor(
        self,
        lphys: usize,
        rphys: usize,
        trunc: Option<BondDim<T::RealField>>,
    ) -> (Self, na::DVector<T::RealField>, Self) {
        if lphys * rphys != self.mdim { panic!("inconsistent dimensions"); }
        let (m, _, n) = self.dims();
        let mat =
            self.data.unwrap()
            .reshape_generic(na::Dyn(m * lphys), na::Dyn(rphys * n));
        let Schmidt { u, s, q: v, rank: _ } = local_decomp(mat, trunc, false);
        let ltens = Self::new_lfused(lphys, u);
        let rtens = Self::new_rfused(rphys, v);
        (ltens, s, rtens)
    }
}

/// Specify a method to set the bond dimension from a Schmidt decomposition.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BondDim<A> {
    /// Truncate to a fixed upper bound. The resulting bond dimensions will be
    /// *at most* this value and at least 1.
    Const(usize),
    /// Truncate based on a threshold imposed on Schmidt values. For threshold
    /// values small enough, this allows the maximum bond dimension to grow
    /// exponentially with system size.
    Cutoff(A),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schmidt<T: ComplexScalar> {
    pub u: na::DMatrix<T>,
    pub s: na::DVector<T::Re>,
    pub q: na::DMatrix<T>,
    pub rank: usize,
}

pub fn local_decomp<A>(
    q: na::DMatrix<A>,
    trunc: Option<BondDim<A::Re>>,
    remul_svals: bool,
)
    -> Schmidt<A>
where A: ComplexScalar
{
    let svd = q.svd(true, true);
    let Some(mut u) = svd.u else { unreachable!() };
    let mut s = svd.singular_values;
    let Some(mut q) = svd.v_t else { unreachable!() };
    let mut norm: A::Re =
        s.iter()
        .filter(|sj| sj.is_normal())
        .map(|sj| sj.powi(2))
        .fold(A::Re::zero(), Add::add)
        .sqrt();
    s.scale_mut(norm.recip());
    let rank =
        match trunc {
            Some(BondDim::Const(r)) => {
                let eps = A::Re::epsilon();
                let n =
                    s.iter()
                    .take_while(|sj| sj.is_normal() && **sj > eps)
                    .count();
                r.max(1).min(n)
            },
            Some(BondDim::Cutoff(eps)) => {
                let eps = eps.abs();
                s.iter()
                    .take_while(|sj| sj.is_normal() && **sj > eps)
                    .count()
                    .max(1)
            },
            None => {
                let eps = A::Re::epsilon();
                s.iter()
                    .take_while(|sj| sj.is_normal() && **sj > eps)
                    .count()
                    .max(1)
            },
        };
    s.resize_vertically_mut(rank, A::Re::zero());
    norm =
        s.iter()
        .filter(|sj| sj.is_normal())
        .map(|sj| sj.powi(2))
        .fold(A::Re::zero(), Add::add)
        .sqrt();
    s.scale_mut(norm.recip());
    q.resize_vertically_mut(rank, A::zero());
    if remul_svals {
        q.row_iter_mut().zip(&s)
            .for_each(|(mut qv, sv)| { qv.scale_mut(*sv * norm); });
    } else {
        q.scale_mut(norm);
    }
    u.resize_horizontally_mut(rank, A::zero());
    Schmidt { u, s, q, rank }
}

