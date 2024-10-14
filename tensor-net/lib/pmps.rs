//! Like [`mps`][crate::mps], but with periodic boundary conditions.

#![allow(unused_imports)]

use std::{
    cmp::Ordering,
    fmt,
    ops::{ Add, Range },
};
use ndarray as nd;
use ndarray_linalg::{
    Eigh,
    SVDInto,
    UPLO,
    types::{ Scalar, Lapack },
};
use num_complex::{ ComplexFloat, Complex64 as C64 };
use num_traits::{ Float, One, Zero, NumAssign };
use once_cell::sync::Lazy;
use rand::{
    Rng,
    distributions::{ Distribution, Standard },
};
use thiserror::Error;
use crate::{
    mps::{ BondDim, MPS, ComplexLinalgScalar },
    ComplexFloatExt,
    circuit::Q,
    gate::{ self, Gate },
    network::{ Network, Pool },
    tensor::{ Idx, Tensor },
};

/// A matrix product (pure) state with periodic boundary conditions.
///
/// The MPS is maintained in a so-called "canonical" factorization based on the
/// Schmidt decomposition. The Schmidt values are readily available at any given
/// time, which enables efficient calculation of entanglement entropies, at the
/// cost of locally re-computing the decomposition for projective measurements
/// and multi-particle unitaries. Schmidt bases at each site are truncated based
/// on the relevant Schmidt values, however, so for most cases the runtime cost
/// of these decompositions should be relatively small.
///
/// Initial factorization of an arbitrary state should be done via
/// [`MPS::from_vector`], the output of which can then be converted to a `PMPS`.
///
/// A generic network diagram of the MPS is given below, with labeling based on
/// which particles are factored out of the *N*-particle state first. `Γ[k]`
/// matrices represent tensors giving the change of basis from the physical
/// indices to the appropriate Schmidt basis, and `Λ[k]` are vectors holding
/// Schmidt values.
///
/// ```text
///                             .-bond n-1-.
///                             V          V
/// .----------------------------- Λ[n-1] ----------------------------.
/// |                                                                 |
/// |        .-bond 0-.        .-bond 1-.       .-bond n-2-.          |
/// |        V        V        V        V       V          V          |
/// '- Γ[0] --- Λ[0] --- Γ[1] --- Λ[1] --- ... --- Λ[n-2] --- Γ[n-1] -'
///     |                 |                                     |
///     | <- physical     | <- physical                         | <- physical
///          index 0           index 1                               index n-1
/// ```
///
/// > **Note**: If the MPS will ever be converted to a [`Tensor`] or
/// > [`Network`], then all physical indices need to be distinguishable. See
/// > [`Idx`] for more information.
#[derive(Clone, PartialEq)]
pub struct PMPS<T, A>
where A: ComplexLinalgScalar
{
    // Number of particles.
    pub(crate) n: usize, // ≥ 1
    // Tensors for each particle. Array `k` has axis signature
    //   [ u{k - 1}, s{k}, u{k} ]
    // where `u{j}` is an MPS bond index and `s{j}` is a physical index.
    // Endpoint dimensions in the MPS are held fixed with
    //   dim(u{-1}) == dim(u{n - 1}) == 1
    pub(crate) data: Vec<nd::Array3<A>>, // length n
    // Physical indices.
    pub(crate) outs: Vec<T>, // length n
    // Singular values.
    pub(crate) svals: Vec<nd::Array1<A::Re>>, // length n
    // Threshold for singular values.
    pub(crate) trunc: Option<BondDim<A::Re>>,
}

impl<T, A> PMPS<T, A>
where A: ComplexLinalgScalar
{
    /// Add periodic boundary conditions to an arbitrary [`MPS`].
    pub fn from_open(mps: MPS<T, A>) -> Self {
        let MPS { n, data, outs, mut svals, trunc } = mps;
        svals.push(nd::array![A::Re::one()]);
        Self { n, data, outs, svals, trunc }
    }
}

impl<T, A> From<MPS<T, A>> for PMPS<T, A>
where A: ComplexLinalgScalar
{
    fn from(mps: MPS<T, A>) -> Self { Self::from_open(mps) }
}

impl<T, A> PMPS<T, A>
where A: ComplexLinalgScalar
{
    /// Return the number of particles.
    pub fn n(&self) -> usize { self.n }

    /// Return a reference to all physical indices.
    pub fn indices(&self) -> &Vec<T> { &self.outs }
}

impl<T, A> PMPS<T, A>
where A: ComplexLinalgScalar
{
    #[allow(unused_variables)]
    fn contract_bond(&self, k: usize) -> nd::Array3<A> {
        todo!()
    }
}

