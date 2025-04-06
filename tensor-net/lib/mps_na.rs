//! Matrix product states factored in a Schmidt decomposition-based canonical
//! form.
//!
//! In this representation, the quantum state is decomposed particle-by-particle
//! so that all singular values are available at all times. This factorization
//! serves two parimary purposes.
//!
//! First, for general *N*-particle states every decomposition is full-rank, but
//! for a significant portion of the *N*-particle Hilbert space, at least some
//! number of singular values are zero. Hence, the amount of data needed to
//! represent states can be reduced from the usual exponential upper bound to a
//! minimum. This idea can be extended to filter singular values by a non-zero
//! lower limit to further reduce the amount of stored data, at the cost of
//! resorting to an approximation of the state.
//!
//! Second, the singular values found by a Schmidt decomposition across a
//! particular bipartition square to the eigenvalues of the density matrices of
//! either reduced system. This means that the Von Neumann entropy across a
//! bipartition can be readily calculated just by reading off an appropriate set
//! of singular values stored in the state.
//!
//! Matrix product states in this form are represented by a tensor network
//! comprising "Γ" matrices, "Λ" vectors, and two distinct kinds of indices.
//! "Physical" indices are unpaired in the network, and represent the physical
//! degrees of freedom of each particle. "Bonded" indices can be thought of
//! as virtual degrees of freedom between particles, which act to encode quantum
//! correlations. Γ matrices (one for every particle) store information on the
//! local state of each particle and can be thought of as basis transformations
//! from the physical basis to that of a relevant Schmidt basis. Λ vectors exist
//! as weighting factors on bonded indices.
//!
//! ```text
//!       .-bond 0-.        .-bond 1-.       .-bond n-2-.
//!       V        V        V        V       V          V
//! Γ[0] --- Λ[0] --- Γ[1] --- Λ[1] --- ... --- Λ[n-2] --- Γ[n-1]
//!  |                 |                                     |
//!  | <- physical     | <- physical                         | <- physical
//!       index 0           index 1                               index n-1
//! ```
//!
//! # Example
//!
//! ```
//! use nalgebra as na;
//! use num_complex::Complex64 as C64;
//! use num_traits::{ Zero, One };
//! use rand::thread_rng;
//! use tensor_net::mps_na::*;
//! use tensor_net::tensor::Idx;
//!
//! #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
//! struct Q(usize); // `Q(k)` is a qubit degree of freedom for the `k`-th qubit.
//!
//! impl Idx for Q {
//!     fn dim(&self) -> usize { 2 }
//!
//!     fn label(&self) -> String { format!("{self:?}") }
//! }
//!
//! let h: na::DMatrix<C64> =
//!     na::dmatrix!(
//!         C64::from(0.5).sqrt(),  C64::from(0.5).sqrt();
//!         C64::from(0.5).sqrt(), -C64::from(0.5).sqrt();
//!     );
//! let cx: na::DMatrix<C64> =
//!     na::dmatrix!(
//!      // ∣00⟩         ∣10⟩         ∣01⟩         ∣11⟩
//!         C64::one(),  C64::zero(), C64::zero(), C64::zero(); // ∣00⟩
//!         C64::zero(), C64::zero(), C64::zero(), C64::one();  // ∣10⟩
//!         C64::zero(), C64::zero(), C64::one(),  C64::zero(); // ∣01⟩
//!         C64::zero(), C64::one(),  C64::zero(), C64::zero(); // ∣11⟩
//!     );
//!
//! let n: usize = 4; // number of qubits
//! let indices: Vec<Q> = (0..n).map(Q).collect();
//!
//! // initialize to the first available quantum number for all particles,
//! // i.e. ∣0000⟩
//! let mut mps: MPS<Q, C64> = MPS::new(indices, None).unwrap();
//!
//! // apply some gates
//! mps.apply_op1(0, &h).unwrap();
//! mps.apply_op2(0, &cx).unwrap();
//!
//! // perform a randomized projective measurement on qubit zero
//! let mut rng = thread_rng();
//! let outcome = mps.measure(0, &mut rng).unwrap();
//! println!("measured qubit 0 as ∣{outcome}⟩");
//!
//! // contract the MPS into an ordinary 1D state vector
//! let (state, _) = mps.into_vector();
//! println!("{state:.2}");
//! // either [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//! // or     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//! // based on the result of the measurement
//! panic!()
//! ```

#![allow(unused_imports)]

use std::{
    cmp::Ordering,
    fmt,
    mem,
    ops::{ Add, RangeBounds, Bound },
};
use nalgebra as na;
use num_complex::{ ComplexFloat, Complex64 as C64 };
use num_traits::{ Float, One, Zero, NumAssign };
use once_cell::sync::Lazy;
use rand::{
    Rng,
    distributions::{ Distribution, Standard },
};
use thiserror::Error;
use crate::{
    ComplexScalar,
    ComplexFloatExt,
    circuit::Q,
    gamma::{ Gamma, GData, BondDim, Schmidt, local_decomp },
    gate::{ self, Gate },
    tensor::Idx,
};

#[derive(Debug, Error)]
pub enum MPSError {
    /// Returned when attempting to create a new MPS for a state of less than 1
    /// particle.
    #[error("error in MPS creation: cannot create for an empty system")]
    EmptySystem,

    /// Returned when attempting to create a new MPS with an invalid quantum
    /// number.
    #[error("error in MPS creation: invalid quantum number")]
    InvalidQuantumNumber,

    /// Returned when attempting to create a new MPS for a state with a
    /// zero-dimensional physical index.
    #[error("error in MPS creation: unphysical zero-dimensional physical index")]
    UnphysicalIndex,

    /// Returned when attempting to create a new MPS from data with a
    /// length/shape that doesn't match the provided indices.
    #[error("error in MPS creation: array length/shape doesn't match indices")]
    StateIncompatibleShape,

    /// Returned when attempting to apply an operator to an MPS with a matrix
    /// whose dimensions do not agree with relevant physical indice(s).
    #[error("error in operator application: incorrect shape")]
    OperatorIncompatibleShape,
}
use MPSError::*;
pub type MPSResult<T> = Result<T, MPSError>;

/// A (pure) matrix product state.
///
/// The MPS is maintained in a so-called "canonical" factorization based on the
/// Schmidt decomposition. The Schmidt values are readily available at any given
/// time, which enables efficient calculation of entanglement entropys, at the
/// cost of locally re-computing the decomposition for projective measurements
/// and multi-particle unitaries.
///
/// Initial factorization of an arbitrary state (see
/// [`from_vector`][Self::from_vector]) happens via a series of singular value
/// decompositions, which uncurs a total runtime cost that os *O*(*n*
/// *D*<sup>*n*</sup>), where *n* is the number of particles and *D* is the
/// maximum dimension of their quantum numbers/indices.
///
/// A generic network diagram of the MPS is given below, with labeling based on
/// which particles are factored out of the *N*-particle state first. `Γ[k]`
/// matrices represent tensors giving the change of basis from the physical
/// indices to the appropriate Schmidt basis, and `Λ[k]` are vectors holding
/// Schmidt values.
///
/// ```text
///       .-bond 0-.        .-bond 1-.       .-bond n-2-.
///       V        V        V        V       V          V
/// Γ[0] --- Λ[0] --- Γ[1] --- Λ[1] --- ... --- Λ[n-2] --- Γ[n-1]
///  |                 |                                     |
///  | <- physical     | <- physical                         | <- physical
///       index 0           index 1                               index n-1
/// ```
#[derive(Clone, PartialEq)]
pub struct MPS<T, A>
where A: ComplexScalar
{
    // Number of particles.
    pub(crate) n: usize, // ≥ 1
    // Tensors for each particle. Array `k` has axis signature
    //   [ u{k - 1}, s{k}, u{k} ]
    // where `u{j}` is an MPS bond index and `s{j}` is a physical index.
    // Endpoint dimensions in the MPS are held fixed with
    //   dim(u{-1}) == dim(u{n - 1}) == 1
    pub(crate) data: Vec<Gamma<A>>, // length n
    // Physical indices.
    pub(crate) idxs: Vec<T>, // length n
    // Singular values.
    pub(crate) svals: Vec<na::DVector<A::Re>>, // length n - 1
    // Threshold for singular values.
    pub(crate) trunc: Option<BondDim<A::Re>>,
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexScalar,
{
    /// Initialize to a separable product state with each particle in the first
    /// of its available eigenstates.
    ///
    /// Optionally provide a global truncation method for discarding singular
    /// values. Defaults to a [`Cutoff`][BondDim::Cutoff] at machine epsilon for
    /// the numerical type `A`.
    ///
    /// Fails if no physical indices are provided or an unphysical
    /// (zero-dimensional) index is encountered.
    pub fn new<I>(indices: I, trunc: Option<BondDim<A::Re>>)
        -> MPSResult<Self>
    where I: IntoIterator<Item = T>
    {
        let indices: Vec<T> = indices.into_iter().collect();
        if indices.is_empty() { return Err(EmptySystem); }
        let n = indices.len();
        let data: Vec<Gamma<A>> =
            indices.iter()
            .map(|idx| {
                let dim = idx.dim();
                if dim == 0 {
                    Err(UnphysicalIndex)
                } else {
                    let mut g: na::DMatrix<A> = na::DMatrix::zeros(dim, 1);
                    if let Some(g000) = g.get_mut((0, 0)) { *g000 = A::one(); }
                    Ok(Gamma::new_lfused(dim, g))
                }
            })
            .collect::<MPSResult<Vec<_>>>()?;
        let svals: Vec<na::DVector<A::Re>> =
            (0 .. n - 1)
            .map(|_| na::dvector!(A::Re::one()))
            .collect();
        Ok(Self { n, data, idxs: indices, svals, trunc })
    }

    /// Initialize to a separable product state with each particle wholly in one
    /// of its available eigenstates.
    ///
    /// Optionally provide a global truncation method for discarding singular
    /// values. Defaults to a [`Cutoff`][BondDim::Cutoff] at machine epsilon for
    /// the numerical type `A`.
    ///
    /// Fails if no physical indices are provided, an unphysical
    /// (zero-dimensional) index is encountered, or the given quantum number for
    /// an index exceeds the available range defined by the index's [`Idx`]
    /// implementation.
    pub fn new_qnums<I>(indices: I, trunc: Option<BondDim<A::Re>>)
        -> MPSResult<Self>
    where I: IntoIterator<Item = (T, usize)>
    {
        let (indices, qnums): (Vec<T>, Vec<usize>) =
            indices.into_iter().unzip();
        if indices.is_empty() { return Err(EmptySystem); }
        if indices.iter().zip(&qnums).any(|(idx, qnum)| *qnum >= idx.dim()) {
            return Err(InvalidQuantumNumber);
        }
        let n = indices.len();
        let data: Vec<Gamma<A>> =
            indices.iter().zip(&qnums)
            .map(|(idx, qnum)| {
                let dim = idx.dim();
                if dim == 0 {
                    Err(UnphysicalIndex)
                } else {
                    let mut g: na::DMatrix<A> = na::DMatrix::zeros(dim, 1);
                    if let Some(g) = g.get_mut((*qnum, 0)) { *g = A::one(); }
                    Ok(Gamma::new_lfused(dim, g))
                }
            })
            .collect::<MPSResult<Vec<_>>>()?;
        let svals: Vec<na::DVector<A::Re>> =
            (0 .. n - 1)
            .map(|_| na::dvector!(A::Re::one()))
            .collect();
        Ok(Self { n, data, idxs: indices, svals, trunc })
    }

    fn factorize(
        idxs: &[T],
        state: na::DVector<A>,
        trunc: Option<BondDim<A::Re>>,
    ) -> (Vec<Gamma<A>>, Vec<na::DVector<A::Re>>)
    {
        let n = idxs.len(); // assume n ≥ 1
        let mut data: Vec<Gamma<A>> = Vec::with_capacity(n);
        let mut svals: Vec<na::DVector<A::Re>> = Vec::with_capacity(n - 1);
        let mut udim: usize = 1;
        let mut outdim: usize;
        let mut statelen = state.len();
        let mut q: na::DMatrix<A> =
            state.reshape_generic(na::Dyn(1), na::Dyn(statelen));
        let mut reshape_m: usize;
        let mut reshape_n: usize;
        for outk in idxs.iter().take(n - 1) {
            outdim = outk.dim();
            // initial reshape to fuse outk with the previous schmidt index
            statelen = q.len();
            reshape_m = udim * outdim;
            reshape_n = statelen / reshape_m;
            q = q.reshape_generic(na::Dyn(reshape_m), na::Dyn(reshape_n));

            // svd/schmidt decomp: left vectors are columns in u, schmidt values are
            // in s, and right vectors are rows in q
            //
            // q is returned here with schmidt index scaled by schmidt values
            let Schmidt { u, s, q: qp, rank } = local_decomp(q, trunc, true);
            q = qp;
            let mut g = Gamma::new_lfused(outdim, u);
            if let Some(slast) = svals.last() {
                g.scale_left(slast.iter().copied());
            }
            data.push(g);
            svals.push(s);
            udim = rank;
        }
        // final Γ tensor
        let mut g = Gamma::new_lfused(idxs[n - 1].dim(), q);
        if n > 1 { g.scale_left(svals[n - 2].iter().copied()); }
        data.push(g);
        (data, svals)
    }

    /// Initialize by factoring an existing pure state vector.
    ///
    /// Optionally provide a global truncation method for discarding singular
    /// values. Defaults to a [`Cutoff`][BondDim::Cutoff] at machine epsilon for
    /// the numerical type `A`.
    ///
    /// **Note:** The state vector should be in *column-major order*. Although
    /// vectors are one-column data structures, this still has implications for
    /// the ordering of individual elements: If you are used to row-major
    /// formats, you will likely order your basis element such that the
    /// right-most index/quantum number varies the fastest. In column-major
    /// order, it is the *left-most* that varies the fastest. For, e.g., *n*
    /// qubits, this is analogous to the endian-ness of the *n*-bit registers:
    ///
    /// | Array index | Basis element (row-major) | Basis element (column-major) |
    /// | :---------- | :------------------------ | :--------------------------- |
    /// | 0           | ∣00...00⟩                 | ∣00...00⟩                    |
    /// | 1           | ∣00...01⟩                 | ∣10...00⟩                    |
    /// | 2           | ∣00...10⟩                 | ∣01...00⟩                    |
    /// | 3           | ∣00...11⟩                 | ∣11...00⟩                    |
    /// | ...         | ...                       | ...                          |
    /// | *n* – 2     | ∣11...10⟩                 | ∣01...11⟩                    |
    /// | *n* – 1     | ∣11...11⟩                 | ∣11...11⟩                    |
    ///
    /// Fails if no particle indices are provided, an unphysical
    /// (zero-dimensional) index is encountered, or the length of the initial
    /// state vector does not agree with the indices.
    pub fn from_vector<I, J>(
        indices: I,
        state: J,
        trunc: Option<BondDim<A::Re>>,
    ) -> MPSResult<Self>
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator<Item = A>,
    {
        let indices: Vec<T> = indices.into_iter().collect();
        if indices.is_empty() { return Err(EmptySystem); }
        if indices.iter().any(|i| i.dim() == 0) { return Err(UnphysicalIndex); }
        let statelen: usize = indices.iter().map(|idx| idx.dim()).product();
        let state: Vec<A> = state.into_iter().collect();
        if state.len() != statelen { return Err(StateIncompatibleShape); }
        let mut state = na::DVector::from_vec(state);
        let norm: A::Re =
            state.iter()
            .map(|a| a.modulus_squared())
            .fold(A::Re::zero(), A::Re::add)
            .sqrt();
        state.scale_mut(norm.recip());
        let n = indices.len();
        if n == 1 {
            let g = state.reshape_generic(na::Dyn(statelen), na::Dyn(1));
            let data = vec![Gamma::new_lfused(statelen, g)];
            let svals: Vec<na::DVector<A::Re>> = Vec::new();
            Ok(Self { n, data, idxs: indices, svals, trunc })
        } else {
            let (data, svals) = Self::factorize(&indices, state, trunc);
            Ok(Self { n, data, idxs: indices, svals, trunc })
        }
    }

    /// Contract the entire state into a bare vector and return the result
    /// alongside all physical indices.
    pub fn into_vector(self) -> (na::DVector<A>, Vec<T>) {
        let Self { n: _, data, idxs, svals, trunc: _ } = self;
        let statelen = idxs.iter().map(|idx| idx.dim()).product();
        let mut gamma_iter = data.into_iter();
        let Some(q0) = gamma_iter.next() else { unreachable!() };
        let statevec: na::DVector<A> =
            svals.into_iter().zip(gamma_iter)
            .fold(q0, |q, (s, g)| q.contract_bond(s.iter().copied(), g).0)
            .unwrap_lfused()
            .reshape_generic(na::Dyn(statelen), na::Const::<1>);
        (statevec, idxs)
    }

    /// Contract into a reduced density matrix over a contiguous subset of
    /// particles, and return the result alongside the corresponding physical
    /// indices.
    #[allow(clippy::needless_late_init)]
    pub fn into_part<P>(mut self, part: P) -> (na::DMatrix<A>, Vec<T>)
    where P: RangeBounds<usize>
    {
        let i0: usize =
            match part.start_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) => i.saturating_add(1),
                Bound::Unbounded   => 0,
            };
        let i1: usize =
            match part.end_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) => i.saturating_sub(1),
                Bound::Unbounded   => self.n - 1,
            };
        if i0 > i1 {
            return (na::dmatrix!(A::one()), Vec::new());
        }
        let mut rho: na::DMatrix<A>;
        match (i0 == 0, i1 == self.n - 1) {
            (true, true) => {
                let (state, idxs) = self.into_vector();
                rho = state.kronecker(&state.adjoint());
                return (rho, idxs);
            },
            (true, false) => {
                let mut gamma_iter = self.data.drain(i0..=i1);
                let Some(q0) = gamma_iter.next() else { unreachable!() };
                let part: Gamma<A> =
                    self.svals[i0..i1].iter().zip(gamma_iter)
                    .fold(q0, |q, (s, g)| q.contract_bond(s.iter().copied(), g).0);
                let part_conj = part.adjoint();
                let (_, dim, _) = part.dims();
                rho =
                    part.contract_bond(
                        self.svals[i1].iter().map(|s| s.powi(2)), part_conj).0
                    .unwrap_lfused()
                    .reshape_generic(na::Dyn(dim), na::Dyn(dim));
            },
            (false, true) => {
                let mut gamma_iter = self.data.drain(i0..=i1).rev();
                let Some(q1) = gamma_iter.next() else { unreachable!() };
                let part: Gamma<A> =
                    gamma_iter.zip(self.svals[i0..i1].iter().rev())
                    .fold(q1, |q, (g, s)| g.contract_bond(s.iter().copied(), q).0);
                let part_conj = part.adjoint();
                let (_, dim, _) = part.dims();
                rho =
                    part_conj.contract_bond(
                        self.svals[i0 - 1].iter().map(|s| s.powi(2)), part).0
                    .unwrap_lfused()
                    .reshape_generic(na::Dyn(dim), na::Dyn(dim));
                rho.transpose_mut();
            },
            (false, false) => {
                let mut gamma_iter = self.data.drain(i0..=i1);
                let Some(mut q0) = gamma_iter.next() else { unreachable!() };
                q0.scale_left(self.svals[i0 - 1].iter().copied());
                let part: Gamma<A> =
                    self.svals[i0..i1].iter().zip(gamma_iter)
                    .fold(q0, |q, (s, g)| q.contract_bond(s.iter().copied(), g).0);
                let part_conj = part.adjoint();
                let (m, dim, n) = part.dims();
                let rho_big =
                    if m > n {
                        let mut tr =
                            part_conj.contract_bond(
                                self.svals[i0 - 1].iter().map(|s| s.powi(2)), part).0
                            .unwrap_lfused()
                            .reshape_generic(na::Dyn(n * dim), na::Dyn(n * dim));
                        tr.transpose_mut();
                        tr
                    } else {
                        part.contract_bond(
                            self.svals[i1].iter().map(|s| s.powi(2)), part_conj).0
                        .unwrap_lfused()
                        .reshape_generic(na::Dyn(m * dim), na::Dyn(m * dim))
                    };
                rho = na::DMatrix::zeros(dim, dim);
                let d = m.min(n);
                for j in 0..d {
                    rho += rho_big.view_with_steps(
                        (j, j), (dim, dim), (d - 1, d - 1));
                }
            },
        }
        let idxs: Vec<T> = self.idxs.drain(i0..=i1).collect();
        (rho, idxs)
    }
}

impl<T, A> MPS<T, A>
where A: ComplexScalar
{
    /// Return the number of particles.
    pub fn n(&self) -> usize { self.n }

    /// Return a reference to all physical indices.
    pub fn indices(&self) -> &Vec<T> { &self.idxs }

    /// Return the current bond dimension truncation setting.
    pub fn get_trunc(&self) -> Option<BondDim<A::Re>> { self.trunc }

    /// Set a new bond dimension truncation setting.
    pub fn set_trunc(&mut self, trunc: Option<BondDim<A::Re>>) {
        self.trunc = trunc;
    }

    /// Return the maximum bond dimension in the MPS.
    ///
    /// This function will always return `Some` if `self` comprises at least two
    /// particles.
    pub fn max_bond_dim(&self) -> Option<usize> {
        self.svals.iter().map(|s| s.len()).max()
    }

    /// Return the index of the bond with largest dimension in the
    /// MPS.
    ///
    /// This function will always return `Some` if `self` comprises at least two
    /// particles.
    pub fn max_bond_dim_idx(&self) -> Option<usize> {
        self.svals.iter().enumerate()
            .max_by_key(|(_, s)| s.len())
            .map(|(k, _)| k)
    }

    /// Return the dimension of the bond between the `k`-th and `k + 1`-th
    /// particles.
    pub fn bond_dim(&self, k: usize) -> Option<usize> {
        self.svals.get(k).map(|s| s.len())
    }

    /// Compute the Von Neumann entropy across a bipartition placed on the
    /// `b`-th bond.
    ///
    /// Returns `None` if `b` is out of bounds.
    pub fn entropy_vn(&self, b: usize) -> Option<A::Re> {
        self.svals.get(b)
            .map(|s| {
                s.iter().copied()
                .map(|sk| {
                    let sk2 = sk * sk;
                    -sk2 * sk2.ln()
                })
                .fold(A::Re::zero(), A::Re::add)
            })
    }

    /// Compute the `a`-th Rényi entropy across a bipartition placed on the
    /// `b`-th bond in the Schmidt basis.
    ///
    /// Returns the Von-Neumann entropy for `a == 1` and `None` if `b` is out of
    /// bounds.
    pub fn entropy_ry_schmidt(&self, a: A::Re, b: usize) -> Option<A::Re> {
        if a.is_one() {
            self.entropy_vn(b)
        } else {
            self.svals.get(b)
                .map(|s| {
                    s.iter().copied()
                    .map(|sk| (sk * sk).powf(a))
                    .fold(A::Re::zero(), A::Re::add)
                    .ln()
                    * (A::Re::one() - a).recip()
                })
        }
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexScalar,
{
    // calculate the norm of the subspace belonging to the `k` particle
    // assumes `k` is in bounds
    fn local_norm(&mut self, k: usize) -> A::Re {
        if self.n == 1 {
            let g = self.data[k].mat();
            g.dotc(g).real().sqrt()
        } else if k == 0 {
            unsafe {
                self.data[k].make_lfused()
                .column_iter()
                .zip(&self.svals[k])
                .fold(A::Re::zero(), |acc, (g0_v, lv)| {
                    g0_v.dotc(&g0_v).real() * lv.powi(2)
                    + acc
                })
                .sqrt()
            }
        } else if k == self.n - 1 {
            unsafe {
                self.data[k].make_rfused()
                .row_iter()
                .zip(&self.svals[k - 1])
                .fold(A::Re::zero(), |acc, (gu_0, lu)| {
                    lu.powi(2) * gu_0.dotc(&gu_0).real()
                    + acc
                })
                .sqrt()
            }
        } else {
            let (m, s, _) = self.data[k].dims();
            unsafe {
                self.data[k].make_lfused()
                .column_iter()
                .zip(&self.svals[k])
                .fold(A::Re::zero(), |acc, (g__v, rv)| {
                    g__v.reshape_generic(na::Dyn(m), na::Dyn(s))
                    .row_iter()
                    .zip(&self.svals[k - 1])
                    .fold(A::Re::zero(), |acc, (gu_v, lu)| {
                        lu.powi(2) * gu_v.dotc(&gu_v).real()
                        + acc
                    })
                    * rv.powi(2)
                    + acc
                })
                .sqrt()
            }
        }
    }

    // renormalizes the Γ tensor belonging to particle `k`
    //
    // assumes `k` is in bounds
    fn local_renormalize(&mut self, k: usize) {
        let norm = self.local_norm(k);
        unsafe { self.data[k].mat_mut().scale_mut(norm.recip()); }
    }

    // calculate the probability of the `k`-th particle being in its `p`-th
    // state
    //
    // assumes `k` and `p` are in bounds
    fn local_prob(&self, k: usize, p: usize) -> A::Re {
        if self.n == 1 {
            match self.data[k].data() {
                GData::LFused(g) => g.get((p, 0)).unwrap().modulus_squared(),
                GData::RFused(g) => g.get((0, p)).unwrap().modulus_squared(),
            }
        } else if k == 0 {
            unsafe {
                make_mut(&self.data[k]).make_lfused()
                .row(p)
                .iter()
                .zip(&self.svals[k])
                .fold(A::Re::zero(), |acc, (g0pv, lv)| {
                    g0pv.modulus_squared() * lv.powi(2)
                    + acc
                })
            }
        } else if k == self.n - 1 {
            unsafe {
                make_mut(&self.data[k]).make_rfused()
                .column(p)
                .iter()
                .zip(&self.svals[k - 1])
                .fold(A::Re::zero(), |acc, (gup0, lu)| {
                    lu.powi(2) * gup0.modulus_squared()
                    + acc
                })
            }
        } else {
            let (m, s, _) = self.data[k].dims();
            unsafe {
                make_mut(&self.data[k]).make_lfused()
                .column_iter()
                .zip(&self.svals[k])
                .fold(A::Re::zero(), |acc, (g__v, rv)| {
                    g__v.reshape_generic(na::Dyn(m), na::Dyn(s))
                    .column(p)
                    .iter()
                    .zip(&self.svals[k - 1])
                    .fold(A::Re::zero(), |acc, (gupv, lu)| {
                        lu.powi(2) * gupv.modulus_squared()
                        + acc
                    })
                    * rv.powi(2)
                    + acc
                })
            }
        }
    }

    /// Return the probability of the `k`-th particle being in its `p`-th state.
    ///
    /// Returns `None` if `k` or `p` is out of bounds.
    pub fn prob(&self, k: usize, p: usize) -> Option<A::Re> {
        if self.idxs.get(k).is_none_or(|idx| p >= idx.dim()) { return None; }
        Some(self.local_prob(k, p))
    }

    // calculates the probabilities for all available quantum numbers for the
    // `k`-th particle
    //
    // assumes `k` is in bounds
    fn local_probs(&self, k: usize) -> Vec<A::Re> {
        if self.n == 1 {
            let g = self.data[k].mat();
            g.iter().map(|gs| gs.modulus_squared()).collect()
        } else if k == 0 {
            unsafe {
                make_mut(&self.data[k]).make_lfused()
                .row_iter()
                .map(|g0s_| {
                    g0s_.iter().zip(&self.svals[k])
                    .fold(A::Re::zero(), |acc, (g0sv, lv)| {
                        g0sv.modulus_squared() * lv.powi(2)
                        + acc
                    })
                })
                .collect()
            }
        } else if k == self.n - 1 {
            unsafe {
                make_mut(&self.data[k]).make_rfused()
                .column_iter()
                .map(|g_s0| {
                    g_s0.iter().zip(&self.svals[k - 1])
                    .fold(A::Re::zero(), |acc, (gus0, lu)| {
                        lu.powi(2) * gus0.modulus_squared()
                        + acc
                    })
                })
                .collect()
            }
        } else {
            let (m, s, _) = self.data[k].dims();
            unsafe {
                make_mut(&self.data[k]).make_lfused()
                .column_iter()
                .zip(&self.svals[k])
                .fold(vec![A::Re::zero(); s], |mut acc, (g__v, rv)| {
                    g__v.reshape_generic(na::Dyn(m), na::Dyn(s))
                    .column_iter()
                    .zip(acc.iter_mut())
                    .for_each(|(g_sv, accs)| {
                        *accs +=
                            g_sv.iter().zip(&self.svals[k - 1])
                            .fold(A::Re::zero(), |acc, (gusv, lu)| {
                                lu.powi(2) * gusv.modulus_squared()
                                + acc
                            })
                            * rv.powi(2);
                    });
                    acc
                })
            }
        }
    }

    /// Return the probabilities of the `k`-th particle being in each of its
    /// available states.
    ///
    /// Returns `None` if `k` is out of bounds.
    pub fn probs(&self, k: usize) -> Option<Vec<A::Re>> {
        if k >= self.n { return None; }
        Some(self.local_probs(k))
    }

    /// Evaluate the expectation value of a local operator acting (only) on the
    /// `k`-th particle.
    ///
    /// Returns zero if `k` is out of bounds. The arrangement of the elements of
    /// `op` should correspond to the usual left-matrix-multiplication view of
    /// operator application.
    ///
    /// Fails if `op` is not square with a dimension equal to that of the `k`-th
    /// index.
    pub fn expectation_value(&self, k: usize, op: &na::DMatrix<A>)
        -> MPSResult<A>
    {
        if k >= self.n { return Ok(A::zero()); }
        let dk = self.idxs[k].dim();
        if op.shape() != (dk, dk) { return Err(OperatorIncompatibleShape); }
        // this is pretty ugly, but I think it's actually faster than trying to
        // use matmul; maybe we might be able to play around with cloning into
        // owned arrays and using SIMD for the singular values
        if self.n == 0 {
            let g = self.data[k].mat();
            let ev =
                g.iter()
                .zip(op.column_iter())
                .fold(A::zero(), |acc, (g0s0, op_s)| {
                    g.iter()
                    .zip(op_s.iter())
                    .fold(A::zero(), |acc, (g0t0, opts)| {
                        g0t0.conj() * *opts
                        + acc
                    })
                    * *g0s0
                    + acc
                });
            Ok(ev)
        } else if k == 0 {
            let ev = unsafe {
                let g = make_mut(&self.data[k]).make_lfused();
                g.row_iter()
                .zip(op.column_iter())
                .fold(A::zero(), |acc, (g0s_, op_s)| {
                    g.row_iter()
                    .zip(op_s.iter())
                    .fold(A::zero(), |acc, (g0t_, opts)| {
                        g0s_.iter()
                        .zip(g0t_.iter())
                        .zip(&self.svals[k])
                        .fold(A::zero(), |acc, ((g0sv, g0tv), lv)| {
                            g0tv.conj() * *g0sv * A::from_re(lv.powi(2))
                            + acc
                        })
                        * *opts
                        + acc
                    })
                    + acc
                })
            };
            Ok(ev)
        } else if k == self.n - 1 {
            let ev = unsafe {
                let g = make_mut(&self.data[k]).make_rfused();
                g.column_iter()
                .zip(op.column_iter())
                .fold(A::zero(), |acc, (g_s0, op_s)| {
                    g.column_iter()
                    .zip(op_s.iter())
                    .fold(A::zero(), |acc, (g_t0, opts)| {
                        g_s0.iter()
                        .zip(g_t0.iter())
                        .zip(&self.svals[k - 1])
                        .fold(A::zero(), |acc, ((gus0, gut0), lu)| {
                            gut0.conj() * *gus0 * A::from_re(lu.powi(2))
                            + acc
                        })
                        * *opts
                        + acc
                    })
                    + acc
                })
            };
            Ok(ev)
        } else {
            let ev = unsafe {
                let (m, s, _) = self.data[k].dims();
                let g = make_mut(&self.data[k]).make_lfused();
                g.column_iter()
                .zip(&self.svals[k])
                .fold(A::zero(), |acc, (g__v, rv)| {
                    let g__v = g__v.reshape_generic(na::Dyn(m), na::Dyn(s));
                    g__v.column_iter()
                    .zip(op.column_iter())
                    .fold(A::zero(), |acc, (g_sv, op_s)| {
                        g__v.column_iter()
                        .zip(op_s.iter())
                        .fold(A::zero(), |acc, (g_tv, opts)| {
                            g_sv.iter()
                            .zip(g_tv.iter())
                            .zip(&self.svals[k - 1])
                            .fold(A::zero(), |acc, ((gusv, gutv), lu)| {
                                gutv.conj() * *gusv * A::from_re(lu.powi(2))
                                + acc
                            })
                            * *opts
                            + acc
                        })
                        + acc
                    })
                    * A::from_re(rv.powi(2))
                    + acc
                })
            };
            Ok(ev)
        }
    }

    // contract a pair of Γ tensors across the `k`-th bond, perform an action on
    // the resulting local two-particle state, and then refactor
    //
    // if available, the singular values on either side of the pair are
    // multiplied into the two-particle state before the action is performed,
    // and then divided out afterward
    //
    // assumes `k` is in bounds (i.e. `0..=n-2`)
    fn map_pair<F>(&mut self, k: usize, f: F)
    where F: FnOnce(Gamma<A>) -> Gamma<A>
    {
        unsafe {
            let gl = replace_zeroed(&mut self.data[k]);
            let gr = replace_zeroed(&mut self.data[k + 1]);
            let (mut g2, (lphys, rphys)) =
                gl.contract_bond(self.svals[k].iter().copied(), gr);
            if k != 0 {
                g2.scale_left(self.svals[k - 1].iter().copied());
            }
            if k != self.n - 2 {
                g2.scale_right(self.svals[k + 1].iter().copied());
            }
            let mut g2_new = f(g2);
            if k != 0 {
                g2_new.scale_left(
                    self.svals[k - 1].iter().copied().map(|s| s.recip()));
            }
            if k != self.n - 2 {
                g2_new.scale_right(
                    self.svals[k + 1].iter().copied().map(|s| s.recip()));
            }
            let (gl_new, lk_new, gr_new) =
                g2_new.factor(lphys, rphys, self.trunc);
            let _ = mem::replace(&mut self.data[k], gl_new);
            self.svals[k] = lk_new;
            let _ = mem::replace(&mut self.data[k + 1], gr_new);
        }
    }

    // contract the `k`-th and `k + 1`-th Γ tensors (with neighboring singular
    // values if applicable and refactor to re-construct the bond between only
    // those sites
    //
    // assumes `k` and `k + 1` are in bounds
    fn local_refactor(&mut self, k: usize) {
        self.map_pair(k, |g| g);
        self.local_renormalize(k);
        self.local_renormalize(k + 1);
    }

    // apply a full bidirectional refactoring sweep of the MPS, avoiding the
    // double-refactor of the last bond
    //
    // all sites are renormalized after
    fn refactor_sweep(&mut self) {
        for k in  0 .. self.n - 1        { self.map_pair(k, |g| g); }
        for k in (0 .. self.n - 2).rev() { self.map_pair(k, |g| g); }
        for k in  0 .. self.n            { self.local_renormalize(k); }
    }

    /// Apply an operator to the `k`-th particle in place.
    ///
    /// Does nothing if `k` is out of bounds. The arrangement of the elements of
    /// `op` should correspond to the usual left-matrix-multiplication view of
    /// operator application.
    ///
    /// Fails if `op` is not square with a dimension equal to that of the `k`-th
    /// physical index.
    pub fn apply_op1(&mut self, k: usize, op: &na::DMatrix<A>)
        -> MPSResult<&mut Self>
    {
        if k >= self.n { return Ok(self); }
        let dk = self.idxs[k].dim();
        if op.shape() != (dk, dk) { return Err(OperatorIncompatibleShape); }
        self.data[k].apply_op(op);
        Ok(self)
    }

    /// Apply an operator to the `k`-th and `k + 1`-th particles in place.
    ///
    /// Does nothing if either `k` or `k + 1` are out of bounds. The arrangement
    /// of the elements of `op` should correspond to the usual
    /// left-matrix-multiplication view of operator application.
    ///
    /// Fails if `op` is not square with a dimension equal to the product of
    /// those of the `k`-th and `k + 1`-th physical indices.
    pub fn apply_op2(&mut self, k: usize, op: &na::DMatrix<A>)
        -> MPSResult<&mut Self>
    {
        if self.n == 1 || k >= self.n - 1 { return Ok(self); }
        let dk = self.idxs[k].dim();
        let dkp1 = self.idxs[k + 1].dim();
        if op.shape() != (dk * dkp1, dk * dkp1) {
            return Err(OperatorIncompatibleShape);
        }
        self.map_pair(k, |mut g| { g.apply_op(op); g });
        Ok(self)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexScalar,
    Standard: Distribution<A::Re>,
{
    // calculate the state probabilities for the `k`-th particle and randomly
    // sample one from that distribution, returning the quantum number and its
    // probability
    //
    // assumes `k` is in bounds
    fn sample_state<R>(&self, k: usize, rng: &mut R) -> (usize, A::Re)
    where R: Rng + ?Sized
    {
        let r: A::Re = rng.gen();
        let probs: Vec<A::Re> = self.local_probs(k);
        let p =
            probs.iter().copied()
            .scan(A::Re::zero(), |cu, pr| { *cu += pr; Some(*cu) })
            .enumerate()
            .find_map(|(j, cuprob)| (r < cuprob).then_some(j))
            .unwrap();
        (p, probs[p])
    }

    // project the `k`-th particle into its `p`-th state with a renormalization
    // factor, but don't change any singular values
    //
    // assumes `k` is in bounds
    fn project_state(&mut self, k: usize, p: usize, renorm: A::Re) {
        let (m, s, n) = self.data[k].dims();
        let zero = A::zero();
        if n == 1 {
            unsafe { self.data[k].make_rfused() }
            .column_iter_mut()
            .enumerate()
            .for_each(|(j, mut g_j0)| {
                if j == p {
                    g_j0.scale_mut(renorm.recip());
                } else {
                    g_j0.fill(zero);
                }
            });
        } else if m == 1 {
            unsafe { self.data[k].make_lfused() }
            .row_iter_mut()
            .enumerate()
            .for_each(|(j, mut g0j_)| {
                if j == p {
                    g0j_.scale_mut(renorm.recip());
                } else {
                    g0j_.fill(zero);
                }
            });
        } else {
            unsafe { self.data[k].make_lfused() }
            .column_iter_mut()
            .for_each(|g__v| {
                g__v.reshape_generic(na::Dyn(m), na::Dyn(s))
                .column_iter_mut()
                .enumerate()
                .for_each(|(j, mut g_jv)| {
                    if j == p {
                        g_jv.scale_mut(renorm.recip());
                    } else {
                        g_jv.fill(zero);
                    }
                });
            });
        }
    }

    /// Perform a projective measurement on the `k`-th particle, reporting the
    /// quantum number of the (randomized) outcome state for that particle.
    ///
    /// If `k` is out of bounds, do nothing and return `None`.
    pub fn measure<R>(&mut self, k: usize, rng: &mut R) -> Option<usize>
    where R: Rng + ?Sized
    {
        if k >= self.n { return None; }
        let (p, prob) = self.sample_state(k, rng);
        self.project_state(k, p, prob.sqrt());
        self.refactor_sweep();
        Some(p)
    }

    /// Perform a projective measurement on the `k`-th particle, post-selected
    /// to a particular outcome.
    ///
    /// If `k` is out of bounds or `p` is an invalid quantum number, do nothing.
    pub fn measure_postsel(&mut self, k: usize, p: usize) {
        if k >= self.n || p >= self.idxs[k].dim() { return; }
        let prob = self.local_prob(k, p);
        self.project_state(k, p, prob.sqrt());
        self.refactor_sweep();
    }
}

#[allow(clippy::mem_replace_with_uninit)]
unsafe fn replace_zeroed<T>(loc: &mut T) -> T {
    unsafe { mem::replace(loc, mem::zeroed()) }
}

#[allow(clippy::mut_from_ref)]
unsafe fn make_mut<T>(x: &T) -> &mut T {
    std::ptr::from_ref(x).cast_mut().as_mut().unwrap()
}

