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
//! use ndarray as nd;
//! use num_complex::Complex32 as C32;
//! use num_traits::{ Zero, One };
//! use rand::thread_rng;
//! use tensor_net::mps::*;
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
//! let h: nd::Array2<C32> // hadamard
//!     = nd::array![
//!         [C32::from(0.5).sqrt(),  C32::from(0.5).sqrt()],
//!         [C32::from(0.5).sqrt(), -C32::from(0.5).sqrt()],
//!     ];
//! let cx: nd::Array2<C32> // CNOT
//!     = nd::array![
//!         [C32::one(),  C32::zero(), C32::zero(), C32::zero()],
//!         [C32::zero(), C32::one(),  C32::zero(), C32::zero()],
//!         [C32::zero(), C32::zero(), C32::zero(), C32::one() ],
//!         [C32::zero(), C32::zero(), C32::one(),  C32::zero()],
//!     ];
//!
//! let n: usize = 4; // number of qubits
//! let indices: Vec<Q> = (0..n).map(Q).collect();
//!
//! // initialize to the first available quantum number for all particles,
//! // i.e. ∣00000⟩
//! let mut mps: MPS<Q, C32> = MPS::new(indices, None).unwrap();
//!
//! // apply some gates
//! mps.apply_unitary1(0, &h).unwrap();
//! mps.apply_unitary2(0, &cx).unwrap();
//!
//! // perform a randomized projective measurement on qubit zero
//! let mut rng = thread_rng();
//! let outcome = mps.measure(0, &mut rng).unwrap();
//! println!("measured qubit 0 as ∣{outcome}⟩");
//!
//! // contract the MPS into a single tensor, which can then be flattened into a
//! // normal 1D state vector
//! let mut tens = mps.into_tensor();
//! tens.sort_indices(); // make sure all the qubits are in the right order
//! let (_, state) = tens.into_flat();
//! println!("{state}");
//! // either [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//! // or     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
//! // based on the result of the measurement
//! ```

use std::{
    cmp::Ordering,
    fmt,
    iter::Sum,
    ops::{ Add, Range },
};
use itertools::Itertools;
use ndarray as nd;
use ndarray_linalg::{
    Eigh,
    SVDInto,
    UPLO,
    types::{ Scalar, Lapack },
};
use num_complex::{ ComplexFloat, Complex32 as C32 };
use num_traits::{ Float, One, Zero };
use once_cell::sync::Lazy;
use rand::{
    Rng,
    distributions::{ Distribution, Standard },
};
use thiserror::Error;
use crate::{
    ComplexFloatExt,
    circuit::Q,
    gate::{ self, Gate },
    network::Network,
    pool::ContractorPool,
    tensor::{ Idx, Tensor },
};

#[derive(Debug, Error)]
pub enum MPSError {
    /// Returned when attempting to create a new MPS for a state of less than 1
    /// particle.
    #[error("error in MPS creation: cannot create for an empty system")]
    EmptySystem,

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

/// A matrix product (pure) state.
///
/// This is a specific case of a [`Network<T, A>`][crate::network::Network],
/// where tensors are arranged in a 1D chain with closed boundaries, and have
/// one physical index each.
///
/// The MPS is maintained in a so-called "canonical" factorization based on the
/// Schmidt decomposition. The Schmidt values are readily available at any given
/// time, which enables efficient calculation of entanglement entropies, at the
/// cost of locally re-computing the decomposition for projective measurements
/// and multi-particle unitaries. Schmidt bases at each site are truncated based
/// on the relevant Schmidt values, however, so for most cases the runtime cost
/// of these decompositions should be relatively small.
///
/// Initial factorization of an arbitrary state (see [`Self::from_vector`])
/// happens via a series of singular value decompositions, which incurs a total
/// runtime cost that is *O*(*n* *D*<sup>*n*</sup>), where *n* is the number of
/// particles and *D* is the maximum dimension of their quantum numbers/indices.
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
///
/// > **Note**: If the MPS will ever be converted to a [`Tensor`] or
/// > [`Network`], or used to compute a density matrix (including for
/// > calculation of the Rényi entropy via [`Self::entropy_ry`] /
/// > [`Self::entropy_ry_par`]), then all physical indices need to be
/// > distinguishable. See [`Idx`] for more information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MPS<T, A>
where A: ComplexFloat
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
    // Singular values. When contracting with `data` tensors, use
    // `ComplexFloat + ComplexFloatExt::from_real`.
    pub(crate) svals: Vec<Vec<A::Real>>, // length n - 1
    // Threshold for singular values.
    pub(crate) eps: A::Real,
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + ComplexFloatExt,
    nd::Array2<A>: SVDInto<U = nd::Array2<A>, Sigma = nd::Array1<A::Real>, VT = nd::Array2<A>>,
{
    /// Initialize to a separable product state with each particle in the first
    /// of its available eigenstates.
    ///
    /// Optionally provide a global cutoff threshold for singular values, which
    /// defaults to machine epsilon.
    ///
    /// Fails if no particle indices are provided.
    #[inline]
    pub fn new<I>(indices: I, eps: Option<A::Real>) -> MPSResult<Self>
    where I: IntoIterator<Item = T>
    {
        let eps = Float::abs(eps.unwrap_or_else(A::Real::epsilon));
        let indices: Vec<T> = indices.into_iter().collect();
        if indices.is_empty() { return Err(EmptySystem); }
        let n = indices.len();
        let data: Vec<nd::Array3<A>>
            = indices.iter()
            .map(|idx| {
                let mut g: nd::Array3<A> = nd::Array::zeros((1, idx.dim(), 1));
                g[[0, 0, 0]] = A::one();
                g
            })
            .collect();
        let svals: Vec<Vec<A::Real>>
            = (0..n - 1)
            .map(|_| vec![A::Real::one()])
            .collect();
        Ok(Self { n, data, outs: indices, svals, eps })
    }

    /// Initialize by factoring an existing pure state vector.
    ///
    /// Optionally provide a global cutoff threshold for singular values, which
    /// defaults to machine epsilon.
    ///
    /// Fails if no particle indices are provided or the initial state vector
    /// does not have length Π<sub>*k*</sub> *D*<sub>*k*</sub>, where
    /// *D*<sub>*k*</sub> is the dimension of the *k*-th index.
    #[inline]
    pub fn from_vector<I, J>(indices: I, state: J, eps: Option<A::Real>)
        -> MPSResult<Self>
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator<Item = A>,
    {
        let eps = Float::abs(eps.unwrap_or_else(A::Real::epsilon));
        let indices: Vec<T> = indices.into_iter().collect();
        if indices.is_empty() { return Err(EmptySystem); }
        let statelen: usize = indices.iter().map(|idx| idx.dim()).product();
        let mut state: nd::Array1<A> = state.into_iter().collect();
        if state.len() != statelen { return Err(StateIncompatibleShape); }
        let norm
            = state.iter()
            .map(|a| *a * a.conj())
            .fold(A::zero(), |acc, a| acc + a)
            .sqrt();
        state.iter_mut().for_each(|a| { *a = *a / norm; });
        let n = indices.len();
        if n == 1 {
            let mut g: nd::Array3<A>
                = nd::Array::zeros((1, indices[0].dim(), 1));
            state.move_into(g.slice_mut(nd::s![0, .., 0]));
            let data: Vec<nd::Array3<A>> = vec![g];
            let svals: Vec<Vec<A::Real>> = Vec::with_capacity(0);
            Ok(Self { n, data, outs: indices, svals, eps })
        } else {
            Ok(Self::do_from_vector(indices, state, eps))
        }
    }

    fn do_from_vector(outs: Vec<T>, state: nd::Array1<A>, eps: A::Real) -> Self
    {
        let n = outs.len(); // assume n ≥ 2
        let mut data: Vec<nd::Array3<A>> = Vec::with_capacity(n);
        let mut svals: Vec<Vec<A::Real>> = Vec::with_capacity(n - 1);
        let mut udim: usize = 1;
        let mut outdim: usize;
        let statelen = state.len();
        let mut q: nd::Array2<A> = state.into_shape((1, statelen)).unwrap();
        let mut reshape_m: usize;
        let mut reshape_n: usize;
        let mut rank: usize;
        let mut rankslice: nd::Slice;
        for (k, outk) in outs.iter().take(n - 1).enumerate() {
            outdim = outk.dim();

            // initial reshape to fuse outk with the previous schmidt index
            reshape_m = udim * outdim;
            reshape_n = q.len() / reshape_m;
            q = q.into_shape((reshape_m, reshape_n)).unwrap();

            // SVD/Schmidt decomp: left vectors are columns in u, Schmidt values
            // are in s, and right vectors are rows in q
            // this is done in place to avoid constructing a full-rank matrix
            // for the right vectors since we want to truncate based on the
            // Schmidt rank
            let (Some(u), s, Some(vh)) = q.svd_into(true, true).unwrap()
                else { unreachable!() };
            rank = s.iter().take_while(|sk| sk > &&eps).count();
            rankslice = nd::Slice::new(0, Some(rank as isize), 1);
            q = vh;
            q.slice_axis_inplace(nd::Axis(0), rankslice);

            // construct the Γ tensor from u; Γ_k = (Σ_k-1)^-1 . u
            // can't slice u in place on the column index because it'd mean the
            // remaining data won't be contiguous in memory; this would make the
            // reshape fail
            let mut g
                = u.slice(nd::s![.., ..rank]).to_owned()
                .into_shape((udim, outdim, rank))
                .unwrap();
            if let Some(slast) = svals.last() {
                g.axis_iter_mut(nd::Axis(0))
                    .zip(slast)
                    .for_each(|(mut gv, sv)| {
                        gv.map_inplace(|gvk| {
                            *gvk = *gvk / A::from_real(*sv);
                        });
                    });
            }

            // done for the particle
            data.push(g);
            // if there are more SVDs remaining, prepare by recombining with the
            // singular values; otherwise the final state of q is exactly Γ_n-1
            if k < n - 2 { // if there are SVDs remaining, prepare
                q.axis_iter_mut(nd::Axis(0))
                    .zip(&s)
                    .for_each(|(mut qv, sv)| {
                        qv.map_inplace(|qvk| {
                            *qvk = *qvk * A::from_real(*sv);
                        });
                    });
            }
            svals.push(s.into_iter().take(rank).collect());
            udim = rank;
        }
        // final Γ tensor
        data.push(q.into_shape((udim, outs[n - 1].dim(), 1)).unwrap());

        Self { n, data, outs, svals, eps }
    }

    /// Initialize by factoring an existing tensor representing a pure state.
    ///
    /// Note the additional `Ord` bound on the index type, which is used to
    /// ensure determinism in the MPS factorization.
    ///
    /// Optionally provide a global cutoff threshold for singular values, which
    /// defaults to machine epsilon.
    ///
    /// Fails if no particle indices are provided.
    #[inline]
    pub fn from_tensor(mut state: Tensor<T, A>, eps: Option<A::Real>)
        -> MPSResult<Self>
    where T: Ord
    {
        let eps = Float::abs(eps.unwrap_or_else(A::Real::epsilon));
        state.sort_indices();
        let (indices, mut state) = state.into_flat();
        if indices.is_empty() { return Err(EmptySystem); }
        let norm
            = state.iter()
            .map(|a| *a * a.conj())
            .fold(A::zero(), |acc, a| acc + a)
            .sqrt();
        state.iter_mut().for_each(|a| { *a = *a / norm; });
        let n = indices.len();
        if n == 1 {
            let mut g: nd::Array3<A>
                = nd::Array::zeros((1, indices[0].dim(), 1));
            state.move_into(g.slice_mut(nd::s![0, .., 0]));
            let data: Vec<nd::Array3<A>> = vec![g];
            let svals: Vec<Vec<A::Real>> = Vec::with_capacity(0);
            Ok(Self { n, data, outs: indices, svals, eps })
        } else {
            Ok(Self::do_from_vector(indices, state, eps))
        }
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + 'static,
{
    /// Apply a unitary transformation to the `k`-th particle.
    ///
    /// Does nothing if `k` is out of bounds. The arrangement of the elements of
    /// `op` should correspond to the usual left-matrix-multiplication view of
    /// operator application. `op` is not checked for unitarity.
    ///
    /// Fails if `op` is not square with a dimension equal to that of the `k`-th
    /// physical index.
    #[inline]
    pub fn apply_unitary1(&mut self, k: usize, op: &nd::Array2<A>)
        -> MPSResult<&mut Self>
    {
        if k >= self.n { return Ok(self); }
        let dk = self.outs[k].dim();
        if op.shape() != [dk, dk] { return Err(OperatorIncompatibleShape); }
        self.data[k]
            .axis_iter_mut(nd::Axis(0))
                .for_each(|mut gkv| {
                    gkv.axis_iter_mut(nd::Axis(1))
                        .for_each(|mut gkv_u| {
                            gkv_u.assign(&op.dot(&gkv_u));
                        });
                });
        Ok(self)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + ComplexFloatExt + 'static,
{
    /// Evaluate the expectation value of a local operator acting on (only) the
    /// `k`-th particle.
    ///
    /// Returns zero if `k` is out of bounds. The arrangement of the elements of
    /// `op` should correspond to the usual left-matrix-multiplication view of
    /// operator application.
    ///
    /// Fails if `op` is not square with a dimension equal to that of the `k`-th
    /// physical index.
    #[inline]
    pub fn expectation_value(&self, k: usize, op: &nd::Array2<A>)
        -> MPSResult<A>
    {
        if k >= self.n { return Ok(A::zero()); }
        let dk = self.outs[k].dim();
        if op.shape() != [dk, dk] { return Err(OperatorIncompatibleShape); }
        let gk = &self.data[k];
        if k == 0 {
            let lk = &self.svals[k];
            let ev: A
                = (0..dk)
                .cartesian_product(0..dk)
                .cartesian_product(lk.iter().enumerate())
                .map(|((sk, ssk), (w, lkw))| {
                    gk[[0, ssk, w]].conj()
                        * op[[ssk, sk]]
                        * gk[[0, sk, w]]
                        * A::from_real(Float::powi(*lkw, 2))
                })
                .fold(A::zero(), A::add);
            Ok(ev)
        } else if k == self.n - 1 {
            let lkm1 = &self.svals[k - 1];
            let ev: A
                = lkm1.iter().enumerate()
                .cartesian_product(0..dk)
                .cartesian_product(0..dk)
                .map(|(((v, lkm1v), sk), ssk)| {
                    A::from_real(Float::powi(*lkm1v, 2))
                        * gk[[v, ssk, 0]].conj()
                        * op[[ssk, sk]]
                        * gk[[v, sk, 0]]
                })
                .fold(A::zero(), A::add);
            Ok(ev)
        } else {
            let lkm1 = &self.svals[k - 1];
            let lk = &self.svals[k];
            let ev: A
                = lkm1.iter().enumerate()
                .cartesian_product(0..dk)
                .cartesian_product(0..dk)
                .cartesian_product(lk.iter().enumerate())
                .map(|((((v, lkm1v), sk), ssk), (w, lkw))| {
                    A::from_real(Float::powi(*lkm1v, 2))
                        * gk[[v, ssk, w]].conj()
                        * op[[ssk, sk]]
                        * gk[[v, sk, w]]
                        * A::from_real(Float::powi(*lkw, 2))
                })
                .fold(A::zero(), A::add);
            Ok(ev)
        }
    }
}

struct Svd<A: ComplexFloat> {
    u: nd::Array2<A>,
    s: Vec<A::Real>,
    q: nd::Array2<A>,
    rank: usize,
}

#[inline]
fn do_svd_local<A>(q: nd::Array2<A>, eps: A::Real) -> Svd<A>
where
    A: ComplexFloat + ComplexFloatExt,
    nd::Array2<A>: SVDInto<U = nd::Array2<A>, Sigma = nd::Array1<A::Real>, VT = nd::Array2<A>>,
{
    let (Some(u), mut s, Some(mut q)) = q.svd_into(true, true).unwrap()
        else { unreachable!() };
    let mut norm: A::Real = Float::sqrt(
        s.iter()
            .map(|sj| Float::powi(*sj, 2))
            .fold(A::Real::zero(), A::Real::add)
    );
    s.iter_mut().for_each(|sj| { *sj = *sj / norm; });
    let rank = s.iter().take_while(|sj| **sj > eps).count();
    norm = Float::sqrt(
        s.iter()
            .map(|sj| Float::powi(*sj, 2))
            .fold(A::Real::zero(), A::Real::add)
    );
    let rankslice = nd::Slice::new(0, Some(rank as isize), 1);
    q.slice_axis_inplace(nd::Axis(0), rankslice);
    q.iter_mut().for_each(|qv| { *qv = *qv * A::from_real(norm); });
    // q.axis_iter_mut(nd::Axis(0))
    //     .zip(&s)
    //     .for_each(|(mut qv, sv)| {
    //         qv.map_inplace(|qvj| { *qvj = *qvj * A::from_real(*sv); });
    //     });
    let u = u.slice(nd::s![.., ..rank]).to_owned();
    let s = s.into_iter().take(rank).collect();
    Svd { u, s, q, rank }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + ComplexFloatExt,
    nd::Array2<A>: SVDInto<U = nd::Array2<A>, Sigma = nd::Array1<A::Real>, VT = nd::Array2<A>>,
{
    /// Apply a unitary operator to the `k`-th and `k + 1`-th particles.
    ///
    /// Does nothing if either `k` or `k + 1` are out of bounds. The arrangement
    /// of the elements of `op` should correspond to the usual
    /// left-matrix-multiplication view of operator application. `op` is not
    /// checked for unitarity.
    ///
    /// Fails if `u` is not square with a size equal to the product of the
    /// dimensions of the `k`-th and `k + 1`-th physical indices.
    #[inline]
    pub fn apply_unitary2(&mut self, k: usize, op: &nd::Array2<A>)
        -> MPSResult<&mut Self>
    {
        if self.n == 1 || k >= self.n - 1 { return Ok(self); }
        let dk = self.outs[k].dim();
        let dkp1 = self.outs[k + 1].dim();
        if op.shape() != [dk * dkp1, dk * dkp1] {
            return Err(OperatorIncompatibleShape);
        }
        let gk = &self.data[k];
        let shk = gk.shape().to_vec();
        let lk = &self.svals[k];
        let gkp1 = &self.data[k + 1];
        let shkp1 = gkp1.shape().to_vec();
        let z1 = dk;
        let z2 = shkp1[2];
        let q: nd::Array2<A>
            = nd::Array2::from_shape_fn(
                (shk[0] * dk, dkp1 * shkp1[2]),
                |(v_sk, skp1_w)| {
                    let v = v_sk / z1;
                    let sk = v_sk % z1;
                    let skp1 = skp1_w / z2;
                    let w = skp1_w % z2;
                    (0..dk)
                        .cartesian_product(0..dkp1)
                        .cartesian_product(lk.iter().enumerate())
                        .map(|((ssk, sskp1), (u, lku))| {
                            gk[[v, ssk, u]]
                                * A::from_real(*lku)
                                * gkp1[[u, sskp1, w]]
                                * op[[sk * dkp1 + skp1, ssk * dkp1 + sskp1]]
                        })
                        .fold(A::zero(), A::add)
                },
            );
        let Svd { u, s, q, rank } = do_svd_local(q, self.eps);
        let gk_new = u.into_shape((shk[0], dk, rank)).unwrap();
        let gkp1_new = q.into_shape((rank, dkp1, shkp1[2])).unwrap();
        self.data[k] = gk_new;
        self.svals[k] = s;
        self.data[k + 1] = gkp1_new;
        Ok(self)
    }

    /// Like [`Self::apply_unitary2`], but for a transformation described by a
    /// tensor with unfused indices (i.e. rank 4).
    ///
    /// `op` should have shape `[a, b, a, b]`, where `a` and `b` are the
    /// dimensions of the `k`-th and `k + 1`-th physical indices, respectively.
    /// The last two are summed over.
    #[inline]
    pub fn apply_unitary2x2(&mut self, k: usize, op: &nd::Array4<A>)
        -> MPSResult<&mut Self>
    {
        if self.n == 1 || k >= self.n - 1 { return Ok(self); }
        let dk = self.outs[k].dim();
        let dkp1 = self.outs[k + 1].dim();
        if op.shape() != [dk, dkp1, dk, dkp1] {
            return Err(OperatorIncompatibleShape);
        }
        let gk = &self.data[k];
        let shk = gk.shape().to_vec();
        let lk = &self.svals[k];
        let gkp1 = &self.data[k + 1];
        let shkp1 = gkp1.shape().to_vec();
        let z1 = dk;
        let z2 = shkp1[2];
        let q: nd::Array2<A>
            = nd::Array2::from_shape_fn(
                (shk[0] * dk, dkp1 * shkp1[2]),
                |(v_sk, skp1_w)| {
                    let v = v_sk / z1;
                    let sk = v_sk % z1;
                    let skp1 = skp1_w / z2;
                    let w = skp1_w % z2;
                    (0..dk)
                        .cartesian_product(0..dkp1)
                        .cartesian_product(lk.iter().enumerate())
                        .map(|((ssk, sskp1), (u, lku))| {
                            gk[[v, ssk, u]]
                                * A::from_real(*lku)
                                * gkp1[[u, sskp1, w]]
                                * op[[sk, skp1, ssk, sskp1]]
                        })
                        .fold(A::zero(), A::add)
                },
            );
        let Svd { u, s, q, rank } = do_svd_local(q, self.eps);
        let gk_new = u.into_shape((shk[0], dk, rank)).unwrap();
        let gkp1_new = q.into_shape((rank, dkp1, shkp1[2])).unwrap();
        self.data[k] = gk_new;
        self.svals[k] = s;
        self.data[k + 1] = gkp1_new;
        Ok(self)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + ComplexFloatExt,
    nd::Array2<A>: SVDInto<U = nd::Array2<A>, Sigma = nd::Array1<A::Real>, VT = nd::Array2<A>>,
    Standard: Distribution<A::Real>,
{
    /// Perform a randomized projective measurement on the `k`-th particle,
    /// reporting the index value of the outcome state for that particle.
    ///
    /// If `k` is out of bounds, do nothing and return `None`.
    #[inline]
    pub fn measure<R>(&mut self, k: usize, rng: &mut R) -> Option<usize>
    where R: Rng + ?Sized
    {
        // broad algorithm:
        // 1. calculate measurement probabilities
        // 2. sample the quantum number of the measurement outcome
        // 3. perform a local contraction with k's nearest neighbors in order to
        // 4. redo SVDs locally to maintain proper normalization

        if k >= self.n { return None; }
        let r: A::Real = rng.gen();
        if k == 0 {
            // calculate measurement probabilities
            let gk = &self.data[k];
            let dk = self.outs[k].dim();
            let lk = &self.svals[k];
            let probs: Vec<A::Real>
                = (0..dk)
                .map(|p| {
                    gk.index_axis(nd::Axis(1), p).iter()
                        .zip(lk)
                        .map(|(gkv, lkv)| {
                            (*gkv * gkv.conj()).re()
                                * Float::powi(*lkv, 2)
                        })
                        .fold(A::Real::zero(), A::Real::add)
                })
                .collect();

            // sample the measurement outcome
            let p: usize
                = probs.iter().copied()
                .scan(A::Real::zero(), |cu, pr| { *cu = *cu + pr; Some(*cu) })
                .enumerate()
                .find_map(|(k, cuprob)| (r < cuprob).then_some(k))
                .unwrap();
            let renorm = A::from_real(Float::sqrt(probs[p]));

            // local contraction into an SVD-able matrix
            let gkp1 = &self.data[k + 1];
            let shkp1 = gkp1.shape().to_vec();
            let z = shkp1[2];
            let q: nd::Array2<A>
                = nd::Array2::from_shape_fn(
                    (dk, shkp1[1] * shkp1[2]),
                    |(sk, skp1_w)| {
                        let skp1 = skp1_w / z;
                        let w = skp1_w % z;
                        if sk == p {
                            (0..shkp1[0])
                                .map(|u| {
                                    gk[[0, sk, u]]
                                        * A::from_real(lk[u])
                                        * gkp1[[u, skp1, w]]
                                        / renorm
                                })
                                .fold(A::zero(), A::add)
                        } else {
                            A::zero()
                        }
                    },
                );

            // redo SVDs
            let Svd { u, s, q, rank } = do_svd_local(q, self.eps);
            let gk_new = u.into_shape((1, dk, rank)).unwrap();
            let gkp1_new = q.into_shape((rank, shkp1[1], shkp1[2])).unwrap();
            self.data[k] = gk_new;
            self.svals[k] = s;
            self.data[k + 1] = gkp1_new;
            // done
            Some(p)
        } else if k == self.n - 1 {
            // calculate measurement probabilities
            let lkm1 = &self.svals[k - 1];
            let gk = &self.data[k];
            let dk = self.outs[k].dim();
            let probs: Vec<A::Real>
                = (0..dk)
                .map(|p| {
                    gk.index_axis(nd::Axis(1), p).iter()
                        .zip(lkm1)
                        .map(|(gkv, lkm1v)| {
                            Float::powi(*lkm1v, 2)
                                * (*gkv * gkv.conj()).re()
                        })
                        .fold(A::Real::zero(), A::Real::add)
                })
                .collect();

            // sample the measurement outcome
            let p: usize
                = probs.iter().copied()
                .scan(A::Real::zero(), |cu, pr| { *cu = *cu + pr; Some(*cu) })
                .enumerate()
                .find_map(|(k, cuprob)| (r < cuprob).then_some(k))
                .unwrap();
            let renorm = A::from_real(Float::sqrt(probs[p]));

            // local contraction into an SVD-able matrix
            let gkm1 = &self.data[k - 1];
            let shkm1 = gkm1.shape().to_vec();
            let z = shkm1[1];
            let q: nd::Array2<A>
                = nd::Array2::from_shape_fn(
                    (shkm1[0] * shkm1[1], dk),
                    |(v_skm1, sk)| {
                        let v = v_skm1 / z;
                        let skm1 = v_skm1 % z;
                        if sk == p {
                            (0..shkm1[2])
                                .map(|u| {
                                    gkm1[[v, skm1, u]]
                                        * A::from_real(lkm1[u])
                                        * gk[[u, sk, 0]]
                                        / renorm
                                })
                                .fold(A::zero(), A::add)
                        } else {
                            A::zero()
                        }
                    },
                );

            // redo SVDs
            let Svd { u, s, q, rank } = do_svd_local(q, self.eps);
            let gkm1_new = u.into_shape((shkm1[0], shkm1[1], rank)).unwrap();
            let gk_new = q.into_shape((rank, dk, 1)).unwrap();
            self.data[k - 1] = gkm1_new;
            self.svals[k - 1] = s;
            self.data[k] = gk_new;
            // done
            Some(p)
        } else {
            // calculate measurement probabilities
            let lkm1 = &self.svals[k - 1];
            let gk = &self.data[k];
            let dk = self.outs[k].dim();
            let lk = &self.svals[k];
            let probs: Vec<A::Real>
                = (0..dk)
                .map(|p| {
                    gk.index_axis(nd::Axis(1), p).iter()
                        .zip(lkm1.iter().cartesian_product(lk))
                        .map(|(gkvu, (lkm1v, lku))| {
                            Float::powi(*lkm1v, 2)
                                * (*gkvu * gkvu.conj()).re()
                                * Float::powi(*lku, 2)
                        })
                        .fold(A::Real::zero(), A::Real::add)
                })
                .collect();

            // sample the measurement outcome
            let p: usize
                = probs.iter().copied()
                .scan(A::Real::zero(), |cu, pr| { *cu = *cu + pr; Some(*cu) })
                .enumerate()
                .find_map(|(k, cuprob)| (r < cuprob).then_some(k))
                .unwrap();
            let renorm = A::from_real(Float::sqrt(probs[p]));

            // local contraction into an SVD-able matrix
            let gkm1 = &self.data[k - 1];
            let gkp1 = &self.data[k + 1];
            let shkm1 = gkm1.shape().to_vec();
            let shkp1 = gkp1.shape().to_vec();
            let z1 = shkm1[1];
            let z2 = shkp1[1] * shkp1[2];
            let z3 = shkp1[2];
            let q: nd::Array2<A>
                = nd::Array2::from_shape_fn(
                    (shkm1[0] * shkm1[1], dk * shkp1[1] * shkp1[2]),
                    |(v_skm1, sk_skp1_w)| {
                        let v = v_skm1 / z1;
                        let skm1 = v_skm1 % z1;
                        let sk = sk_skp1_w / z2;
                        let skp1 = (sk_skp1_w % z2) / z3;
                        let w = sk_skp1_w % z3;
                        if sk == p {
                            (0..shkm1[2]).cartesian_product(0..shkp1[0])
                                .map(|(u1, u2)| {
                                    gkm1[[v, skm1, u1]]
                                        * A::from_real(lkm1[u1])
                                        * gk[[u1, sk, u2]]
                                        * A::from_real(lk[u2])
                                        * gkp1[[u2, skp1, w]]
                                        / renorm
                                })
                                .fold(A::zero(), A::add)
                        } else {
                            A::zero()
                        }
                    },
                );

            // redo SVDs
            let Svd { u, s, mut q, rank } = do_svd_local(q, self.eps); // k - 1 and k
            let gkm1_new = u.into_shape((shkm1[0], shkm1[1], rank)).unwrap();
            self.data[k - 1] = gkm1_new;
            self.svals[k - 1] = s;
            //
            q = q.into_shape((rank * dk, shkp1[1] * shkp1[2])).unwrap();
            let udim = rank;
            let Svd { u, s, q, rank } = do_svd_local(q, self.eps); // k and k + 1
            let gk_new = u.into_shape((udim, dk, rank)).unwrap();
            let gkp1_new = q.into_shape((rank, shkp1[1], shkp1[2])).unwrap();
            self.data[k] = gk_new;
            self.svals[k] = s;
            self.data[k + 1] = gkp1_new;
            // done
            Some(p)
        }
    }
}

impl<T, A> MPS<T, A>
where
    A: ComplexFloat,
    A::Real: Float,
{
    /// Return the number of particles.
    #[inline]
    pub fn n(&self) -> usize { self.n }

    /// Compute the Von Neumann entropy for a bipartition placed on the `b`-th
    /// bond.
    ///
    /// Returns `None` if `b` is out of bounds.
    #[inline]
    pub fn entropy_vn(&self, b: usize) -> Option<A::Real> {
        let zero = A::Real::zero();
        self.svals.get(b)
            .map(|s| {
                s.iter().copied()
                    .filter(|sk| sk > &zero)
                    .map(|sk| {
                        let sk2 = sk * sk;
                        -sk2 * Float::ln(sk2)
                    })
                    .fold(zero, |acc, term| acc + term)
            })
    }

    /// Compute the `a`-th Rényi entropy for a bipartition placed on the `b`-th
    /// bond in the Schmidt basis.
    ///
    /// Returns the Von Neumann entropy for `a == 1` and `None` if `b` is out of
    /// bounds.
    #[inline]
    pub fn entropy_ry_schmidt(&self, a: A::Real, b: usize) -> Option<A::Real> {
        let one = A::Real::one();
        if a == one {
            self.entropy_vn(b)
        } else {
            self.svals.get(b)
                .map(|s| {
                    Float::ln(
                        s.iter().copied()
                            .map(|sk| Float::powf(sk * sk, a))
                            .fold(A::Real::zero(), |acc, sk| acc + sk)
                    ) * Float::recip(one - a)
                })
        }
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + ComplexFloatExt,
{
    /// Convert to an unevaluated [`Network`] representing the pure state.
    #[inline]
    pub fn into_network(self) -> Network<MPSIndex<T>, A> {
        let Self { data, outs, svals, n, .. } = self;
        let mut network: Network<MPSIndex<T>, A> = Network::new();

        // gamma matrices
        for (k, (datak, idxk)) in data.into_iter().zip(outs).enumerate() {
            if k == 0 {
                let idx = MPSIndex::Physical(idxk);
                let u = MPSIndex::BondL { label: k, dim: svals[k].len() };
                let data = datak.into_shape((idx.dim(), u.dim())).unwrap();
                let data = unsafe {
                    Tensor::from_array_unchecked([idx, u], data)
                };
                network.push(data).unwrap();
            } else if k == n - 1 {
                let idx = MPSIndex::Physical(idxk);
                let v = MPSIndex::BondR { label: k - 1, dim: svals[k - 1].len() };
                let data = datak.into_shape((v.dim(), idx.dim())).unwrap();
                let data = unsafe {
                    Tensor::from_array_unchecked([v, idx], data)
                };
                network.push(data).unwrap();
            } else {
                let idx = MPSIndex::Physical(idxk);
                let v = MPSIndex::BondR { label: k - 1, dim: svals[k - 1].len() };
                let u = MPSIndex::BondL { label: k, dim: svals[k].len() };
                let data = datak.into_shape((v.dim(), idx.dim(), u.dim()))
                    .unwrap();
                let data = unsafe {
                    Tensor::from_array_unchecked([v, idx, u], data)
                };
                network.push(data).unwrap();
            }
        }

        // singular value matrices
        for (k, sk) in svals.into_iter().enumerate() {
            let d = sk.len();
            let mut data = nd::Array::zeros((d, d));
            data.diag_mut().iter_mut()
                .zip(sk)
                .for_each(|(dj, skj)| { *dj = A::from_real(skj); });
            let data = unsafe {
                Tensor::from_array_unchecked(
                    [
                        MPSIndex::BondL { label: k, dim: d },
                        MPSIndex::BondR { label: k, dim: d },
                    ],
                    data,
                )
            };
            network.push(data).unwrap();
        }
        network
    }

    /// Convert to an unevaluated [`Network`] representing the pure state as a
    /// density matrix.
    #[inline]
    pub fn into_network_density(self) -> Network<MPSMatIndex<T>, A> {
        unsafe {
            let mut net
                = self.into_network()
                .map_indices(MPSIndex::into_mpsmat)
                .unwrap();
            let conj_nodes: Vec<_>
                = net.nodes()
                .map(|(_, tens)| tens.conj().map_indices(MPSMatIndex::conj))
                .collect();
            conj_nodes.into_iter()
                .for_each(|tens| { net.push(tens).unwrap(); });
            net
        }
    }

    /// Convert to an unevaluated [`Network`] representing a contiguous subset
    /// of particles, with the remainder traced over.
    pub fn into_network_part(self, part: Range<usize>)
        -> Network<MPSMatIndex<T>, A>
    {
        let Self { data, outs, svals, n, .. } = self;
        let mut network: Network<MPSMatIndex<T>, A> = Network::new();
        for (k, (datak, idxk)) in data.into_iter().zip(outs).enumerate() {
            if !part.contains(&k) { continue; }
            if k == 0 {
                let idx = MPSMatIndex::PhysicalKet(idxk);
                let u = MPSMatIndex::BondLKet { label: k, dim: svals[k].len() };
                let data = datak.into_shape((idx.dim(), u.dim())).unwrap();
                let data = unsafe {
                    Tensor::from_array_unchecked([idx, u], data)
                };
                network.push(data).unwrap();
            } else if k == n - 1 {
                let idx = MPSMatIndex::PhysicalKet(idxk);
                let v = MPSMatIndex::BondRKet { label: k - 1, dim: svals[k - 1].len() };
                let data = datak.into_shape((v.dim(), idx.dim())).unwrap();
                let data = unsafe {
                    Tensor::from_array_unchecked([v, idx], data)
                };
                network.push(data).unwrap();
            } else {
                let idx = MPSMatIndex::PhysicalKet(idxk);
                let v = MPSMatIndex::BondRKet { label: k - 1, dim: svals[k - 1].len() };
                let u = MPSMatIndex::BondLKet { label: k, dim: svals[k].len() };
                let data = datak.into_shape((v.dim(), idx.dim(), u.dim()))
                    .unwrap();
                let data = unsafe {
                    Tensor::from_array_unchecked([v, idx, u], data)
                };
                network.push(data).unwrap();
            }
        }
        for (k, sk) in svals.iter().enumerate() {
            if part.contains(&k) || part.contains(&(k + 1)) {
                let d = sk.len();
                let mut data = nd::Array::zeros((d, d));
                data.diag_mut().iter_mut()
                    .zip(sk)
                    .for_each(|(dj, skj)| { *dj = A::from_real(*skj); });
                let u = MPSMatIndex::BondLKet { label: k, dim: d };
                let v = MPSMatIndex::BondRKet { label: k, dim: d };
                let data = unsafe {
                    Tensor::from_array_unchecked([u, v], data)
                };
                network.push(data).unwrap();
            }
        }
        if part.start != 0 {
            let k = part.start - 1;
            let d = svals[k].len();
            let uket = MPSMatIndex::BondLKet { label: k, dim: d };
            let ubra = MPSMatIndex::BondLBra { label: k, dim: d };
            let tracer = unsafe {
                Tensor::from_array_unchecked(
                    [uket, ubra],
                    nd::array![[A::one(), A::zero()], [A::zero(), A::one()]],
                )
            };
            network.push(tracer).unwrap();
        }
        if part.end != n {
            let k = part.end - 1;
            let d = svals[k].len();
            let vket = MPSMatIndex::BondRKet { label: k, dim: d };
            let vbra = MPSMatIndex::BondRBra { label: k, dim: d };
            let tracer = unsafe {
                Tensor::from_array_unchecked(
                    [vket, vbra],
                    nd::array![[A::one(), A::zero()], [A::zero(), A::one()]],
                )
            };
            network.push(tracer).unwrap();
        }
        unsafe {
            let conj_nodes: Vec<_>
                = network.nodes()
                .map(|(_, tens)| tens.conj().map_indices(MPSMatIndex::conj))
                .collect();
            conj_nodes.into_iter()
                .for_each(|tens| { network.push(tens).unwrap(); });
            network
        }
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + ComplexFloatExt + Sum,
{
    /// Contract the MPS and convert to a single [`Tensor`] representing the
    /// pure state.
    #[inline]
    pub fn into_tensor(self) -> Tensor<T, A> {
        let tens
            = self.into_network()
            .contract()
            .unwrap();
        unsafe { tens.map_indices(MPSIndex::into_physical) }
    }

    /// Contract the MPS and convert to a single [`Tensor`] representing the
    /// pure state as a density matrix.
    #[inline]
    pub fn into_tensor_density(self) -> Tensor<MatIndex<T>, A> {
        let dens
            = self.into_network_density()
            .contract()
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Contract the MPS and convert to a single [`Tensor`] representing the
    /// partial trace of the state.
    ///
    /// Particles outside of `part` will be traced over.
    #[inline]
    pub fn into_tensor_part(self, part: Range<usize>) -> Tensor<MatIndex<T>, A>
    {
        let dens
            = self.into_network_part(part)
            .contract()
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Contract the MPS into a density matrix and compute the `a`-th Rényi
    /// entropy of a subspace.
    #[inline]
    pub fn entropy_ry(&self, a: u32, part: Range<usize>)
        -> <A as ComplexFloat>::Real
    where
        T: Ord,
        A: Scalar + Lapack,
        nd::Array2<A>:
        Eigh<EigVal = nd::Array1<<A as ComplexFloat>::Real>, EigVec = nd::Array2<A>>
        + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
    {
        let mut rho_tens = self.clone().into_tensor_part(part);
        rho_tens.sort_indices();
        let (indices, data) = rho_tens.into_flat();
        let sidelen: usize
            = indices.iter()
            .map(|idx| idx.dim())
            .product();
        let rho: nd::Array2<A> = data.into_shape((sidelen, sidelen)).unwrap();
        entropy_ry(&rho, a)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx + Send + 'static,
    A: ComplexFloat + ComplexFloatExt + Sum + Send + 'static,
{
    /// Like [`Self::into_tensor`], but using a [`ContractorPool`] for parallel
    /// contractions.
    #[inline]
    pub fn into_tensor_par(self, pool: &ContractorPool<MPSIndex<T>, A>)
        -> Tensor<T, A>
    {
        let tens
            = self.into_network()
            .contract_par(pool)
            .unwrap();
        unsafe { tens.map_indices(MPSIndex::into_physical) }
    }

    /// Like [`Self::into_tensor_density`], but using a [`ContractorPool`] for
    /// parallel contractions.
    #[inline]
    pub fn into_tensor_density_par(
        self,
        pool: &ContractorPool<MPSMatIndex<T>, A>,
    ) -> Tensor<MatIndex<T>, A>
    {
        let dens
            = self.into_network_density()
            .contract_par(pool)
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Like [`Self::into_tensor_part`], but using a [`ContractorPool`] for
    /// parallel contractions.
    #[inline]
    pub fn into_tensor_part_par(
        self,
        part: Range<usize>,
        pool: &ContractorPool<MPSMatIndex<T>, A>,
    ) -> Tensor<MatIndex<T>, A>
    {
        let dens
            = self.into_network_part(part)
            .contract_par(pool)
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Like [`Self::entropy_ry`], but using a [`ContractorPool`] for parallel
    /// contractions.
    #[inline]
    pub fn entropy_ry_par(
        &self,
        a: u32,
        part: Range<usize>,
        pool: &ContractorPool<MPSMatIndex<T>, A>,
    ) -> <A as ComplexFloat>::Real
    where
        T: Ord,
        A: Scalar + Lapack,
        nd::Array2<A>:
        Eigh<EigVal = nd::Array1<<A as ComplexFloat>::Real>, EigVec = nd::Array2<A>>
        + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
    {
        let mut rho_tens = self.clone().into_tensor_part_par(part, pool);
        rho_tens.sort_indices();
        let (indices, data) = rho_tens.into_flat();
        let sidelen: usize
            = indices.iter()
            .map(|idx| idx.dim())
            .product();
        let rho: nd::Array2<A> = data.into_shape((sidelen, sidelen)).unwrap();
        entropy_ry(&rho, a)
    }
}

impl MPS<Q, C32> {
    /// Perform the action of a gate.
    ///
    /// Does nothing if any qubit indices are out of bounds.
    #[inline]
    pub fn apply_gate(&mut self, gate: Gate) -> &mut Self {
        match gate {
            Gate::U(k, alpha, beta, gamma) if k < self.n => {
                self.apply_unitary1(k, &gate::make_u(alpha, beta, gamma))
                    .unwrap();
            },
            Gate::H(k) if k < self.n => {
                self.apply_unitary1(k, Lazy::force(&gate::HMAT))
                    .unwrap();
            },
            Gate::X(k) if k < self.n => {
                self.apply_unitary1(k, Lazy::force(&gate::XMAT))
                    .unwrap();
            },
            Gate::Z(k) if k < self.n => {
                self.apply_unitary1(k, Lazy::force(&gate::ZMAT))
                    .unwrap();
            },
            Gate::S(k) if k < self.n => {
                self.apply_unitary1(k, Lazy::force(&gate::SMAT))
                    .unwrap();
            },
            Gate::SInv(k) if k < self.n => {
                self.apply_unitary1(k, Lazy::force(&gate::SINVMAT))
                    .unwrap();
            },
            Gate::XRot(k, alpha) if k < self.n => {
                self.apply_unitary1(k, &gate::make_xrot(alpha))
                    .unwrap();
            },
            Gate::ZRot(k, alpha) if k < self.n => {
                self.apply_unitary1(k, &gate::make_zrot(alpha))
                    .unwrap();
            },
            Gate::CX(k) if k < self.n - 1 => {
                self.apply_unitary2(k, Lazy::force(&gate::CXMAT))
                    .unwrap();
            },
            Gate::CXRev(k) if k < self.n - 1 => {
                self.apply_unitary2(k, Lazy::force(&gate::CXREVMAT))
                    .unwrap();
            },
            Gate::CZ(k) if k < self.n - 1 => {
                self.apply_unitary2(k, Lazy::force(&gate::CZMAT))
                    .unwrap();
            },
            _ => { },
        }
        self
    }

    /// Perform a series of gates.
    #[inline]
    pub fn apply_circuit<'a, I>(&mut self, gates: I) -> &mut Self
    where I: IntoIterator<Item = &'a Gate>
    {
        gates.into_iter().copied().for_each(|g| { self.apply_gate(g); });
        self
    }
}

impl<T, A> fmt::Display for MPS<T, A>
where
    T: Idx,
    A: ComplexFloat + fmt::Display,
    A::Real: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter
            = self.data.iter().zip(&self.svals).zip(&self.outs).enumerate();
        for (k, ((g, l), out)) in iter {
            let sh = g.shape();
            writeln!(f, "Γ[{}] :: {{ <{}>, {}<{}>, <{}> }}",
                k, sh[0], out.label(), sh[1], sh[2]
            )?;
            g.fmt(f)?;
            writeln!(f)?;
            write!(f, "Λ[{}] = [", k)?;
            for (j, lj) in l.iter().enumerate() {
                lj.fmt(f)?;
                if j < sh[2] - 1 { write!(f, ", ")?; }
            }
            writeln!(f, "]")?;
        }
        let sh = self.data[self.n - 1].shape();
        writeln!(f, "Γ[{}] :: {{ <{}>, {}<{}>, <{}> }}",
            self.n - 1, sh[0], self.outs[self.n - 1].label(), sh[1], sh[2]
        )?;
        self.data[self.n - 1].fmt(f)?;
        Ok(())
    }
}

/// Wrapper around a physical index originally belonging to a pure state, now in
/// the context of a density matrix.
///
/// This type implements `Ord` if `T` implements it, sorting all `Ket` indices
/// first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MatIndex<T> {
    /// Corresponds to the ket/row side of the density matrix.
    Ket(T),
    /// Corresponds to the bra/column side of the density matrix.
    Bra(T),
}

impl<T> Idx for MatIndex<T>
where T: Idx
{
    fn dim(&self) -> usize {
        match self {
            Self::Ket(idx) => idx.dim(),
            Self::Bra(idx) => idx.dim(),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Ket(idx) => idx.label(),
            Self::Bra(idx) => idx.label() + "'",
        }
    }
}

impl<T> MatIndex<T> {
    /// Return `true` if `self` is `Ket`.
    #[inline]
    pub fn is_ket(&self) -> bool { matches!(self, Self::Ket(..)) }

    /// Return `true` if `self` is `Bra`.
    #[inline]
    pub fn is_bra(&self) -> bool { matches!(self, Self::Bra(..)) }
}

impl<T> PartialOrd for MatIndex<T>
where T: PartialOrd
{
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match (self, rhs) {
            (Self::Ket(l), Self::Ket(r)) => l.partial_cmp(r),
            (Self::Ket(_), Self::Bra(_)) => Some(Ordering::Less),
            (Self::Bra(_), Self::Ket(_)) => Some(Ordering::Greater),
            (Self::Bra(l), Self::Bra(r)) => l.partial_cmp(r),
        }
    }
}

impl<T> Ord for MatIndex<T>
where T: Ord
{
    fn cmp(&self, rhs: &Self) -> Ordering {
        match (self, rhs) {
            (Self::Ket(l), Self::Ket(r)) => l.cmp(r),
            (Self::Ket(_), Self::Bra(_)) => Ordering::Less,
            (Self::Bra(_), Self::Ket(_)) => Ordering::Greater,
            (Self::Bra(l), Self::Bra(r)) => l.cmp(r),
        }
    }
}

/// Wrapper around an index originally belonging to a [`MPS`], now used in the
/// context of a [`Network`].
///
/// Because all indices in a `Network` must be unique up to pairing, every
/// non-physical index in the `MPS` is identified as either the left or right
/// bond of a particular singular value vector in the `MPS`, starting at 0 on
/// the leftmost such vector in the original factorization.
///
/// ```text
///       BondL { label: 0, .. }
///       |        BondR { label: 0, .. }
///       |        |        BondL { label: 1, .. }
///       |        |        |        BondR { label: 1, .. }
///       |        |        |        |       BondL { label: n-2, .. }
///       |        |        |        |       |          BondR { label: n-2, .. }
///       V        V        V        V       V          V
/// Γ[0] --- Λ[0] --- Γ[1] --- Λ[1] --- ... --- Λ[n-2] --- Γ[n-1]
///  |                 |                                     |
///  | <- Physical(..) | <- Physical(..)                     | <- Physical(..)
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MPSIndex<T> {
    /// Left index of the `label`-th singular value vector.
    BondL { label: usize, dim: usize },
    /// Right index of the `label`-th singular value vector.
    BondR { label: usize, dim: usize },
    /// Unpaired, physical index.
    Physical(T),
}

impl<T> Idx for MPSIndex<T>
where T: Idx
{
    fn dim(&self) -> usize {
        match self {
            Self::BondL { label: _, dim: d } => *d,
            Self::BondR { label: _, dim: d } => *d,
            Self::Physical(idx) => idx.dim(),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::BondL { label: i, dim: _ } => format!("mps:<{i}"),
            Self::BondR { label: i, dim: _ } => format!("mps:{i}>"),
            Self::Physical(idx) => idx.label(),
        }
    }
}

impl<T> MPSIndex<T> {
    /// Return `true` if `self` is `BondL`.
    #[inline]
    pub fn is_lbond(&self) -> bool { matches!(self, Self::BondL { .. }) }

    /// Return `true` if `self` is `BondR`.
    #[inline]
    pub fn is_rbond(&self) -> bool { matches!(self, Self::BondR { .. }) }

    /// Return `true` if `self` is `Physical`.
    #[inline]
    pub fn is_physical(&self) -> bool { matches!(self, Self::Physical(..)) }

    /// Unwrap `self` into a bare physical index.
    ///
    /// *Panics* if `self` is `BondL` or `BondR`.
    #[inline]
    pub fn into_physical(self) -> T {
        match self {
            Self::Physical(idx) => idx,
            Self::BondL { .. } => panic!(
                "cannot unwrap a MPSIndex::BondL into a bare physical index"
            ),
            Self::BondR { .. } => panic!(
                "cannot unwrap a MPSIndex::BondR into a bare physical index"
            ),
        }
    }

    // convert to a ket-side MPSMatIndex
    fn into_mpsmat(self) -> MPSMatIndex<T> { MPSMatIndex::from_mps(self) }
}

/// Wrapper around an index originally belonging to a [`MPS`], now used in the
/// context of a [`Network`] for a density matrix.
///
/// This index type is a fusion of a [`MPSIndex`] with a [`MatIndex`]: it
/// distinguishes both internal MPS indices from physical ones as well as those
/// belonging to the ket (row) and bra (column) sides of the density matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MPSMatIndex<T> {
    /// Left index of the `label`-th singular value vector on the ket side.
    BondLKet { label: usize, dim: usize },
    /// Left index of the `label`-th singular value vector on the bra side.
    BondLBra { label: usize, dim: usize },
    /// Right index of the `label`-th singular value vector on the ket side.
    BondRKet { label: usize, dim: usize },
    /// Right index of the `label`-th singular value vector on the bra side.
    BondRBra { label: usize, dim: usize },
    /// Unpaired, ket-side physical index.
    PhysicalKet(T),
    /// Unpaired, bra-side physical index.
    PhysicalBra(T),
}

impl<T> Idx for MPSMatIndex<T>
where T: Idx
{
    fn dim(&self) -> usize {
        match self {
            Self::BondLKet { label: _, dim: d } => *d,
            Self::BondLBra { label: _, dim: d } => *d,
            Self::BondRKet { label: _, dim: d } => *d,
            Self::BondRBra { label: _, dim: d } => *d,
            Self::PhysicalKet(idx) => idx.dim(),
            Self::PhysicalBra(idx) => idx.dim(),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::BondLKet { label: i, dim: _ } => format!("mpsmat:<{i}"),
            Self::BondLBra { label: i, dim: _ } => format!("mpsmat:<{i}'"),
            Self::BondRKet { label: i, dim: _ } => format!("mpsmat:{i}>"),
            Self::BondRBra { label: i, dim: _ } => format!("mpsmat:{i}>'"),
            Self::PhysicalKet(idx) => idx.label(),
            Self::PhysicalBra(idx) => format!("{}'", idx.label()),
        }
    }
}

impl<T> MPSMatIndex<T> {
    /// Return `true` if `self` is `BondLKet`.
    #[inline]
    pub fn is_lbondket(&self) -> bool { matches!(self, Self::BondLKet { .. }) }

    /// Return `true` if `self` is `BondLBra`.
    #[inline]
    pub fn is_lbondbra(&self) -> bool { matches!(self, Self::BondLBra { .. }) }

    /// Return `true` if `self` is `BondLKet` or `BondLBra`.
    #[inline]
    pub fn is_lbond(&self) -> bool {
        matches!(self, Self::BondLKet { .. } | Self::BondLBra { .. })
    }

    /// Return `true` if `self` is `BondRKet`.
    #[inline]
    pub fn is_rbondket(&self) -> bool { matches!(self, Self::BondRKet { .. }) }

    /// Return `true` if `self` is `BondRBra`.
    #[inline]
    pub fn is_rbondbra(&self) -> bool { matches!(self, Self::BondRBra { .. }) }

    /// Return `true` if `self` is `BondRKet` or `BondRBra`.
    #[inline]
    pub fn is_rbond(&self) -> bool {
        matches!(self, Self::BondRKet { .. } | Self::BondRBra { .. })
    }

    /// Return `true` if `self` is `PhysicalKet`.
    #[inline]
    pub fn is_physicalket(&self) -> bool { matches!(self, Self::PhysicalKet(..)) }

    /// Return `true` if `self` is `PhysicalBra`.
    #[inline]
    pub fn is_physicalbra(&self) -> bool { matches!(self, Self::PhysicalBra(..)) }

    /// Return `true` if `self` is `PhysicalKet` or `PhysicalBra`.
    #[inline]
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::PhysicalKet(..) | Self::PhysicalBra(..))
    }

    /// Return `true` if `self` is `BondLKet`, `BondRKet`, or `PhysicalKet`.
    #[inline]
    pub fn is_ket(&self) -> bool {
        matches!(
            self,
            Self::BondLKet { .. }
            | Self::BondRKet { .. }
            | Self::PhysicalKet(..)
        )
    }

    /// Return `true` if `self` is `BondLBra`, `BondRBra`, or `PhysicalBra`.
    #[inline]
    pub fn is_bra(&self) -> bool {
        matches!(
            self,
            Self::BondLBra { .. }
            | Self::BondRBra { .. }
            | Self::PhysicalBra(..)
        )
    }

    /// Unwrap `self` into a physical index, preserving ket/bra-ness.
    ///
    /// *Panics* if `self` is `BondLKet`, `BondLBra`, `BondRKet`, or `BondRBra`.
    #[inline]
    pub fn into_physical(self) -> MatIndex<T> {
        match self {
            Self::PhysicalKet(idx) => MatIndex::Ket(idx),
            Self::PhysicalBra(idx) => MatIndex::Bra(idx),
            Self::BondLKet { .. } => panic!(
                "cannot unwrap a MPSMatIndex::BondLKet into a physical index"
            ),
            Self::BondLBra { .. } => panic!(
                "cannot unwrap a MPSMatIndex::BondLBra into a physical index"
            ),
            Self::BondRKet { .. } => panic!(
                "cannot unwrap a MPSMatIndex::BondRKet into a physical index"
            ),
            Self::BondRBra { .. } => panic!(
                "cannot unwrap a MPSMatIndex::BondRBra into a physical index"
            ),
        }
    }

    #[inline]
    fn from_mps(idx: MPSIndex<T>) -> Self {
        match idx {
            MPSIndex::BondL { label, dim } => Self::BondLKet { label, dim },
            MPSIndex::BondR { label, dim } => Self::BondRKet { label, dim },
            MPSIndex::Physical(idx) => Self::PhysicalKet(idx),
        }
    }

    /// Flip the bra/ket-ness of `self`.
    #[inline]
    pub fn conj(self) -> Self {
        match self {
            Self::BondLKet { label, dim } => Self::BondLBra { label, dim },
            Self::BondLBra { label, dim } => Self::BondLKet { label, dim },
            Self::BondRKet { label, dim } => Self::BondRBra { label, dim },
            Self::BondRBra { label, dim } => Self::BondRKet { label, dim },
            Self::PhysicalKet(idx) => Self::PhysicalBra(idx),
            Self::PhysicalBra(idx) => Self::PhysicalKet(idx),
        }
    }
}

// compute the matrix (natural) logarithm via eigendecomp
// panics of the matrix is not Hermitian
pub(crate) fn mat_ln<A>(x: &nd::Array2<A>) -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt + Scalar + Lapack,
    nd::Array2<A>: Eigh<EigVal = nd::Array1<<A as ComplexFloat>::Real>, EigVec = nd::Array2<A>>,
{
    let (e, v) = x.eigh(UPLO::Upper).expect("mat_ln: error in diagonalization");
    let u = v.t().mapv(|a| a.conj());
    let log_e = nd::Array2::from_diag(
        &e.mapv(|ek| <A as ComplexFloatExt>::from_real(Float::ln(ek)))
    );
    v.dot(&log_e).dot(&u)
}

// compute an integer power of a matrix
// panics if the matrix is not square
// expecting most cases to use n < 5
pub(crate) fn mat_pow<A>(x: &nd::Array2<A>, n: u32) -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    nd::Array2<A>: nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>> + Clone
{
    if !x.is_square() { panic!("mat_pow: non-square matrix"); }
    match n {
        0 => nd::Array2::eye(x.shape()[0]),
        1 => x.clone(),
        m if m < 4 => {
            let mut acc = x.clone();
            for _ in 1..m { acc = acc.dot(x); }
            acc
        },
        m if m < 12 => {
            // exponent on the largest power of 2 that's ≤ m
            let k: u32 = (1..).find(|p| 1 << (p + 1) > m).unwrap();
            let r: u32 = m - (1 << k); // remainder
            let mut acc = x.clone();
            for _ in 0..k { acc = acc.dot(&acc); }
            for _ in 0..r { acc = acc.dot(x); }
            acc
        },
        m => {
            // exponent on the largest power of 2 that's ≤ m
            let k1: u32 = (1..).find(|p| 1 << (p + 1) > m).unwrap();
            let r1: u32 = m - (1 << k1); // remainder
            // expoment on the largest power of 2 that's ≤ r1
            let k2: u32 = (1..).find(|p| 1 << (p + 1) > r1).unwrap();
            let r: u32 = r1 - (1 << k2); // remainder
            let mut acc1 = x.clone();
            for _ in 0..k2 { acc1 = acc1.dot(&acc1); } // k2 ≤ k1
            let acc2 = acc1.clone(); // save the intermediate result
            for _ in 0..k1 - k2 { acc1 = acc1.dot(&acc1); }
            acc1 = acc1.dot(&acc2);
            for _ in 0..r { acc1 = acc1.dot(x); }
            acc1
        }
    }
}

/// Compute the Von Neumann entropy of a density matrix.
///
/// The input must be a valid density matrix, but can be pure or mixed.
pub fn entropy_vn<A>(rho: &nd::Array2<A>) -> <A as ComplexFloat>::Real
where
    A: ComplexFloat + ComplexFloatExt + Scalar + Lapack,
    nd::Array2<A>:
        Eigh<EigVal = nd::Array1<<A as ComplexFloat>::Real>, EigVec = nd::Array2<A>>
        + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
{
    rho.dot(&mat_ln(rho))
        .diag().iter().copied()
        .map(ComplexFloat::re)
        .fold(
            <A as ComplexFloat>::Real::zero(),
            <A as ComplexFloat>::Real::add,
        ) * (-<A as ComplexFloat>::Real::one())
}

/// Compute the *n*-th Rényi entropy of a density matrix.
///
/// Returns the [Von Neumann entropy][entropy_vn] if `n == 1`. The input must be
/// a valid density matrix, but can be pure or mixed.
pub fn entropy_ry<A>(rho: &nd::Array2<A>, n: u32) -> <A as ComplexFloat>::Real
where
    A: ComplexFloat + ComplexFloatExt + Scalar + Lapack,
    nd::Array2<A>:
        Eigh<EigVal = nd::Array1<<A as ComplexFloat>::Real>, EigVec = nd::Array2<A>>
        + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
{
    if n == 1 {
        entropy_vn(rho)
    } else {
        let nfloat: <A as ComplexFloat>::Real
            = (0..n)
            .map(|_| <A as ComplexFloat>::Real::one())
            .fold(
                <A as ComplexFloat>::Real::zero(),
                <A as ComplexFloat>::Real::add,
            );
        let prefactor = Float::recip(<A as ComplexFloat>::Real::one() - nfloat);
        Float::ln(
            mat_pow(rho, n)
                .diag().iter().copied()
                .map(ComplexFloat::re)
                .fold(
                    <A as ComplexFloat>::Real::zero(),
                    <A as ComplexFloat>::Real::add,
                )
        ) * prefactor
    }
}

// pub fn entropy_renyi(rho: &nd::Array2<C32>, n: u32) -> f32 {
//     if n == 1 {
//         entropy_vn(rho)
//     } else {
//         let pref = (1.0 - n as f32).recip();
//         mat_pow(rho, n)
//             .diag().iter()
//             .map(|r| r.re)
//             .sum::<f32>()
//             .ln() * pref
//     }
// }

// /// A gate in a quantum circuit, used for [`MPS`][crate::mps::MPS] and
// /// [`MPSCircuit`][crate::mpscircuit::MPSCircuit].
// ///
// /// Two-qubit gates are limited to nearest neighbors.
// #[derive(Copy, Clone, Debug, PartialEq)]
// pub enum MPSGate {
//     /// A gate formed from an Euler angle decomposition.
//     U(usize, f32, f32, f32),
//     /// Hadamard.
//     H(usize),
//     /// π rotation about X.
//     X(usize),
//     /// π rotation about Z.
//     Z(usize),
//     /// π/2 rotation about Z.
//     S(usize),
//     /// –π/2 rotation about Z.
//     SInv(usize),
//     /// Arbitrary rotation about X.
//     XRot(usize, f32),
//     /// Arbitrary rotation about Z.
//     ZRot(usize, f32),
//     /// Z-controlled π rotation about X.
//     CX(usize),
//     /// Z-controlled π rotation about Z.
//     CZ(usize),
//     /// `CX` with the control and target reversed.
//     CXRev(usize),
// }
//
// impl Gate {
//     /// Return `true` if `self` is `U`.
//     pub fn is_u(&self) -> bool { matches!(self, Self::U(..)) }
//
//     /// Return `true` if `self` is `H`.
//     pub fn is_h(&self) -> bool { matches!(self, Self::H(..)) }
//
//     /// Return `true` if `self` is `X`.
//     pub fn is_x(&self) -> bool { matches!(self, Self::X(..)) }
//
//     /// Return `true` if `self` is `Z`.
//     pub fn is_z(&self) -> bool { matches!(self, Self::Z(..)) }
//
//     /// Return `true` if `self` is `S`.
//     pub fn is_s(&self) -> bool { matches!(self, Self::S(..)) }
//
//     /// Return `true` if `self` is `SInv`.
//     pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv(..)) }
//
//     /// Return `true` if `self` is `XRot`.
//     pub fn is_xrot(&self) -> bool { matches!(self, Self::XRot(..)) }
//
//     /// Return `true` if `self` is `ZRot`.
//     pub fn is_zrot(&self) -> bool { matches!(self, Self::ZRot(..)) }
//
//     /// Return `true` if `self` is `CX`.
//     pub fn is_cx(&self) -> bool { matches!(self, Self::CX(..)) }
//
//     /// Return `true` if `self` is `CZ`.
//     pub fn is_cz(&self) -> bool { matches!(self, Self::CZ(..)) }
//
//     /// Return `true` if `self` is `CXRev`.
//     pub fn is_cxrev(&self) -> bool { matches!(self, Self::CXRev(..)) }
//
//     /// Return the [kind][G] of `self`.
//     pub fn kind(&self) -> G {
//         use G::*;
//         use G1::*;
//         use G2::*;
//         match *self {
//             Self::U(..) => Q1(U),
//             Self::H(..) => Q1(H),
//             Self::X(..) => Q1(X),
//             Self::Y(..) => Q1(Y),
//             Self::Z(..) => Q1(Z),
//             Self::S(..) => Q1(S),
//             Self::SInv(..) => Q1(SInv),
//             Self::XRot(..) => Q1(XRot),
//             Self::YRot(..) => Q1(YRot),
//             Self::ZRot(..) => Q1(ZRot),
//             Self::CX(..) => Q2(CX),
//             Self::CY(..) => Q2(CY),
//             Self::CZ(..) => Q2(CZ),
//         }
//     }
//
//     /// Sample a random single gate.
//     ///
//     /// General rotation gates will sample their angles uniformly from `[0,
//     /// 2π)`. [`U`][Self::U] gates will sample all three rotation angles
//     /// independently.
//     pub fn sample_single<G, R>(kind: G, op: G::QubitArg, rng: &mut R) -> Self
//     where
//         G: GateToken<Self>,
//         R: Rng + ?Sized,
//     {
//         kind.sample(op, rng)
//     }
//
//     /// Sample a random one-qubit Clifford gate.
//     pub fn sample_c1<R>(k: usize, rng: &mut R) -> Self
//     where R: Rng + ?Sized
//     {
//         match rng.gen_range(0..6) {
//             0 => Self::H(k),
//             1 => Self::X(k),
//             2 => Self::Y(k),
//             3 => Self::Z(k),
//             4 => Self::S(k),
//             5 => Self::SInv(k),
//             _ => unreachable!(),
//         }
//     }
//
//     /// Sample a random gate uniformly from a set.
//     pub fn sample_from<'a, I, G, R>(kinds: I, arg: G::QubitArg, rng: &mut R)
//         -> Self
//     where
//         I: IntoIterator<Item = &'a G>,
//         G: GateToken<Self> + 'a,
//         R: Rng + ?Sized
//     {
//         let kinds: Vec<&G> = kinds.into_iter().collect();
//         let n = kinds.len();
//         kinds[rng.gen_range(0..n)].sample(arg, rng)
//     }
// }

