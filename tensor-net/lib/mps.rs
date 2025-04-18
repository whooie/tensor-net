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
//! use num_complex::Complex64 as C64;
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
//! let h: nd::Array2<C64> = // hadamard
//!     nd::array![
//!         [C64::from(0.5).sqrt(),  C64::from(0.5).sqrt()],
//!         [C64::from(0.5).sqrt(), -C64::from(0.5).sqrt()],
//!     ];
//! let cx: nd::Array2<C64> = // CNOT
//!     nd::array![
//!         [C64::one(),  C64::zero(), C64::zero(), C64::zero()],
//!         [C64::zero(), C64::one(),  C64::zero(), C64::zero()],
//!         [C64::zero(), C64::zero(), C64::zero(), C64::one() ],
//!         [C64::zero(), C64::zero(), C64::one(),  C64::zero()],
//!     ];
//!
//! let n: usize = 4; // number of qubits
//! let indices: Vec<Q> = (0..n).map(Q).collect();
//!
//! // initialize to the first available quantum number for all particles,
//! // i.e. ∣00000⟩
//! let mut mps: MPS<Q, C64> = MPS::new(indices, None).unwrap();
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
//! // contract the MPS into an ordinary 1D state vector
//! let (_, state) = mps.into_contract();
//! println!("{state}");
//! // either [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//! // or     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
//! // based on the result of the measurement
//! ```

use std::{
    cmp::Ordering,
    fmt,
    ops::{ Add, Range },
};
// use itertools::Itertools;
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
    ComplexFloatExt,
    circuit::Q,
    gate::{ self, Gate },
    network::{ Network, Pool },
    tensor::{ Idx, Tensor },
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

/// Specify a method to set the bond dimension from a Schmidt decomposition.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BondDim<A> {
    /// Truncate to a fixed upper bound. The resulting bond dimensions will be
    /// *at most* this value and at least 1.
    Const(usize),
    /// Truncate based on a threshold imposed on Schmidt values. For threshold
    /// value small enough, this allows the maximum bond dimension to grow
    /// exponentially with system size.
    Cutoff(A),
}

impl<A> BondDim<A> {
    /// Return the upper bound if `self` is `Const`.
    pub fn bound(&self) -> Option<usize> {
        match self {
            Self::Const(x) => Some(*x),
            Self::Cutoff(_) => None,
        }
    }
}

/// Convenience trait to identify complex numbers that can be used in
/// linear-algebraic operations.
pub trait ComplexLinalgScalar
where
    Self:
        ComplexFloat<Real = Self::Re>
        + ComplexFloatExt
        + Scalar<Real = Self::Re>
        + Lapack,
    Self::Re: std::fmt::Debug,
{
    /// Type for associated real values.
    type Re: Float + NumAssign;
}

impl<A> ComplexLinalgScalar for A
where
    A:
        ComplexFloat<Real = <A as Scalar>::Real>
        + ComplexFloatExt
        + Scalar
        + Lapack,
{
    type Re = <A as Scalar>::Real;
}

/// Data struct holding a Schmidt decomposition repurposed for MPS
/// factorization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schmidt<A>
where A: ComplexLinalgScalar
{
    /// Matrix whose columns are the left Schmidt vectors.
    pub u: nd::Array2<A>,
    /// Vector of Schmidt values.
    pub s: nd::Array1<A::Re>,
    /// Matrix whose rows are the right Schmidt vectors, scaled by their
    /// respective Schmidt values.
    pub q: nd::Array2<A>,
    /// Schmidt rank.
    pub rank: usize,
}

/// Succinctly names a complex-valued 2D array that can be factorized via
/// Schmidt/singular value decomposition.
pub trait SchmidtDecomp<A>
where
    A: ComplexLinalgScalar,
    Self: SVDInto<U = nd::Array2<A>, Sigma = nd::Array1<A::Re>, VT = nd::Array2<A>>,
{
    /// Return the Schmidt decomposition of `self`, with each of the right
    /// Schmidt vectors multiplied by their respective singular values.
    ///
    /// `trunc` defaults to a [`Cutoff`][BondDim::Cutoff] at machine epsilon for
    /// `A::Real`.
    fn local_decomp(self, trunc: Option<BondDim<A::Re>>) -> Schmidt<A>;
}

fn process_svd<A>(
    u: nd::Array2<A>,
    mut s: nd::Array1<A::Re>,
    mut q: nd::Array2<A>,
    trunc: Option<BondDim<A::Re>>,
) -> Schmidt<A>
where A: ComplexLinalgScalar
{
    let mut norm: A::Re =
        s.iter()
        .map(|sj| sj.powi(2))
        .fold(A::Re::zero(), A::Re::add)
        .sqrt();
    s.map_inplace(|sj| { *sj /= norm; });
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
    let rankslice = nd::Slice::new(0, Some(rank as isize), 1);
    s.slice_axis_inplace(nd::Axis(0), rankslice);
    norm
        = s.iter()
        .map(|sj| sj.powi(2))
        .fold(A::Re::zero(), A::Re::add)
        .sqrt();
    s.map_inplace(|sj| { *sj /= norm; });
    q.slice_axis_inplace(nd::Axis(0), rankslice);
    let renorm = A::from_re(norm);
    nd::Zip::from(q.axis_iter_mut(nd::Axis(0)))
        .and(&s)
        .for_each(|mut qv, sv| {
            let sv_renorm = A::from_re(*sv) * renorm;
            qv.map_inplace(|qvj| { *qvj *= sv_renorm; });
        });
    let u = u.slice(nd::s![.., ..rank]).to_owned();
    Schmidt { u, s, q, rank }
}

impl<A> SchmidtDecomp<A> for nd::Array2<A>
where A: ComplexLinalgScalar
{
    fn local_decomp(self, trunc: Option<BondDim<A::Re>>) -> Schmidt<A> {
        let (Some(u), s, Some(q)) = self.svd_into(true, true).unwrap()
            else { unreachable!() };
        process_svd(u, s, q, trunc)
    }
}

/// A matrix product (pure) state.
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
/// > [`Network`], then all physical indices need to be distinguishable. See
/// > [`Idx`] for more information.
#[derive(Clone, PartialEq)]
pub struct MPS<T, A>
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
    pub(crate) svals: Vec<nd::Array1<A::Re>>, // length n - 1
    // Threshold for singular values.
    pub(crate) trunc: Option<BondDim<A::Re>>,
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
{
    /// Initialize to a separable product state with each particle in the first
    /// of its available eigenstates.
    ///
    /// Optionally provide a global truncation method for discarding singular
    /// values. Defaults to a [`Cutoff`][BondDim::Cutoff] at machine epsilon.
    ///
    /// Fails if no physical indices are provided.
    pub fn new<I>(indices: I, trunc: Option<BondDim<A::Re>>)
        -> MPSResult<Self>
    where I: IntoIterator<Item = T>
    {
        let indices: Vec<T> = indices.into_iter().collect();
        if indices.is_empty() { return Err(EmptySystem); }
        if indices.iter().any(|i| i.dim() == 0) { return Err(UnphysicalIndex); }
        let n = indices.len();
        let data: Vec<nd::Array3<A>> =
            indices.iter()
            .map(|idx| {
                let mut g: nd::Array3<A> = nd::Array::zeros((1, idx.dim(), 1));
                if let Some(g000) = g.first_mut() { *g000 = A::one(); }
                g
            })
            .collect();
        let svals: Vec<nd::Array1<A::Re>> =
            (0..n - 1)
            .map(|_| nd::array![A::Re::one()])
            .collect();
        Ok(Self { n, data, outs: indices, svals, trunc })
    }

    /// Initialize to a separable product state with each particle wholly in one
    /// of its available eigenstates.
    ///
    /// Optionally provide a global truncation method for discarding singular
    /// values. Defaults to a [`Cutoff`][BondDim::Cutoff] at machine epsilon.
    ///
    /// Fails if no physical indices are provided or the given quantum number
    /// for an index exceeds the available range defined by the index's [`Idx`]
    /// implementation.
    pub fn new_qnums<I>(indices: I, trunc: Option<BondDim<A::Re>>)
        -> MPSResult<Self>
    where I: IntoIterator<Item = (T, usize)>
    {
        let indices: Vec<(T, usize)> = indices.into_iter().collect();
        if indices.is_empty() { return Err(EmptySystem); }
        if indices.iter().any(|(idx, qnum)| *qnum >= idx.dim()) {
            return Err(InvalidQuantumNumber);
        }
        let n = indices.len();
        let data: Vec<nd::Array3<A>> =
            indices.iter()
            .map(|(idx, qnum)| {
                let mut g: nd::Array3<A> = nd::Array::zeros((1, idx.dim(), 1));
                g[[0, *qnum, 0]] = A::one();
                g
            })
            .collect();
        let svals: Vec<nd::Array1<A::Re>> =
            (0..n - 1)
            .map(|_| nd::array![A::Re::one()])
            .collect();
        let indices: Vec<T> =
            indices.into_iter()
            .map(|(idx, _)| idx)
            .collect();
        Ok(Self { n, data, outs: indices, svals, trunc })
    }
}

impl<T, A> MPS<T, A>
where A: ComplexLinalgScalar
{
    /// Return the number of particles.
    pub fn n(&self) -> usize { self.n }

    /// Return a reference to all physical indices.
    pub fn indices(&self) -> &Vec<T> { &self.outs }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
    nd::Array2<A>: SchmidtDecomp<A>,
{
    fn factorize(
        outs: &[T],
        state: nd::Array1<A>,
        trunc: Option<BondDim<A::Re>>,
    ) -> (Vec<nd::Array3<A>>, Vec<nd::Array1<A::Re>>)
    {
        let n = outs.len(); // assume n ≥ 1
        let mut data: Vec<nd::Array3<A>> = Vec::with_capacity(n);
        let mut svals: Vec<nd::Array1<A::Re>> =
            Vec::with_capacity(n - 1);
        let mut udim: usize = 1;
        let mut outdim: usize;
        let mut statelen = state.len();
        let mut q: nd::Array2<A> = state.into_shape((1, statelen)).unwrap();
        let mut reshape_m: usize;
        let mut reshape_n: usize;
        for outk in outs.iter().take(n - 1) {
            outdim = outk.dim();

            // initial reshape to fuse outk with the previous schmidt index
            statelen = q.len();
            reshape_m = udim * outdim;
            reshape_n = statelen / reshape_m;
            q = q.into_shape((reshape_m, reshape_n)).unwrap();

            // svd/schmidt decomp: left vectors are columns in u, schmidt values
            // are in s, and right vectors are rows in q
            let Schmidt { u, s, q: qp, rank } = q.local_decomp(trunc);
            q = qp;
            // prepare the Γ tensor from u; Γ_k = (s_{k-1})^-1 . u
            // can't slice u in place on the column index because it would mean
            // the remaining data won't be contiguous in memory; this would make
            // the reshape fail
            let mut g =
                u.slice(nd::s![.., ..rank]).to_owned()
                .into_shape((udim, outdim, rank))
                .unwrap();
            if let Some(slast) = svals.last() {
                nd::Zip::from(g.axis_iter_mut(nd::Axis(0)))
                    .and(slast)
                    .for_each(|mut gv, sv| {
                        gv.map_inplace(|gvk| { *gvk /= A::from_real(*sv); });
                    });
            }

            // done for the particle
            data.push(g);
            svals.push(s);
            udim = rank;
        }
        // final Γ tensor
        let mut g = q.into_shape((udim, outs[n - 1].dim(), 1)).unwrap();
        nd::Zip::from(g.axis_iter_mut(nd::Axis(0)))
            .and(&svals[n - 2])
            .for_each(|mut gv, sv| {
                gv.map_inplace(|gvk| { *gvk /= A::from_real(*sv); });
            });
        data.push(g);

        (data, svals)
    }

    /// Initialize by factoring an existing pure state vector.
    ///
    /// Optionally provide a global cutoff threshold for singular values, which
    /// defaults to zero.
    ///
    /// Fails if no particle indices are provided or if the length of the
    /// initial state vector does not agree with the indices.
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
        let mut state: nd::Array1<A> = state.into_iter().collect();
        if state.len() != statelen { return Err(StateIncompatibleShape); }
        let norm = ComplexFloat::sqrt(
            state.iter()
                .map(|a| *a * a.conj())
                .fold(A::zero(), A::add)
        );
        state.map_inplace(|a| { *a /= norm; });
        let n = indices.len();
        if n == 1 {
            let mut g: nd::Array3<A> =
                nd::Array::zeros((1, indices[0].dim(), 1));
            state.move_into(g.slice_mut(nd::s![0, .., 0]));
            let data: Vec<nd::Array3<A>> = vec![g];
            let svals: Vec<nd::Array1<A::Re>> = Vec::with_capacity(0);
            Ok(Self { n, data, outs: indices, svals, trunc })
        } else {
            let (data, svals) = Self::factorize(&indices, state, trunc);
            Ok(Self { n, data, outs: indices, svals, trunc })
        }
    }

    /// Initialize by factoring an existing tensor representing a pure state.
    ///
    /// Not the additional `Ord` bound on the index type, which is used to
    /// ensure determinism in the MPS factorization.
    ///
    /// Optionally provide a global cutoff threshold for singular values, which
    /// defaults to zero.
    ///
    /// Fails if no particle indices are provided.
    pub fn from_tensor(
        mut state: Tensor<T, A>,
        trunc: Option<BondDim<A::Re>>,
    ) -> MPSResult<Self>
    where T: Ord
    {
        state.sort_indices();
        let (indices, mut state) = state.into_flat();
        if indices.is_empty() { return Err(EmptySystem); }
        if indices.iter().any(|i| i.dim() == 0) { return Err(UnphysicalIndex); }
        let norm = ComplexFloat::sqrt(
            state.iter()
                .map(|a| *a * a.conj())
                .fold(A::zero(), A::add)
        );
        state.map_inplace(|a| { *a /= norm; });
        let n = indices.len();
        if n == 1 {
            let mut g: nd::Array3<A> =
                nd::Array::zeros((1, indices[0].dim(), 1));
            state.into_owned().move_into(g.slice_mut(nd::s![0, .., 0]));
            let data: Vec<nd::Array3<A>> = vec![g];
            let svals: Vec<nd::Array1<A::Re>> = Vec::with_capacity(0);
            Ok(Self { n, data, outs: indices, svals, trunc })
        } else {
            let (data, svals) =
                Self::factorize(&indices, state.into_owned(), trunc);
            Ok(Self { n, data, outs: indices, svals, trunc })
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct BondData<'a, A>
where A: ComplexLinalgScalar
{
    ls: Option<&'a nd::Array1<A::Re>>,
    l: &'a nd::Array3<A>,
    s: &'a nd::Array1<A::Re>,
    r: &'a nd::Array3<A>,
    rs: Option<&'a nd::Array1<A::Re>>,
}

impl<T, A> MPS<T, A>
where A: ComplexLinalgScalar
{
    // assume `k` is in bounds
    fn get_bond_data(&self, k: usize) -> BondData<'_, A> {
        let ls = (k != 0).then(|| &self.svals[k - 1]);
        let l = &self.data[k];
        let s = &self.svals[k];
        let r = &self.data[k + 1];
        let rs = (k != self.n - 2).then(|| &self.svals[k + 1]);
        BondData { ls, l, s, r, rs }
    }

    // do a single contraction between two particles' gamma matrices
    //
    // the result is returned as a 2D array in which the left and right physical
    // indices are fused with the outermost indices in the contraction; i.e.
    // for axis signatures
    //      l :: { v, sl, u }
    //      s :: { u }
    //      r :: { u, sr, w }
    // the result has axis signature
    //      q :: { u<>sl, sr<>w }
    // (where `<>` denotes fusion)
    fn do_contract2(
        l: &nd::Array3<A>,
        s: &nd::Array1<A::Re>,
        r: &nd::Array3<A>,
    ) -> nd::Array2<A>
    {
        let shl = l.dim();
        let l =
            l.as_standard_layout()
            .into_shape((shl.0 * shl.1, shl.2))
            .unwrap();
        let shr = r.dim();
        let mut r =
            r.as_standard_layout()
            .into_shape((shr.0, shr.1 * shr.2))
            .unwrap();
        nd::Zip::from(r.rows_mut())
            .and(s.view())
            .for_each(|mut ru__, su| {
                let su = A::from_re(*su);
                ru__.map_inplace(|rusw| { *rusw *= su; });
            });
        l.dot(&r)
    }

    // like `do_contract2`, but with the return value as a 3D array with
    // physical indices fused in axis 1; i.e.
    // for axis signatures
    //      l :: { v, sl, u }
    //      s :: { u }
    //      r :: { u, sr, w }
    // the result has axis signature
    //      q :: { u, sl<>sr, w }
    // (where `<>` denotes fusion)
    fn do_contract3(
        l: &nd::Array3<A>,
        s: &nd::Array1<A::Re>,
        r: &nd::Array3<A>,
    ) -> nd::Array3<A>
    {
        let shl = l.shape();
        let shr = r.shape();
        Self::do_contract2(l, s, r)
            .into_shape((shl[0], shl[1] * shr[1], shr[2]))
            .unwrap()
    }

    // like `do_contract2`, but with the return value as a 4D array with
    // all indices unfused; i.e.
    // for axis signatures
    //      l :: { v, sl, u }
    //      s :: { u }
    //      r :: { u, sr, w }
    // the result has axis signature
    //      q :: { u, sl, sr, w }
    fn do_contract4(
        l: &nd::Array3<A>,
        s: &nd::Array1<A::Re>,
        r: &nd::Array3<A>,
    ) -> nd::Array4<A>
    {
        let shl = l.shape();
        let shr = r.shape();
        Self::do_contract2(l, s, r)
            .into_shape((shl[0], shl[1], shr[1], shr[2]))
            .unwrap()
    }

    // contraction via binary division as a rough heuristic to minimize runtime
    // at the cost of having to allocate intermediate results
    fn do_contract_multi(data: &[nd::Array3<A>], svals: &[nd::Array1<A::Re>])
        -> nd::Array3<A>
    {
        match data.len() {
            0 => nd::array![[[A::one()]]],
            1 => data[0].clone(),
            2 => Self::do_contract3(&data[0], &svals[0], &data[1]),
            n => {
                let n2 = n / 2;
                let d_left = &data[..n2];
                let s_left = &svals[..n2 - 1];
                let d_right = &data[n2..];
                let s_right = &svals[n2..];
                let l = Self::do_contract_multi(d_left, s_left);
                let r = Self::do_contract_multi(d_right, s_right);
                Self::do_contract3(&l, &svals[n2 - 1], &r)
            },
        }
    }

    // contract the `k`-th and `k + 1`-th Γ tensors with neighboring singular
    // values if applicable
    //
    // the result is returned as a 2D array in which the left and right physical
    // indices are fused with the outermost indices in the contraction; i.e.
    // for axis signatures
    //      l :: { v, sl, u }
    //      s :: { u }
    //      r :: { u, sr, w }
    // the result has axis signature
    //      q :: { u<>sl, sr<>w }
    // (where `<>` denotes fusion)
    //
    // assumes `k` and `k + 1` are in bounds
    fn contract_local2(&self, k: usize) -> nd::Array2<A> {
        let shk = self.data[k].dim();
        let shkp1 = self.data[k + 1].dim();
        let mut q = Self::do_contract3(
            &self.data[k], &self.svals[k], &self.data[k + 1]);
        if k != 0 {
            nd::Zip::from(q.outer_iter_mut())
                .and(&self.svals[k - 1])
                .for_each(|mut qv__, lkm1v| {
                    let lkm1v = A::from_re(*lkm1v);
                    qv__.map_inplace(|qvsw| { *qvsw *= lkm1v; });
                });
        }
        if k != self.n - 2 {
            nd::Zip::from(q.axis_iter_mut(nd::Axis(2)))
                .and(&self.svals[k + 1])
                .for_each(|mut q__w, lkp1w| {
                    let lkp1w = A::from_re(*lkp1w);
                    q__w.map_inplace(|qvsw| { *qvsw *= lkp1w; });
                });
        }
        q.into_shape((shk.0 * shk.1, shkp1.1 * shkp1.2)).unwrap()
    }

    // like `contract_local2`, but with the return value as a 3D array with
    // physical indices fused in axis 1; i.e.
    // for axis signatures
    //      l :: { v, sl, u }
    //      s :: { u }
    //      r :: { u, sr, w }
    // the result has axis signature
    //      q :: { u, sl<>sr, w }
    // (where `<>` denotes fusion)
    //
    // assumes `k` and `k + 1` are in bounds
    fn contract_local3(&self, k: usize) -> nd::Array3<A> {
        let shk = self.data[k].dim();
        let shkp1 = self.data[k + 1].dim();
        self.contract_local2(k)
            .into_shape((shk.0, shk.1 * shkp1.1, shkp1.2))
            .unwrap()
    }

    // like `contract_local2`, but with the return value as a 4D array with
    // all indices unfused; i.e.
    // for axis signatures
    //      l :: { v, sl, u }
    //      s :: { u }
    //      r :: { u, sr, w }
    // the result has axis signature
    //      q :: { u, sl, sr, w }
    //
    // assumes `k` and `k + 1` are in bounds
    fn contract_local4(&self, k: usize) -> nd::Array4<A> {
        let shk = self.data[k].dim();
        let shkp1 = self.data[k + 1].dim();
        self.contract_local2(k)
            .into_shape((shk.0, shk.1, shkp1.1, shkp1.2))
            .unwrap()
    }

    /// Return the contraction of `self` into a flat state vector.
    pub fn contract(&self) -> nd::Array1<A> {
        let res = Self::do_contract_multi(&self.data, &self.svals);
        let statelen = res.shape()[1];
        res.into_shape(statelen).unwrap()
    }

    /// Contract `self` into a flat state vector and all physical indices.
    pub fn into_contract(self) -> (Vec<T>, nd::Array1<A>) {
        let state = self.contract();
        (self.outs, state)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar
{
    /// Return the contraction of `self` into a rank-N tensor.
    pub fn contract_nd(&self) -> nd::ArrayD<A> {
        let state = self.contract();
        let shape: Vec<usize> = self.outs.iter().map(Idx::dim).collect();
        state.into_shape(shape).unwrap()
    }

    /// Contract `self` into an rank-N tensor and all physical indices.
    pub fn into_contract_nd(self) -> (Vec<T>, nd::ArrayD<A>) {
        let (indices, state) = self.into_contract();
        let shape: Vec<usize> = indices.iter().map(Idx::dim).collect();
        (indices, state.into_shape(shape).unwrap())
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
    nd::Array2<A>: SchmidtDecomp<A>,
{
    // perform a contraction of the total state and repeat the initial
    // factorization process to re-construct all Γ tensors and singular values
    fn refactor(&mut self) {
        let q = self.contract();
        let (data_new, svals_new) = Self::factorize(&self.outs, q, self.trunc);
        self.data = data_new;
        self.svals = svals_new;
    }

    // contract the `k`-th and `k + 1`-th Γ tensors (with neighboring singular
    // values if applicable) and refactor to re-construct for only those sites
    //
    // assumes `k` and `k + 1` are in bounds
    fn local_refactor(&mut self, k: usize) {
        let shk = self.data[k].dim();
        let shkp1 = self.data[k + 1].dim();
        let Schmidt { u, s, q, rank } =
            self.contract_local2(k).local_decomp(self.trunc);
        let mut gk_new = u.into_shape((shk.0, shk.1, rank)).unwrap();
        if k != 0 {
            nd::Zip::from(gk_new.outer_iter_mut())
                .and(&self.svals[k - 1])
                .for_each(|mut gkv__, lkm1v| {
                    let lkm1v = A::from_re(*lkm1v);
                    gkv__.map_inplace(|gkvsw| { *gkvsw /= lkm1v; });
                });
        }
        let mut gkp1_new = q.into_shape((rank, shkp1.1, shkp1.2)).unwrap();
        nd::Zip::from(gkp1_new.axis_iter_mut(nd::Axis(0)))
            .and(&s)
            .for_each(|mut gkp1v, sv| {
                let sv = A::from_re(*sv);
                gkp1v.map_inplace(|gkp1vj| { *gkp1vj /= sv; });
            });
        if k != self.n - 2 {
            nd::Zip::from(gkp1_new.axis_iter_mut(nd::Axis(2)))
                .and(&self.svals[k + 1])
                .for_each(|mut gkp1__w, lkp1w| {
                    let lkp1w = A::from_re(*lkp1w);
                    gkp1__w.map_inplace(|gkp1vsw| { *gkp1vsw /= lkp1w; });
                });
        }
        self.data[k] = gk_new;
        self.svals[k] = s;
        self.data[k + 1] = gkp1_new;
        self.local_renormalize(k);
        self.local_renormalize(k + 1);
    }

    // apply local refactors to each bond in the MPS, going left to right
    fn refactor_lrsweep(&mut self) {
        for k in 0..self.n - 1 { self.local_refactor(k); }
    }

    // apply local refactors to each bond in the MPS, going right to left
    fn refactor_rlsweep(&mut self) {
        for k in (0..self.n - 1).rev() { self.local_refactor(k); }
    }

    // apply a full bidirectional sweep of the MPS, avoiding the double-refactor
    // of the last bond
    fn refactor_sweep(&mut self) {
        for k in  0..self.n - 1        { self.local_refactor(k); }
        for k in (0..self.n - 2).rev() { self.local_refactor(k); }
    }

    // apply two bidirectional sweeps of the MPS starting at a particular site,
    // avoiding double-refactors at the edges
    // assumes `c` is in bounds
    fn refactor_sweep_centered(&mut self, c: usize) {
        if c == 0 {
            for k in  0..self.n - 1        { self.local_refactor(k); }
            for k in (0..self.n - 2).rev() { self.local_refactor(k); }
        } else if c == self.n - 1 {
            for k in (0..self.n - 2).rev() { self.local_refactor(k); }
            for k in  0..self.n - 1        { self.local_refactor(k); }
        } else {
            for k in  c..self.n - 1        { self.local_refactor(k); }
            for k in (0..self.n - 2).rev() { self.local_refactor(k); }
            for k in  0..c                 { self.local_refactor(k); }
        }
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
{
    /// Evaluate the expectation value of a local operator acting (only) on the
    /// `k`-th particle.
    ///
    /// Returns zero if `k` is out of bounds. The arrangement of the elements of
    /// `op` should correspond to the usual left-matrix-multiplication view of
    /// operator application.
    ///
    /// Fails if `op` is not square with a dimension equal to that of the `k`-th
    /// index.
    pub fn expectation_value(&self, k: usize, op: &nd::Array2<A>)
        -> MPSResult<A>
    {
        if k >= self.n { return Ok(A::zero()); }
        let dk = self.outs[k].dim();
        if op.shape() != [dk, dk] { return Err(OperatorIncompatibleShape); }
        let gk = &self.data[k];
        // this is pretty ugly, but it's faster than trying to use matmul...
        if k == 0 {
            // op_{ts} * conj(Γ[k]_{0tw}) * Γ[k]_{0sw} * Λ[k]_{w}^2
            let gk0__ = gk.slice(nd::s![0, .., ..]);
            let lk = &self.svals[k];
            let ev: A =
                nd::Zip::from(gk0__.outer_iter())
                .and(op.outer_iter())
                .fold(A::zero(), |acc, gk0t_, opt_| {
                    nd::Zip::from(gk0__.outer_iter())
                    .and(opt_)
                    .fold(A::zero(), |acc, gk0s_, opts| {
                        nd::Zip::from(gk0t_)
                        .and(gk0s_)
                        .and(lk)
                        .fold(A::zero(), |acc, gk0tw, gk0sw, lkw| {
                            gk0tw.conj() * *gk0sw
                                * A::from_re(lkw.powi(2))
                            + acc
                        })
                        * *opts
                        + acc
                    })
                    + acc
                });
            Ok(ev)
        } else if k == self.n - 1 {
            // Λ[k-1]_{v}^2 * op_{ts} * conj(Γ[k]_{vt0}) * Γ[k]_{vs0}
            let gk__0 = gk.slice(nd::s![.., .., 0]);
            let lkm1 = &self.svals[k - 1];
            let ev: A =
                nd::Zip::from(gk__0.outer_iter())
                .and(gk__0.outer_iter())
                .and(lkm1)
                .fold(A::zero(), |acc, gkv_0p, gkv_0, lkm1v| {
                    nd::Zip::from(gkv_0p)
                    .and(op.outer_iter())
                    .fold(A::zero(), |acc, gkvt0, opt_| {
                        nd::Zip::from(gkv_0)
                        .and(opt_)
                        .fold(A::zero(), |acc, gkvs0, opts| {
                            A::from_re(lkm1v.powi(2))
                                * gkvt0.conj() * *gkvs0
                                * *opts
                            + acc
                        })
                        + acc
                    })
                    * A::from_re(lkm1v.powi(2))
                    + acc
                });
            Ok(ev)
        } else {
            // Λ[k-1]_{v}^2 * op_{ts} * conj(Γ[k]_{vtw}) * Γ[k]_{vsw} * Λ[k]_{w}^2
            let lkm1 = &self.svals[k - 1];
            let lk = &self.svals[k];
            let ev: A =
                nd::Zip::from(gk.outer_iter())
                .and(gk.outer_iter())
                .and(lkm1)
                .fold(A::zero(), |acc, gkv__p, gkv__, lkm1v| {
                    nd::Zip::from(gkv__p.outer_iter())
                    .and(op.outer_iter())
                    .fold(A::zero(), |acc, gkvt_, opt_| {
                        nd::Zip::from(gkv__.outer_iter())
                        .and(opt_)
                        .fold(A::zero(), |acc, gkvs_, opts| {
                            nd::Zip::from(gkvt_)
                            .and(gkvs_)
                            .and(lk)
                            .fold(A::zero(), |acc, gkvtw, gkvsw, lkw| {
                                gkvtw.conj() * *gkvsw
                                    * A::from_re(lkw.powi(2))
                                + acc
                            })
                            * *opts
                            + acc
                        })
                        + acc
                    })
                    * A::from_re(lkm1v.powi(2))
                    + acc
                });
            Ok(ev)
        }
    }

    // calculates the probability of the `k`-th particle being in the `p`-th
    // state assume `k` and `p` are in bounds
    fn local_prob(&self, k: usize, p: usize) -> A::Re {
        if k == 0 {
            let gk0p_ = &self.data[k].slice(nd::s![0, p, ..]);
            let lk = &self.svals[k];
            nd::Zip::from(gk0p_)
                .and(lk)
                .fold(A::Re::zero(), |acc, gk0pw, lkw| {
                    (gk0pw.conj() * *gk0pw).re()
                        * lkw.powi(2)
                    + acc
                })
        } else if k == self.n - 1 {
            let lkm1 = &self.svals[k - 1];
            let gk_p0 = &self.data[k].slice(nd::s![.., p, 0]);
            nd::Zip::from(gk_p0)
                .and(lkm1)
                .fold(A::Re::zero(), |acc, gkvp0, lkm1v| {
                    (gkvp0.conj() * *gkvp0).re()
                        * lkm1v.powi(2)
                    + acc
                })
        } else {
            let lkm1 = &self.svals[k - 1];
            let gk_p_ = &self.data[k].slice(nd::s![.., p, ..]);
            let lk = &self.svals[k];
            nd::Zip::from(gk_p_.outer_iter())
                .and(lkm1)
                .fold(A::Re::zero(), |acc, gkvp_, lkm1v| {
                    nd::Zip::from(gkvp_)
                        .and(lk)
                        .fold(A::Re::zero(), |acc, gkvpw, lkw| {
                            (gkvpw.conj() * *gkvpw).re()
                                * lkw.powi(2)
                            + acc
                        })
                        * lkm1v.powi(2)
                    + acc
                })
        }
    }

    /// Return the probability of the `k`-th particle being in its `p`-th state.
    ///
    /// Returns `None` if `k` or `p` is out of bounds.
    pub fn prob(&self, k: usize, p: usize) -> Option<A::Re> {
        if self.outs.get(k).is_none_or(|idx| p >= idx.dim()) { return None; }
        Some(self.local_prob(k, p))
    }

    // calculates the probabilities for all available quantum numbers of the
    // `k`-th particle
    // assume `k` is in bounds
    fn local_probs(&self, k: usize) -> Vec<A::Re> {
        if self.n == 1 {
            self.data[k].iter()
                .map(|g| (g.conj() * *g).re())
                .collect()
        } else if k == 0 {
            let gk0__ = self.data[k].slice(nd::s![0, .., ..]);
            let lk = self.svals[k].view();
            gk0__.outer_iter()
                .map(|gk0s_| {
                    nd::Zip::from(gk0s_)
                    .and(lk)
                    .fold(A::Re::zero(), |acc, gk0su, lku| {
                        (gk0su.conj() * *gk0su).re()
                            * lku.powi(2)
                        + acc
                    })
                })
                .collect()
        } else if k == self.n - 1 {
            let lkm1 = self.svals[k - 1].view();
            let gk__0 = self.data[k].slice(nd::s![.., .., 0]);
            gk__0.axis_iter(nd::Axis(1))
                .map(|gk_s0| {
                    nd::Zip::from(gk_s0)
                    .and(lkm1)
                    .fold(A::Re::zero(), |acc, gkus0, lkm1u| {
                        (gkus0.conj() * *gkus0).re()
                            * lkm1u.powi(2)
                        + acc
                    })
                })
                .collect()
        } else {
            let lkm1 = self.svals[k - 1].view();
            let gk = &self.data[k];
            let lk = self.svals[k].view();
            gk.axis_iter(nd::Axis(1))
                .map(|gk_s_| {
                    nd::Zip::from(gk_s_.outer_iter())
                    .and(lkm1)
                    .fold(A::Re::zero(), |acc, gkvs_, lkm1v| {
                        nd::Zip::from(gkvs_)
                        .and(lk)
                        .fold(A::Re::zero(), |acc, gkvsu, lku| {
                            (gkvsu.conj() * *gkvsu).re()
                                * lku.powi(2)
                            + acc
                        })
                        * lkm1v.powi(2)
                        + acc
                    })
                })
                .collect()
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

    // calculates the norm of the subspace belonging to particle `k`
    // assume `k` is in bounds
    fn local_norm(&self, k: usize) -> A {
        if k == 0 {
            let gk0__ = &self.data[k].slice(nd::s![0, .., ..]);
            let lk = &self.svals[k];
            ComplexFloat::sqrt(
                gk0__.outer_iter()
                    .map(|gk0s_| {
                        nd::Zip::from(gk0s_)
                        .and(lk)
                        .fold(A::zero(), |acc, gk0sw, lkw| {
                            gk0sw.conj() * *gk0sw
                                * A::from_re(lkw.powi(2))
                            + acc
                        })
                    })
                    .fold(A::zero(), A::add)
            )
        } else if k == self.n - 1 {
            let lkm1 = &self.svals[k - 1];
            let gk__0 = &self.data[k].slice(nd::s![.., .., 0]);
            ComplexFloat::sqrt(
                nd::Zip::from(gk__0.outer_iter())
                    .and(lkm1)
                    .fold(A::zero(), |acc, gkv_0, lkm1v| {
                        gkv_0.iter()
                        .map(|gkvs0| gkvs0.conj() * *gkvs0)
                        .fold(A::zero(), A::add)
                        * A::from_re(lkm1v.powi(2))
                        + acc
                    })
            )
        } else {
            let lkm1 = &self.svals[k - 1];
            let gk = &self.data[k];
            let lk = &self.svals[k];
            ComplexFloat::sqrt(
                nd::Zip::from(gk.outer_iter())
                    .and(lkm1)
                    .fold(A::zero(), |acc, gkv__, lkm1v| {
                        gkv__.outer_iter()
                        .map(|gkvs_| {
                            nd::Zip::from(gkvs_)
                            .and(lk)
                            .fold(A::zero(), |acc, gkvsw, lkw| {
                                gkvsw.conj() * *gkvsw
                                    * A::from_re(lkw.powi(2))
                                + acc
                            })
                        })
                        .fold(A::zero(), A::add)
                        * A::from_re(lkm1v.powi(2))
                        + acc
                    })
            )
        }
    }

    // renormalizes the Γ tensor belonging to particle `k`
    // assumes `k` is in bounds
    fn local_renormalize(&mut self, k: usize) {
        let norm = self.local_norm(k);
        self.data[k].map_inplace(|gkvsu| { *gkvsu /= norm; });
    }
}

impl<T, A> MPS<T, A>
where A: ComplexLinalgScalar
{
    /// Return the current bond dimension truncation setting.
    pub fn get_trunc(&self) -> Option<BondDim<A::Re>> { self.trunc }

    /// Set a new bond dimension truncation setting.
    pub fn set_trunc(&mut self, trunc: Option<BondDim<A::Re>>) {
        self.trunc = trunc;
    }

    /// Return the maximum bond dimension in the MPS.
    ///
    /// This function will always return `Some` if `self` has at least two
    /// particles.
    pub fn max_bond_dim(&self) -> Option<usize> {
        self.svals.iter().map(|s| s.len()).max()
    }

    /// Return the index of the bond with largest dimension in the MPS.
    ///
    /// This function will always return `Some` if `self` has at least two
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

    /// Compute the Von Neumann entropy for a bipartition placed on the `b`-th
    /// bond.
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

    /// Compute the `a`-th Rényi entropy for a bipartition placed on the `b`-th
    /// bond in the Schmidt basis.
    ///
    /// Returns the Von Neumann entropy for `a == 1` and `None` if `b` is out of
    /// bounds.
    pub fn entropy_ry_schmidt(&self, a: A::Re, b: usize) -> Option<A::Re> {
        let one = A::Re::one();
        if a == one {
            self.entropy_vn(b)
        } else {
            self.svals.get(b)
                .map(|s| {
                    s.iter().copied()
                    .map(|sk| (sk * sk).powf(a))
                    .fold(A::Re::zero(), A::Re::add)
                    .ln()
                    * (one - a).recip()
                })
        }
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
{
    /// Apply a unitary transformation to the `k`-th particle.
    ///
    /// Does nothing if `k` is out of bounds. The arrangemet of the elements of
    /// `op` should correspond to the usual left-matrix-multiplication view of
    /// operator application. `op` is not checked for unitarity.
    ///
    /// Fails if `op` is not square with a dimension equal to that of the `k`-th
    /// physical index.
    pub fn apply_unitary1(&mut self, k: usize, op: &nd::Array2<A>)
        -> MPSResult<&mut Self>
    {
        if k >= self.n { return Ok(self); }
        let dk = self.outs[k].dim();
        if op.shape() != [dk, dk] { return Err(OperatorIncompatibleShape); }
        if k == self.n - 1 {
            assert_eq!(self.data[k].shape()[2], 1);
            let gmat: nd::ArrayView2<A> = self.data[k].slice(nd::s![.., .., 0]);
            let dot = gmat.dot(&op.t());
            self.data[k].slice_mut(nd::s![.., .., 0]).assign(&dot);
        } else {
            self.data[k].outer_iter_mut()
                .for_each(|mut gv| { gv.assign(&op.dot(&gv)); });
        }
        Ok(self)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
    nd::Array2<A>: SchmidtDecomp<A>,
{
    /// Apply a unitary operator to the `k`-th and `k + 1`-th particles.
    ///
    /// Does nothing if either `k` or `k + 1` are out of bounds. The arrangement
    /// of the elements of `op` should correrspond to the usual
    /// left-matrix-multiplication view of operator application. `op` is not
    /// checked for unitarity.
    ///
    /// Fails if `op` is not square with a size equal to the product of the
    /// dimensions of the `k`-th and `k + 1`-th physical indices.
    pub fn apply_unitary2(&mut self, k: usize, op: &nd::Array2<A>)
        -> MPSResult<&mut Self>
    {
        if self.n == 1 || k >= self.n - 1 { return Ok(self); }

        let shk = self.data[k].dim();
        let shkp1 = self.data[k + 1].dim();
        if op.shape() != [shk.1 * shkp1.1, shk.1 * shkp1.1] {
            return Err(OperatorIncompatibleShape);
        }

        let mut q = self.contract_local3(k);
        if k == self.n - 2 {
            assert_eq!(q.shape()[2], 1);
            let qmat: nd::ArrayView2<A> = q.slice(nd::s![.., .., 0]);
            let dot = qmat.dot(&op.t());
            q.slice_mut(nd::s![.., .., 0]).assign(&dot);
        } else {
            q.outer_iter_mut()
                .for_each(|mut qv| { qv.assign(&op.dot(&qv)); });
        }

        let Schmidt { u, s, q, rank } =
            q.into_shape((shk.0 * shk.1, shkp1.1 * shkp1.2)).unwrap()
            .local_decomp(self.trunc);
        let mut gk_new = u.into_shape((shk.0, shk.1, rank)).unwrap();
        if k != 0 {
            nd::Zip::from(gk_new.outer_iter_mut())
                .and(&self.svals[k - 1])
                .for_each(|mut gkv__, lkm1v| {
                    let lkm1v = A::from_re(*lkm1v);
                    gkv__.map_inplace(|gkvsw| { *gkvsw /= lkm1v; });
                });
        }
        let mut gkp1_new = q.into_shape((rank, shkp1.1, shkp1.2)).unwrap();
        nd::Zip::from(gkp1_new.axis_iter_mut(nd::Axis(0)))
            .and(&s)
            .for_each(|mut gkp1v, sv| {
                let sv = A::from_re(*sv);
                gkp1v.map_inplace(|gkp1vj| { *gkp1vj /= sv; });
            });
        if k != self.n - 2 {
            nd::Zip::from(gkp1_new.axis_iter_mut(nd::Axis(2)))
                .and(&self.svals[k + 1])
                .for_each(|mut gkp1__w, lkp1w| {
                    let lkp1w = A::from_re(*lkp1w);
                    gkp1__w.map_inplace(|gkp1vsw| { *gkp1vsw /= lkp1w; });
                });
        }
        self.data[k] = gk_new;
        self.svals[k] = s;
        self.data[k + 1] = gkp1_new;
        self.local_renormalize(k);
        self.local_renormalize(k + 1);
        Ok(self)
    }

    /// Like [`apply_unitary2`][Self::apply_unitary2], but for a series of
    /// operators applied to the same two qubits.
    ///
    /// No operation is performed if any operator has invalid shape.
    pub fn apply_unitary2_multi<'a, I>(&mut self, k: usize, ops: I)
        -> MPSResult<&mut Self>
    where I: IntoIterator<Item = &'a nd::Array2<A>>
    {
        if self.n == 1 || k >= self.n - 1 { return Ok(self); }
        let shk = self.data[k].dim();
        let shkp1 = self.data[k + 1].dim();
        let shape_expected = [shk.1 * shkp1.1, shk.1 * shkp1.1];
        let op: nd::Array2<A> =
            ops.into_iter()
            .try_fold(
                nd::Array2::eye(shk.1 * shkp1.1),
                |acc, op| {
                    (op.shape() == shape_expected)
                        .then(|| op.dot(&acc))
                        .ok_or(OperatorIncompatibleShape)
                }
            )?;
        self.apply_unitary2(k, &op)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
    nd::Array2<A>: SchmidtDecomp<A>,
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
        // println!("{:?}", probs);
        // let probsum: A::Re =
        //     probs.iter().copied().fold(A::Re::zero(), A::Re::add);
        // let ten: A::Re =
        //     A::Re::one() + A::Re::one() + A::Re::one() + A::Re::one()
        //     + A::Re::one() + A::Re::one() + A::Re::one() + A::Re::one()
        //     + A::Re::one() + A::Re::one();
        // assert!((probsum - A::Re::one()).abs() < ten.powi(3) * A::Re::epsilon());
        let p =
            probs.iter().copied()
            .scan(A::Re::zero(), |cu, pr| { *cu += pr; Some(*cu) })
            .enumerate()
            .find_map(|(j, cuprob)| (r < cuprob).then_some(j))
            .unwrap();
        (p, probs[p])
    }

    // project the `k`-th particle into its `p`-th eigenstate with a
    // renormalization factor without affecting any singular values
    //
    // assumes `k` is in bounds
    fn project(&mut self, k: usize, p: usize, renorm: A) {
        let zero = A::zero();
        self.data[k]
            .axis_iter_mut(nd::Axis(1))
            .enumerate()
            .for_each(|(j, mut slicej)| {
                if j == p {
                    slicej.map_inplace(|g| { *g /= renorm; });
                } else {
                    slicej.fill(zero);
                }
            });
    }

    /// Perform a projective measurement on the `k`-th particle, reporting the
    /// index value of the (randomized) outcome state for that particle.
    ///
    /// If `k` is out of bounds, do nothing and return `None`.
    // ///
    // /// **Note**: This operation requires contracting the state and then
    // /// performing a refactor on each bond in the MPS after applying the
    // /// projective measurement. If multiple measurements are required, consider
    // /// using [`Self::measure_multi`], which only contracts and refactors bonds
    // /// once.
    pub fn measure<R>(&mut self, k: usize, rng: &mut R) -> Option<usize>
    where R: Rng + ?Sized
    {
        if k >= self.n { return None; }
        let (p, prob) = self.sample_state(k, rng);
        let renorm = A::from_re(prob.sqrt());
        self.project(k, p, renorm);
        // self.refactor_lrsweep();
        self.refactor_sweep();
        // self.refactor_sweep_centered(k);
        // self.refactor();
        Some(p)
    }

    /// Perform a projective measurement on the `k`-th particle, post-selected
    /// to a particular outcome.
    ///
    /// If `k` is out of bounds or `p` is an invalid quantum number, do nothing.
    pub fn measure_postsel(&mut self, k: usize, p: usize) {
        if k >= self.n || p >= self.outs[k].dim() { return; }
        let renorm = A::from_re(Float::sqrt(self.local_prob(k, p)));
        self.project(k, p, renorm);
        self.refactor_sweep();
    }

    // /// Perform a batched series of projective measurements on selected
    // /// particles, reporting the index values of each (randomized) outcome
    // /// states for those particles. Projections are performed and outcomes
    // /// reported in the order the particle indices are seen.
    // ///
    // /// If any particle index is out of bounds, no projection is performed and
    // /// its outcome is omitted.
    // pub fn measure_multi<I, R>(&mut self, particles: I, rng: &mut R)
    //     -> Vec<usize>
    // where
    //     I: IntoIterator<Item = usize>,
    //     R: Rng + ?Sized,
    // {
    //     let n = self.n;
    //     let mut particles: Vec<usize>
    //         = particles.into_iter().filter(|k| *k < n).collect();
    //     if particles.is_empty() { return particles; }
    //
    //     let re_zero = <A as ComplexFloat>::Real::zero();
    //     let mut probs: nd::ArrayD<<A as ComplexFloat>::Real>
    //         = self.contract_nd()
    //         .mapv(|a| (a * a.conj()).re());
    //     particles.iter_mut()
    //         .for_each(|k| {
    //             let r: <A as ComplexFloat>::Real = rng.gen();
    //             let (p, prob)
    //                 = probs.axis_iter(nd::Axis(*k))
    //                 .map(|slice| slice.sum())
    //                 .scan(re_zero, |cu, pr| { *cu = *cu + pr; Some((*cu, pr)) })
    //                 .enumerate()
    //                 .find_map(|(j, (cu, pr))| (r < cu).then_some((j, pr)))
    //                 .unwrap();
    //             let renorm = A::from_re(Float::sqrt(prob));
    //
    //             self.project(*k, p, renorm);
    //             probs.axis_iter_mut(nd::Axis(*k))
    //                 .enumerate()
    //                 .for_each(|(j, mut slice)| {
    //                     if j == p {
    //                         slice.map_inplace(|pr| { *pr = *pr / prob; });
    //                     } else {
    //                         slice.fill(re_zero);
    //                     }
    //                 });
    //             *k = p;
    //         });
    //     self.refactor_sweep();
    //     // self.refactor_lrsweep();
    //     // self.refactor_rlsweep();
    //     particles
    // }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
{
    /// Convert to an unevaluated [`Network`].
    ///
    /// All physical indices must be unique.
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
                .for_each(|(dj, skj)| { *dj = A::from_re(skj); });
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
    pub fn into_network_density(self) -> Network<MPSMatIndex<T>, A> {
        unsafe {
            let mut net =
                self.into_network()
                .map_indices(MPSIndex::into_mpsmat)
                .unwrap();
            let conj_nodes: Vec<_> =
                net.nodes()
                .map(|(_, tens)| tens.conj().map_indices(MPSMatIndex::conj))
                .collect();
            conj_nodes.into_iter()
                .for_each(|tens| { net.push(tens).unwrap(); });
            net
        }
    }

    /// Convert to an unevaluated [`Network`] representing a contiguous subset
    /// of particles, with the remainder traced over.
    pub fn into_network_part(self, mut part: Range<usize>)
        -> Network<MPSMatIndex<T>, A>
    {
        let Self { data, outs, svals, n, .. } = self;
        part.end = part.end.min(n);
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
            if part.contains(&k) && part.contains(&(k + 1)) {
                let d = sk.len();
                let mut data = nd::Array::zeros((d, d));
                data.diag_mut().iter_mut()
                    .zip(sk)
                    .for_each(|(dj, skj)| { *dj = A::from_re(*skj); });
                let u = MPSMatIndex::BondLKet { label: k, dim: d };
                let v = MPSMatIndex::BondRKet { label: k, dim: d };
                let data = unsafe {
                    Tensor::from_array_unchecked([u, v], data)
                };
                network.push(data).unwrap();
            }
        }
        unsafe {
            let conj_nodes: Vec<_> =
                network.nodes()
                .map(|(_, tens)| tens.conj().map_indices(MPSMatIndex::conj))
                .collect();
            conj_nodes.into_iter()
                .for_each(|tens| { network.push(tens).unwrap(); });
        }
        if part.start != 0 {
            let k = part.start - 1;
            let d = svals[k].len();
            let mut data = nd::Array::zeros((d, d));
            data.diag_mut().iter_mut()
                .zip(&svals[k])
                .for_each(|(dj, skj)| { *dj = A::from_re(*skj * *skj); });
            let uket = MPSMatIndex::BondRKet { label: k, dim: d };
            let ubra = MPSMatIndex::BondRBra { label: k, dim: d };
            let tracer = unsafe {
                Tensor::from_array_unchecked([uket, ubra], data)
            };
            network.push(tracer).unwrap();
        }
        if part.end != n {
            let k = part.end - 1;
            let d = svals[k].len();
            let mut data = nd::Array::zeros((d, d));
            data.diag_mut().iter_mut()
                .zip(&svals[k])
                .for_each(|(dj, skj)| { *dj = A::from_re(*skj * *skj); });
            let vket = MPSMatIndex::BondLKet { label: k, dim: d };
            let vbra = MPSMatIndex::BondLBra { label: k, dim: d };
            let tracer = unsafe {
                Tensor::from_array_unchecked([vket, vbra], data)
            };
            network.push(tracer).unwrap();
        }
        network
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar,
{
    /// Contract the MPS and convert to a single [`Tensor`] representing the
    /// pure state.
    pub fn into_tensor(self) -> Tensor<T, A> {
        let tens =
            self.into_network()
            .contract()
            .unwrap();
        unsafe { tens.map_indices(MPSIndex::into_physical) }
    }

    /// Contract the MPS and convert to a single [`Tensor`] representing the
    /// pure state as a density matrix.
    pub fn into_tensor_density(self) -> Tensor<MatIndex<T>, A> {
        let dens =
            self.into_network_density()
            .contract()
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Contract the MPS and convert to a single [`Tensor`] representing the
    /// partial trace of the state.
    ///
    /// Particles outside of `part` will be traced over.
    pub fn into_tensor_part(self, part: Range<usize>) -> Tensor<MatIndex<T>, A>
    {
        let dens =
            self.into_network_part(part)
            .contract()
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Contract the MPS into a density matrix and compute the `a`-th Rényi
    /// entropy of a subspace.
    pub fn entropy_ry(&self, a: u32, part: Range<usize>) -> A::Re
    where
        T: Ord,
        nd::Array2<A>:
            Eigh<EigVal = nd::Array1<A::Re>, EigVec = nd::Array2<A>>
            + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
    {
        let mut rho_tens = self.clone().into_tensor_part(part);
        rho_tens.sort_indices();
        let (indices, data) = rho_tens.into_flat();
        let sidelen: usize =
            indices.iter()
            .take(indices.len() / 2)
            .map(|idx| idx.dim())
            .product();
        let rho: nd::Array2<A> =
            data.as_standard_layout()
            .into_shape((sidelen, sidelen))
            .unwrap()
            .into_owned();
        // let rho: nd::Array2<A>
        //     = data.reshape((sidelen, sidelen))
        //     .into_owned();
        entropy_ry(&rho, a)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx + Ord,
    A: ComplexLinalgScalar,
{
    /// Contract the MPS and convert to a single, bare density matrix in the
    /// standard basis.
    pub fn into_matrix(self) -> (Vec<MatIndex<T>>, nd::Array2<A>) {
        let mut tens = self.into_tensor_density();
        tens.sort_indices();
        let (indices, data) = tens.into_flat();
        let sidelen: usize =
            indices.iter()
            .take(indices.len() / 2)
            .map(|idx| idx.dim())
            .product();
        let rho: nd::Array2<A> =
            data.as_standard_layout()
            .into_shape((sidelen, sidelen))
            .unwrap()
            .into_owned();
        (indices, rho)
    }

    /// Contract the MPS and convert to a single, bare density matrix
    /// representing the partial trace of the state in the standard basis.
    pub fn into_matrix_part(self, part: Range<usize>)
        -> (Vec<MatIndex<T>>, nd::Array2<A>)
    {
        let mut tens = self.into_tensor_part(part);
        tens.sort_indices();
        let (indices, data) = tens.into_flat();
        let sidelen: usize =
            indices.iter()
            .take(indices.len() / 2)
            .map(|idx| idx.dim())
            .product();
        let rho: nd::Array2<A> =
            data.as_standard_layout()
            .into_shape((sidelen, sidelen))
            .unwrap()
            .into_owned();
        (indices, rho)
    }
}

impl<T, A> MPS<T, A>
where
    T: Idx + Send + 'static,
    A: ComplexLinalgScalar + Send + Sync,
    // A::Re: std::fmt::Debug,
{
    /// Like [`Self::into_tensor`], but using a [`Pool`] for parallel
    /// contractions.
    pub fn into_tensor_par(self, pool: &Pool<MPSIndex<T>, A>)
        -> Tensor<T, A>
    {
        let tens =
            self.into_network()
            .contract_par(pool)
            .unwrap();
        unsafe { tens.map_indices(MPSIndex::into_physical) }
    }

    /// Like [`Self::into_tensor_density`], but using a [`Pool`] for
    /// parallel contractions.
    pub fn into_tensor_density_par(
        self,
        pool: &Pool<MPSMatIndex<T>, A>,
    ) -> Tensor<MatIndex<T>, A>
    {
        let dens =
            self.into_network_density()
            .contract_par(pool)
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Like [`Self::into_tensor_part`], but using a [`Pool`] for
    /// parallel contractions.
    pub fn into_tensor_part_par(
        self,
        part: Range<usize>,
        pool: &Pool<MPSMatIndex<T>, A>,
    ) -> Tensor<MatIndex<T>, A>
    {
        let dens =
            self.into_network_part(part)
            .contract_par(pool)
            .unwrap();
        unsafe { dens.map_indices(MPSMatIndex::into_physical) }
    }

    /// Like [`Self::entropy_ry`], but using a [`Pool`] for parallel
    /// contractions.
    pub fn entropy_ry_par(
        &self,
        a: u32,
        part: Range<usize>,
        pool: &Pool<MPSMatIndex<T>, A>,
    ) -> A::Re
    where
        T: Ord,
        nd::Array2<A>:
            Eigh<EigVal = nd::Array1<A::Re>, EigVec = nd::Array2<A>>
            + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
    {
        let mut rho_tens = self.clone().into_tensor_part_par(part, pool);
        rho_tens.sort_indices();
        let (indices, data) = rho_tens.into_flat();
        let sidelen: usize =
            indices.iter()
            .map(|idx| idx.dim())
            .product();
        let rho: nd::Array2<A> =
            data.as_standard_layout()
            .into_shape((sidelen, sidelen))
            .unwrap()
            .into_owned();
        // let rho: nd::Array2<A>
        //     = data.reshape((sidelen, sidelen))
        //     .into_owned();
        entropy_ry(&rho, a)
    }
}

impl MPS<Q, C64> {
    /// Create a new `n`-qubit state initialized to all qubits in ∣0⟩.
    ///
    /// Optionally provide a global cutoff threshold for singular values, which
    /// defaults to zero.
    ///
    /// Fails if `n == 0`.
    pub fn new_qubits(
        n: usize,
        trunc: Option<BondDim<f64>>,
    ) -> MPSResult<Self>
    {
        Self::new((0..n).map(Q), trunc)
    }

    /// Perform the action of a gate.
    ///
    /// Does nothing if any qubit indices are out of bounds.
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
            Gate::Haar2(k) if k < self.n - 1 => {
                // TODO: figure out a way to avoid having to call thread_rng
                // here; it's slow
                let mut rng = rand::thread_rng();
                self.apply_unitary2(k, &gate::haar(2, &mut rng))
                    .unwrap();
            },
            _ => { },
        }
        self
    }

    /// Perform a series of gates.
    pub fn apply_circuit<'a, I>(&mut self, gates: I) -> &mut Self
    where I: IntoIterator<Item = &'a Gate>
    {
        gates.into_iter().copied().for_each(|g| { self.apply_gate(g); });
        self
    }

    /// Like [`Self::measure`], but deterministically reset the state to ∣0⟩;
    /// after measurement.
    pub fn measure_reset<R>(&mut self, k: usize, rng: &mut R) -> Option<usize>
    where R: Rng + ?Sized
    {
        let res = self.measure(k, rng);
        if let Some(1) = res {
            self.apply_unitary1(k, Lazy::force(&gate::XMAT))
                .unwrap();
        }
        res
    }

    /// Like [`Self::measure_postsel`], but deterministically reset the state to
    /// ∣0⟩ after measurement.
    pub fn measure_postsel_reset(&mut self, k: usize, p: usize) {
        self.measure_postsel(k, p);
        if p != 0 {
            self.apply_unitary1(k, Lazy::force(&gate::XMAT))
                .unwrap();
        }
    }
}

fn block_indent(tab: &str, level: usize, s: &str) -> String {
    let lines: Vec<String> =
        s.split('\n')
        .filter(|line| !line.trim().is_empty())
        .map(|line| tab.repeat(level) + line)
        .collect();
    lines.join("\n")
}

impl<T, A> fmt::Debug for MPS<T, A>
where
    T: fmt::Debug,
    A: ComplexLinalgScalar + fmt::Debug,
    A::Re: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const TAB: &str = "    ";
        writeln!(f, "MPS {{")?;
        writeln!(f, "    n: {:?},", self.n)?;
        writeln!(f, "    data: [")?;
        for datak in self.data.iter() {
            writeln!(f, "{},", block_indent(TAB, 2, &format!("{datak:?}")))?;
        }
        writeln!(f, "    ],")?;
        writeln!(f, "    outs: {:?},", self.outs)?;
        writeln!(f, "    svals: [")?;
        for svalsk in self.svals.iter() {
            writeln!(f, "        {:?},", svalsk)?;
        }
        writeln!(f, "    ],")?;
        writeln!(f, "    trunc: {:?}", self.trunc)?;
        write!(f, "}}")?;
        Ok(())
    }
}

fn fmt_array3<S, A>(
    arr: &nd::ArrayBase<S, nd::Ix3>,
    indent: usize,
    indent_str: &str,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result
where
    S: nd::Data<Elem = A>,
    A: fmt::Display,
{
    let sh = arr.dim();
    let tab = indent_str.repeat(indent);
    write!(f, "{}", tab)?;
    write!(f, "[")?;
    for (i, arri) in arr.outer_iter().enumerate() {
        if i != 0 { write!(f, "{} ", tab)?; }
        write!(f, "[")?;
        for (j, arrij) in arri.outer_iter().enumerate() {
            if j != 0 { write!(f, "{}  ", tab)?; }
            write!(f, "[")?;
            for (k, arrijk) in arrij.iter().enumerate() {
                arrijk.fmt(f)?;
                if k < sh.2 - 1 { write!(f, ", ")?; }
            }
            write!(f, "]")?;
            if j < sh.1 - 1 { writeln!(f, ",")?; }
        }
        write!(f, "]")?;
        if i < sh.0 - 1 { writeln!(f, ",")?; }
    }
    write!(f, "]")?;
    Ok(())
}

impl<T, A> fmt::Display for MPS<T, A>
where
    T: Idx,
    A: ComplexLinalgScalar + fmt::Display,
    A::Re: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter =
            self.data.iter().zip(&self.svals).zip(&self.outs).enumerate();
        for (k, ((g, l), out)) in iter {
            let sh = g.shape();
            writeln!(f, "Γ[{}] :: {{ <{}>, {}<{}>, <{}> }}",
                k, sh[0], out.label(), sh[1], sh[2]
            )?;
            fmt_array3(g, 1, "    ", f)?;
            writeln!(f)?;
            write!(f, "Λ[{}] = ", k)?;
            l.fmt(f)?;
            writeln!(f)?;
        }
        let sh = self.data[self.n - 1].shape();
        writeln!(f, "Γ[{}] :: {{ <{}>, {}<{}>, <{}> }}",
            self.n - 1, sh[0], self.outs[self.n - 1].label(), sh[1], sh[2]
        )?;
        fmt_array3(&self.data[self.n - 1], 1, "    ", f)?;
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
    pub fn is_ket(&self) -> bool { matches!(self, Self::Ket(..)) }

    /// Return `true` if `self` is `Bra`.
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
    pub fn is_lbond(&self) -> bool { matches!(self, Self::BondL { .. }) }

    /// Return `true` if `self` is `BondR`.
    pub fn is_rbond(&self) -> bool { matches!(self, Self::BondR { .. }) }

    /// Return `true` if `self` is `Physical`.
    pub fn is_physical(&self) -> bool { matches!(self, Self::Physical(..)) }

    /// Unwrap `self` into a bare physical index.
    ///
    /// *Panics* if `self` is `BondL` or `BondR`.
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
    pub fn is_lbondket(&self) -> bool { matches!(self, Self::BondLKet { .. }) }

    /// Return `true` if `self` is `BondLBra`.
    pub fn is_lbondbra(&self) -> bool { matches!(self, Self::BondLBra { .. }) }

    /// Return `true` if `self` is `BondLKet` or `BondLBra`.
    pub fn is_lbond(&self) -> bool {
        matches!(self, Self::BondLKet { .. } | Self::BondLBra { .. })
    }

    /// Return `true` if `self` is `BondRKet`.
    pub fn is_rbondket(&self) -> bool { matches!(self, Self::BondRKet { .. }) }

    /// Return `true` if `self` is `BondRBra`.
    pub fn is_rbondbra(&self) -> bool { matches!(self, Self::BondRBra { .. }) }

    /// Return `true` if `self` is `BondRKet` or `BondRBra`.
    pub fn is_rbond(&self) -> bool {
        matches!(self, Self::BondRKet { .. } | Self::BondRBra { .. })
    }

    /// Return `true` if `self` is `PhysicalKet`.
    pub fn is_physicalket(&self) -> bool { matches!(self, Self::PhysicalKet(..)) }

    /// Return `true` if `self` is `PhysicalBra`.
    pub fn is_physicalbra(&self) -> bool { matches!(self, Self::PhysicalBra(..)) }

    /// Return `true` if `self` is `PhysicalKet` or `PhysicalBra`.
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::PhysicalKet(..) | Self::PhysicalBra(..))
    }

    /// Return `true` if `self` is `BondLKet`, `BondRKet`, or `PhysicalKet`.
    pub fn is_ket(&self) -> bool {
        matches!(
            self,
            Self::BondLKet { .. }
            | Self::BondRKet { .. }
            | Self::PhysicalKet(..)
        )
    }

    /// Return `true` if `self` is `BondLBra`, `BondRBra`, or `PhysicalBra`.
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

    fn from_mps(idx: MPSIndex<T>) -> Self {
        match idx {
            MPSIndex::BondL { label, dim } => Self::BondLKet { label, dim },
            MPSIndex::BondR { label, dim } => Self::BondRKet { label, dim },
            MPSIndex::Physical(idx) => Self::PhysicalKet(idx),
        }
    }

    /// Flip the bra/ket-ness of `self`.
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
    A: ComplexLinalgScalar,
    // A::Re: std::fmt::Debug,
    nd::Array2<A>: Eigh<EigVal = nd::Array1<A::Re>, EigVec = nd::Array2<A>>,
{
    let (e, v) = x.eigh(UPLO::Upper).expect("mat_ln: error in diagonalization");
    let u = v.t().mapv(|a| a.conj());
    let log_e = nd::Array2::from_diag(
        &e.mapv(|ek| A::from_re(ek.ln()))
    );
    v.dot(&log_e).dot(&u)
}

// compute an integer power of a matrix
// panics if the matrix is not square
// expecting most cases to use n < 5
pub(crate) fn mat_pow<A>(x: &nd::Array2<A>, n: u32) -> nd::Array2<A>
where
    A: ComplexLinalgScalar,
    // A::Re: std::fmt::Debug,
    nd::Array2<A>: nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
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
pub fn entropy_vn<A>(rho: &nd::Array2<A>) -> A::Re
where
    A: ComplexLinalgScalar,
    // A::Re: std::fmt::Debug,
    nd::Array2<A>:
        Eigh<EigVal = nd::Array1<A::Re>, EigVec = nd::Array2<A>>
        + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
{
    rho.dot(&mat_ln(rho))
        .diag().iter().copied()
        .map(ComplexFloat::re)
        .fold(A::Re::zero(), A::Re::add)
        * (-A::Re::one())
}

/// Compute the *n*-th Rényi entropy of a density matrix.
///
/// Returns the [Von Neumann entropy][entropy_vn] if `n == 1`. The input must be
/// a valid density matrix, but can be pure or mixed.
pub fn entropy_ry<A>(rho: &nd::Array2<A>, n: u32) -> A::Re
where
    A: ComplexLinalgScalar,
    // A::Re: std::fmt::Debug,
    nd::Array2<A>:
        Eigh<EigVal = nd::Array1<A::Re>, EigVec = nd::Array2<A>>
        + nd::linalg::Dot<nd::Array2<A>, Output = nd::Array2<A>>,
{
    if n == 1 {
        entropy_vn(rho)
    } else {
        let nfloat: A::Re =
            (0..n)
            .map(|_| A::Re::one())
            .fold(A::Re::zero(), A::Re::add);
        let prefactor = (A::Re::one() - nfloat).recip();
        mat_pow(rho, n)
            .diag()
            .iter()
            .copied()
            .map(ComplexFloat::re)
            .fold(A::Re::zero(), A::Re::add)
            .ln()
            * prefactor
    }
}


