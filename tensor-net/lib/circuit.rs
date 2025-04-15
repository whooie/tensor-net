//! Abstractions for driving layer-based circuits on qubits.

use std::{
    borrow::Cow,
    io::{ Read, Write },
    path::Path
};
use nalgebra as na;
use num_complex::Complex64 as C64;
use rand::Rng;
use serde::{ Serialize, Deserialize };
use thiserror::Error;
use crate::{
    gate::{ Clifford, Gate },
    mps::{ MPS, MPSError },
    tensor::Idx,
};

#[derive(Debug, Error)]
pub enum CircuitError {
    /// Returned when a [`MPS`] encounters an error during a circuit-level
    /// operation.
    #[error("MPS error: {0}")]
    MPSError(#[from] MPSError),

    /// Returned when attempting to serialize a piece of data not supported by
    /// the [`postcard`] format.
    #[error("serialization error: encountered unsupported data")]
    SerError,

    /// Returned when either attempting to deserialize malformed binary data
    /// (i.e. not in the [`postcard`] format) or attempting to to deserialize to
    /// the wrong data type.
    #[error("deserialization error: malformed input")]
    DeserError,

    /// General input/output error.
    #[error("IO error: {0}")]
    IOError(std::io::Error),
}
pub type CircuitResult<T> = Result<T, CircuitError>;

/// [Index type][Idx] for qubits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Q(
    /// The qubit index.
    pub usize,
);

impl From<usize> for Q {
    fn from(k: usize) -> Self { Self(k) }
}

impl From<Q> for usize {
    fn from(q: Q) -> Self { q.0 }
}

impl Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String { format!("q{}", self.0) }
}

/// The result of a measurement.
#[derive(
    Copy, Clone,
    Debug,
    PartialEq, Eq,
    Hash,
    PartialOrd, Ord,
    Serialize, Deserialize
)]
pub enum Outcome {
    /// ∣0⟩
    Zero = 0,
    /// ∣1⟩
    One = 1,
}

impl Outcome {
    /// Returns `true` if `self` is `Zero`.
    pub fn is_zero(self) -> bool { matches!(self, Self::Zero) }

    /// Returns `true` if `self` is `One`.
    pub fn is_one(self) -> bool { matches!(self, Self::One) }

    /// The logical negation of `self`.
    pub fn negate(self) -> Self {
        match self {
            Self::Zero => Self::One,
            Self::One => Self::Zero,
        }
    }

    /// The logical conjunction of `self` and `other`.
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::Zero,
            (Self::Zero, Self::One ) => Self::Zero,
            (Self::One,  Self::Zero) => Self::Zero,
            (Self::One,  Self::One ) => Self::One,
        }
    }

    /// The logical disjunction of `self` and `other`.
    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::Zero,
            (Self::Zero, Self::One ) => Self::One,
            (Self::One,  Self::Zero) => Self::One,
            (Self::One,  Self::One ) => Self::One,
        }
    }

    /// The logical material conditional of `self` and `other`.
    pub fn implies(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::One,
            (Self::Zero, Self::One ) => Self::One,
            (Self::One,  Self::Zero) => Self::Zero,
            (Self::One,  Self::One ) => Self::One,
        }
    }

    /// The logical converse conditional of `self` and `other`.
    pub fn converse(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::One,
            (Self::Zero, Self::One ) => Self::Zero,
            (Self::One,  Self::Zero) => Self::One,
            (Self::One,  Self::One ) => Self::One,
        }
    }

    /// The logical Sheffer stroke of `self` and `other`.
    pub fn nand(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::One,
            (Self::Zero, Self::One ) => Self::One,
            (Self::One,  Self::Zero) => Self::One,
            (Self::One,  Self::One ) => Self::Zero,
        }
    }

    /// The logical non-disjunction of `self` and `other`.
    pub fn nor(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::One,
            (Self::Zero, Self::One ) => Self::Zero,
            (Self::One,  Self::Zero) => Self::Zero,
            (Self::One,  Self::One ) => Self::Zero,
        }
    }

    /// The logical biconditional of `self` and `other`.
    pub fn xnor(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::One,
            (Self::Zero, Self::One ) => Self::Zero,
            (Self::One,  Self::Zero) => Self::Zero,
            (Self::One,  Self::One ) => Self::One,
        }
    }
}

macro_rules! conv_outcome_int {
    ($int:ty) => {
        impl From<$int> for Outcome {
            fn from(n: $int) -> Self {
                if n == 0 { Self::Zero } else { Self::One }
            }
        }

        impl From<Outcome> for $int {
            fn from(o: Outcome) -> Self { o as $int }
        }
    }
}
conv_outcome_int!(usize);
conv_outcome_int!(u8);
conv_outcome_int!(u16);
conv_outcome_int!(u32);
conv_outcome_int!(u64);
conv_outcome_int!(u128);
conv_outcome_int!(isize);
conv_outcome_int!(i8);
conv_outcome_int!(i16);
conv_outcome_int!(i32);
conv_outcome_int!(i64);
conv_outcome_int!(i128);

impl From<bool> for Outcome {
    fn from(b: bool) -> Self { if b { Self::One } else { Self::Zero } }
}

impl From<Outcome> for bool {
    fn from(o: Outcome) -> Self {
        match o {
            Outcome::Zero => false,
            Outcome::One => true,
        }
    }
}

/// A single unitary, either a compact [`Gate`] or an explicitly defined matrix.
///
/// **Note:** matrices *must* be either 2×2 or 4×4, and will be *assumed* to be
/// unitary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Uni {
    /// A succinct [`Gate`] description.
    Gate(Gate),
    /// An explicitly represented matrix.
    Mat(usize, na::DMatrix<C64>),
    // /// An explicitly represented matrix for one-qubit states.
    // MatQ1(usize, na::Matrix2<C64>),
    // /// An explicitly represented matrix for two-qubit states.
    // MatQ2(usize, na::Matrix4<C64>),
}

impl Uni {
    /// Return `true` if `self` is `Gate`.
    pub fn is_gate(&self) -> bool { matches!(self, Self::Gate(_)) }

    /// Return `true` if `self` is `Mat`.
    pub fn is_mat(&self) -> bool { matches!(self, Self::Mat(..)) }

    /// Return the (left-most) qubit index.
    pub fn idx(&self) -> usize {
        match self {
            Self::Gate(g) => g.idx(),
            Self::Mat(k, _) => *k,
            // Self::MatQ1(k, _) => *k,
            // Self::MatQ2(k, _) => *k,
        }
    }

    /// Return a mutable reference to the (left-most) qubit index.
    pub fn idx_mut(&mut self) -> &mut usize {
        match self {
            Self::Gate(g) => g.idx_mut(),
            Self::Mat(k, _) => k,
            // Self::MatQ1(k, _) => k,
            // Self::MatQ2(k, _) => k,
        }
    }

    /// Force evaluation of `self` as an explicit matrix and return the result.
    ///
    /// Note that this may require random sampling, for which the local thread
    /// generator will need to be acquired here and then released immediately
    /// afterward. If you are creating matrices for many `Cliff*` or `Haar*`
    /// [`Gate`]s, this will be inefficient and you may which to use
    /// [`make_mat_rng`][Self::make_mat_rng] instead, which takes a cached
    /// generator as argument; this is also useful for fixed-seed applications.
    pub fn make_mat(&mut self) -> (usize, &na::DMatrix<C64>) {
        match self {
            Self::Gate(g) => {
                let (k, cow_mat) = (*g).into_matrix();
                let mat = cow_mat.into_owned();
                *self = Self::Mat(k, mat);
            },
            Self::Mat(..) => { },
            // Self::MatQ1(..) => { },
            // Self::MatQ2(..) => { },
        }
        self.mat().unwrap()
    }

    /// Like [`make_mat`][Self::make_mat], but taking a cached generator.
    pub fn make_mat_rng<R>(&mut self, rng: &mut R) -> (usize, &na::DMatrix<C64>)
    where R: Rng + ?Sized
    {
        match self {
            Self::Gate(g) => {
                let (k, cow_mat) = (*g).into_matrix_rng(rng);
                let mat = cow_mat.into_owned();
                *self = Self::Mat(k, mat);
            },
            Self::Mat(..) => { },
            // Self::MatQ1(..) => { },
            // Self::MatQ2(..) => { },
        }
        self.mat().unwrap()
    }

    /// Return a reference to the underlying [`Gate`] definition, if `self` is
    /// `Gate`.
    pub fn gate(&self) -> Option<&Gate> {
        match self {
            Self::Gate(g) => Some(g),
            Self::Mat(..) => None,
            // Self::MatQ1(..) => None,
            // Self::MatQ2(..) => None,
        }
    }

    /// Return a mutable reference to the underlying [`Gate`] definition, if
    /// `self` is `Gate`.
    pub fn gate_mut(&mut self) -> Option<&mut Gate> {
        match self {
            Self::Gate(g) => Some(g),
            Self::Mat(..) => None,
            // Self::MatQ1(..) => None,
            // Self::MatQ2(..) => None,
        }
    }

    /// Return a reference to the underlying matrix, if `self` is `Mat`.
    pub fn mat(&self) -> Option<(usize, &na::DMatrix<C64>)> {
        match self {
            Self::Gate(_) => None,
            Self::Mat(k, mat) => Some((*k, mat)),
            // Self::MatQ1(..) => todo!(),
            // Self::MatQ2(..) => todo!(),
        }
    }

    /// Return a mutable reference to the underlying matrix, if `self` is `Mat`.
    ///
    /// # Safety
    /// Matrices *must* be either 2×2 or 4×4, and will be *assumed* to be
    /// unitary.
    pub unsafe fn mat_mut(&mut self) -> Option<(usize, &mut na::DMatrix<C64>)> {
        match self {
            Self::Gate(_) => None,
            Self::Mat(k, mat) => Some((*k, mat)),
            // Self::MatQ1(..) => todo!(),
            // Self::MatQ2(..) => todo!(),
        }
    }
}

impl From<&Gate> for Uni {
    fn from(gate: &Gate) -> Self { Self::Gate(*gate) }
}

impl From<Gate> for Uni {
    fn from(gate: Gate) -> Self { Self::Gate(gate) }
}

impl From<(usize, na::DMatrix<C64>)> for Uni {
    fn from(uni: (usize, na::DMatrix<C64>)) -> Self { Self::Mat(uni.0, uni.1) }
}

impl From<(na::DMatrix<C64>, usize)> for Uni {
    fn from(uni: (na::DMatrix<C64>, usize)) -> Self { Self::Mat(uni.1, uni.0) }
}

impl<'a> From<(usize, Cow<'a, na::DMatrix<C64>>)> for Uni {
    fn from(uni: (usize, Cow<'a, na::DMatrix<C64>>)) -> Self {
        Self::Mat(uni.0, uni.1.into_owned())
    }
}

impl<'a> From<(Cow<'a, na::DMatrix<C64>>, usize)> for Uni {
    fn from(uni: (Cow<'a, na::DMatrix<C64>>, usize)) -> Self {
        Self::Mat(uni.1, uni.0.into_owned())
    }
}

/// A single qubit measurement operation, either naive or post-selected
/// (projected).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Meas {
    Rand(usize),
    Proj(usize, Outcome),
}

impl Meas {
    /// Return `true` if `self` is `Rand`.
    pub fn is_rand(&self) -> bool { matches!(self, Self::Rand(_)) }

    /// Return `true` if `self` is `Proj`.
    pub fn is_proj(&self) -> bool { matches!(self, Self::Proj(..)) }

    /// Return the target qubit index.
    pub fn idx(&self) -> usize {
        match self {
            Self::Rand(k) => *k,
            Self::Proj(k, _) => *k,
        }
    }

    /// Return a mutable reference to the target qubit index.
    pub fn idx_mut(&mut self) -> &mut usize {
        match self {
            Self::Rand(k) => k,
            Self::Proj(k, _) => k,
        }
    }

    /// Convert `self` into a post-selected measurement using the given outcome.
    /// If `self` is already `Proj`, the previous outcome is *not* replaced.
    pub fn make_proj(&mut self, outcome: Outcome) {
        match self {
            Self::Rand(k) => {
                *self = Self::Proj(*k, outcome);
            },
            Self::Proj(..) => { },
        }
    }

    /// Force `self` into a post-selected measurement with the given outcome. If
    /// `self` is already `Proj`, the previous outcome is replaced and returned.
    ///
    /// See also [`make_proj`][Self::make_proj], which doesn't replace the
    /// previous outcome.
    pub fn force_proj(&mut self, mut outcome: Outcome) -> Option<Outcome> {
        match self {
            Self::Rand(k) => {
                *self = Self::Proj(*k, outcome);
                None
            },
            Self::Proj(_, prev_out) => {
                std::mem::swap(prev_out, &mut outcome);
                Some(outcome)
            },
        }
    }

    /// Get the post-selected outcome if `self` is `Proj`.
    pub fn outcome(&self) -> Option<Outcome> {
        match self {
            Self::Rand(_) => None,
            Self::Proj(_, outcome) => Some(*outcome),
        }
    }

    /// Get a mutable reference to the post-selected outcome if `self` is
    /// `Proj`.
    pub fn outcome_mut(&mut self) -> Option<&mut Outcome> {
        match self {
            Self::Rand(_) => None,
            Self::Proj(_, outcome) => Some(outcome),
        }
    }
}

impl From<usize> for Meas {
    fn from(idx: usize) -> Self { Self::Rand(idx) }
}

impl From<(usize, Outcome)> for Meas {
    fn from(proj: (usize, Outcome)) -> Self { Self::Proj(proj.0, proj.1) }
}

impl From<(Outcome, usize)> for Meas {
    fn from(proj: (Outcome, usize)) -> Self { Self::Proj(proj.1, proj.0) }
}

/// A single unitary or measurement operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Op {
    Uni(Uni),
    Meas(Meas),
}

impl From<Uni> for Op {
    fn from(uni: Uni) -> Self { Self::Uni(uni) }
}

impl From<&Gate> for Op {
    fn from(gate: &Gate) -> Self { Self::Uni(gate.into()) }
}

impl From<Gate> for Op {
    fn from(gate: Gate) -> Self { Self::Uni(gate.into()) }
}

impl From<(usize, na::DMatrix<C64>)> for Op {
    fn from(uni: (usize, na::DMatrix<C64>)) -> Self { Self::Uni(uni.into()) }
}

impl From<(na::DMatrix<C64>, usize)> for Op {
    fn from(uni: (na::DMatrix<C64>, usize)) -> Self { Self::Uni(uni.into()) }
}

impl From<Meas> for Op {
    fn from(meas: Meas) -> Self { Self::Meas(meas) }
}

impl From<usize> for Op {
    fn from(idx: usize) -> Self { Self::Meas(idx.into()) }
}

impl From<(usize, Outcome)> for Op {
    fn from(proj: (usize, Outcome)) -> Self { Self::Meas(proj.into()) }
}

impl From<(Outcome, usize)> for Op {
    fn from(proj: (Outcome, usize)) -> Self { Self::Meas(proj.into()) }
}

/// A collection of elements in a circuit.
///
/// This is really just a [`Vec<T>`], but wrapped in a newtype for serialization
/// and interface design purposes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Elements<T>(pub Vec<T>);

impl<T> AsRef<Vec<T>> for Elements<T> {
    fn as_ref(&self) -> &Vec<T> { &self.0 }
}

impl<T> AsMut<Vec<T>> for Elements<T> {
    fn as_mut(&mut self) -> &mut Vec<T> { &mut self.0 }
}

impl<T> AsRef<[T]> for Elements<T> {
    fn as_ref(&self) -> &[T] { self.0.as_ref() }
}

impl<T> AsMut<[T]> for Elements<T> {
    fn as_mut(&mut self) -> &mut [T] { self.0.as_mut() }
}

impl<T> Elements<T>
where T: Serialize
{
    /// Write the contents of `self` to a file.
    ///
    /// Contents are serialized using the [`postcard`] format. Existing files
    /// are silently overwritten.
    pub fn save<P>(&self, path: P) -> CircuitResult<()>
    where P: AsRef<Path>
    {
        let bytes: Vec<u8> =
            postcard::to_stdvec(self)
            .map_err(|_| CircuitError::SerError)?;
        let mut outfile =
            std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .map_err(CircuitError::IOError)?;
        outfile.write_all(&bytes)
            .map_err(CircuitError::IOError)?;
        Ok(())
    }
}

impl<T> Elements<T>
where for<'de> T: Deserialize<'de>
{
    /// Load a sequence of layers from a file.
    ///
    /// Contents are expected in the [`postcard`] format.
    pub fn load<P>(path: P) -> CircuitResult<Self>
    where P: AsRef<Path>
    {
        let mut buf: Vec<u8> = Vec::new();
        let mut infile =
            std::fs::OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(CircuitError::IOError)?;
        infile.read_to_end(&mut buf)
            .map_err(CircuitError::IOError)?;
        let data =
            postcard::from_bytes(&buf)
            .map_err(|_| CircuitError::DeserError)?;
        Ok(data)
    }
}

impl<T, U> FromIterator<U> for Elements<T>
where U: Into<T>
{
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = U>
    {
        Self(iter.into_iter().map(|x| x.into()).collect())
    }
}

impl<'a, T> IntoIterator for &'a Elements<T> {
    type Item = &'a T;
    type IntoIter = <&'a Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl<'a, T> IntoIterator for &'a mut Elements<T> {
    type Item = &'a mut T;
    type IntoIter = <&'a mut Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.iter_mut() }
}

impl<T> IntoIterator for Elements<T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

/// A single layer of unitaries.
pub type UniSeq = Elements<Uni>;

/// Iterator type for qubit indices under a two-qubit gate tiling.
pub struct TileQ2(std::iter::StepBy<std::ops::Range<usize>>);

impl TileQ2 {
    /// Create a new `TileQ2`.
    pub fn new(nqubits: usize, offs: bool) -> Self {
        Self((usize::from(offs) .. nqubits - 1).step_by(2))
    }
}

impl Iterator for TileQ2 {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
}

impl DoubleEndedIterator for TileQ2 {
    fn next_back(&mut self) -> Option<Self::Item> { self.0.next_back() }
}

impl ExactSizeIterator for TileQ2 {
    fn len(&self) -> usize { self.0.len() }
}

impl std::iter::FusedIterator for TileQ2 { }

/// Generate a layer of elements from a function applied to iteration over every
/// other qubit index, optionally offset by 1.
pub fn stagger_layer<F, T>(nqubits: usize, offs: bool, f: F) -> Elements<T>
where F: FnMut(usize) -> T
{
    TileQ2::new(nqubits, offs).map(f).collect()
}

/// Generate a brickwork layer of two-qubit Clifford *matrices*, optionally
/// offset by 1.
pub fn brickwork_cliff<R>(nqubits: usize, offs: bool, rng: &mut R) -> UniSeq
where R: Rng + ?Sized
{
    stagger_layer(
        nqubits, offs, |k| Gate::Cliff2(k).into_matrix_rng(rng).into())
}

/// Generate a brickwork layer of two-qubit Clifford *[`Gate`] sequences*,
/// optionally offset by 1.
pub fn brickwork_cliff_gates<R>(nqubits: usize, offs: bool, rng: &mut R)
    -> UniSeq
where R: Rng + ?Sized
{
    TileQ2::new(nqubits, offs)
        .flat_map(|k| {
            Clifford::gen(2, rng)
            .into_iter()
            .map(move |cliffgate| Gate::from_cliff2(cliffgate).map_idx(|_| k))
        })
        .collect()
}

/// Generate a brickwork layer of Haar-random two-qubit *matrices*, optionally
/// offset by 1.
pub fn brickwork_haar<R>(nqubits: usize, offs: bool, rng: &mut R) -> UniSeq
where R: Rng + ?Sized
{
    stagger_layer(
        nqubits, offs, |k| Gate::Haar2(k).into_matrix_rng(rng).into())
}

/// A single layer of measurements.
pub type MeasSeq = Elements<Meas>;

/// Generate a layer of elements from a function applied to each qubit index,
/// where an operation `T` may or may not be applied to a particular qubit. Each
/// qubit is visited exactly once.
pub fn option_layer<F, T>(nqubits: usize, f: F) -> Elements<T>
where F: FnMut(usize) -> Option<T>
{
    (0..nqubits).flat_map(f).collect()
}

/// Generate a layer of measurements, each independently applied to a single
/// qubit with probability `p`.
pub fn uniform_meas<R>(nqubits: usize, p: f64, rng: &mut R) -> MeasSeq
where R: Rng + ?Sized
{
    option_layer(nqubits, |k| (p < rng.gen::<f64>()).then_some(Meas::Rand(k)))
}

/// A collection of general operations.
pub type OpSeq = Elements<Op>;

/// A combination of a [`UniSeq`] and a [`MeasSeq`], with unitaries held
/// separate from measurements.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiLayer {
    pub unis: UniSeq,
    pub meas: MeasSeq,
}

impl From<(UniSeq, MeasSeq)> for BiLayer {
    fn from(bilayer: (UniSeq, MeasSeq)) -> Self {
        Self { unis: bilayer.0, meas: bilayer.1 }
    }
}

impl From<(MeasSeq, UniSeq)> for BiLayer {
    fn from(bilayer: (MeasSeq, UniSeq)) -> Self {
        Self { unis: bilayer.1, meas: bilayer.0 }
    }
}

/// A circuit formed as a sequence of distinct layers.
pub type LayerCircuit<T> = Elements<Elements<T>>;

/// A circuit formed from distinct layers of unitaries.
pub type UniCircuit = LayerCircuit<Uni>;

/// A circuit formed from distinct unitary + measurement layers.
pub type BiLayerCircuit = Elements<BiLayer>;

/// Apply a single pair of unitary and measurement layers to a [`MPS`], with the
/// unitary layer applied first.
///
/// Measurement outcomes are optionally pushed onto an output buffer as all
/// [`Meas::Proj`], and outcome probabilities are likewise optionally pushed
/// onto a different output buffer along with the qubit index.
pub fn apply_bilayer<'a, U, M, R>(
    state: &mut MPS<Q, C64>,
    unis: U,
    meas: M,
    outcomes: Option<&mut Vec<Meas>>,
    probs: Option<&mut Vec<(usize, f64)>>,
    rng: &mut R,
) -> CircuitResult<()>
where
    U: IntoIterator<Item = &'a Uni>,
    M: IntoIterator<Item = &'a Meas>,
    R: Rng + ?Sized
{
    for uni in unis.into_iter() { state.apply_uni_rng(uni, rng)?; }
    let meas_iter =
        meas.into_iter()
        .filter_map(|m| state.apply_meas_prob(m, rng).map(|o| (m.idx(), o)));
    match (outcomes, probs) {
        (Some(out_buf), Some(prob_buf)) => {
            meas_iter.for_each(|(k, (out, prob))| {
                out_buf.push(Meas::Proj(k, out));
                prob_buf.push((k, prob));
            });
        },
        (Some(out_buf), None) => {
            meas_iter.for_each(|(k, (out, _prob))| {
                out_buf.push(Meas::Proj(k, out));
            });
        },
        (None, Some(prob_buf)) => {
            meas_iter.for_each(|(k, (_out, prob))| {
                prob_buf.push((k, prob));
            });
        },
        (None, None) => {
            meas_iter.for_each(|_| ());
        },
    }
    Ok(())
}

