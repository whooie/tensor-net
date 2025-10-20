//! Definitions of common one- and two-qubit gates for use with
//! [`MPS`][crate::mps::MPS].

use std::borrow::Cow;
use itertools::Itertools;
use ndarray as nd;
use ndarray_linalg::QRSquareInplace;
use num_complex::{ ComplexFloat, Complex64 as C64 };
use num_traits::One;
use once_cell::sync::Lazy;
use rand::{
    Rng,
    thread_rng,
    distributions::Distribution,
};
use serde::{ Serialize, Deserialize };
use statrs::distribution::Normal;
use crate::ComplexFloatExt;

/// A gate in a quantum circuit.
///
/// Two-qubit gates are limited to nearest neighbors, with the held value always
/// referring to the leftmost of the two relevant qubit indices.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Gate {
    /// A gate formed from an Euler angle decomposition.
    U(usize, f64, f64, f64),
    /// Hadamard.
    H(usize),
    /// π rotation about X.
    X(usize),
    /// π rotation about Z.
    Z(usize),
    /// π/2 rotation about Z.
    S(usize),
    /// –π/2 rotation about Z.
    SInv(usize),
    /// An arbitrary rotation about X.
    XRot(usize, f64),
    /// An arbitrary rotation about Z.
    ZRot(usize, f64),
    /// Z-controlled π rotation about X.
    ///
    /// `CX(k)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CX(usize),
    /// Like `CX`, but with the control and target qubits reversed.
    CXRev(usize),
    /// Z-controlled π rotation about Z.
    ///
    /// `CZ(k)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CZ(usize),
    /// A Haar-random two-qubit unitary.
    ///
    /// `Haar2(k)` applies the unitary to the subspace corresponding to the
    /// `k`-th and `k + 1`-th qubits.
    Haar2(usize),
}

impl Gate {
    /// Return `true` if `self` is `U`.
    pub fn is_u(&self) -> bool { matches!(self, Self::U(..)) }

    /// Return `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::H(..)) }

    /// Return `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::X(..)) }

    /// Return `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Z(..)) }

    /// Return `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::S(..)) }

    /// Return `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv(..)) }

    /// Return `true` if `self` is `XRot`.
    pub fn is_xrot(&self) -> bool { matches!(self, Self::XRot(..)) }

    /// Return `true` if `self` is `ZRot`.
    pub fn is_zrot(&self) -> bool { matches!(self, Self::ZRot(..)) }

    /// Return `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX(..)) }

    /// Return `true` if `self` is `CXRev`.
    pub fn is_cxrev(&self) -> bool { matches!(self, Self::CXRev(..)) }

    /// Return `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::CZ(..)) }

    /// Return `true` if `self` is `Haar2`.
    pub fn is_haar2(&self) -> bool { matches!(self, Self::Haar2(..)) }

    /// Return `true` if `self` is a one-qubit gate.
    pub fn is_q1(&self) -> bool {
        matches!(
            self,
            Self::U(..)
            | Self::H(..)
            | Self::X(..)
            | Self::Z(..)
            | Self::S(..)
            | Self::SInv(..)
            | Self::XRot(..)
            | Self::ZRot(..)
        )
    }

    /// Return `true` if `self` is a two-qubit gate.
    pub fn is_q2(&self) -> bool {
        matches!(
            self,
            Self::CX(..) | Self::CXRev(..) | Self::CZ(..) | Self::Haar2(..)
        )
    }

    /// Return the [kind][G] of `self`.
    pub fn kind(&self) -> G {
        use G::*;
        use G1::*;
        use G2::*;
        match *self {
            Self::U(..) => Q1(U),
            Self::H(..) => Q1(H),
            Self::X(..) => Q1(X),
            Self::Z(..) => Q1(Z),
            Self::S(..) => Q1(S),
            Self::SInv(..) => Q1(SInv),
            Self::XRot(..) => Q1(XRot),
            Self::ZRot(..) => Q1(ZRot),
            Self::CX(..) => Q2(CX),
            Self::CXRev(..) => Q2(CXRev),
            Self::CZ(..) => Q2(CZ),
            Self::Haar2(..) => Q2(Haar2),
        }
    }

    /// Sample a single gate.
    pub fn sample_single<G, R>(kind: G, op: G::QubitArg, rng: &mut R) -> Self
    where
        G: GateToken<Self>,
        R: Rng + ?Sized,
    {
        kind.sample(op, rng)
    }

    /// Sample a random one-qubit Clifford gate.
    pub fn sample_c1<R>(k: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        use std::f64::consts::TAU;
        match rng.gen_range(0..8) {
            0 => Self::H(k),
            1 => Self::X(k),
            3 => Self::Z(k),
            4 => Self::S(k),
            5 => Self::SInv(k),
            6 => Self::XRot(k, TAU * rng.gen::<f64>()),
            7 => Self::ZRot(k, TAU * rng.gen::<f64>()),
            _ => unreachable!(),
        }
    }

    /// Sample a random gate uniformly from a set.
    pub fn sample_from<'a, I, G, R>(kinds: I, arg: G::QubitArg, rng: &mut R)
        -> Self
    where
        I: IntoIterator<Item = &'a G>,
        G: GateToken<Self> + 'a,
        R: Rng + ?Sized
    {
        let kinds: Vec<&G> = kinds.into_iter().collect();
        let n = kinds.len();
        kinds[rng.gen_range(0..n)].sample(arg, rng)
    }

    /// Return the index of the (left-most) qubit that the unitary acts on.
    pub fn idx(&self) -> usize {
        match self {
            Self::U(k, ..) => *k,
            Self::H(k) => *k,
            Self::X(k) => *k,
            Self::Z(k) => *k,
            Self::S(k) => *k,
            Self::SInv(k) => *k,
            Self::XRot(k, _) => *k,
            Self::ZRot(k, _) => *k,
            Self::CX(k) => *k,
            Self::CXRev(k) => *k,
            Self::CZ(k) => *k,
            Self::Haar2(k) => *k,
        }
    }

    /// Return a mutable reference to the index of the (left-most) qubit that
    /// the unitary acts on.
    pub fn idx_mut(&mut self) -> &mut usize {
        match self {
            Self::U(k, ..) => k,
            Self::H(k) => k,
            Self::X(k) => k,
            Self::Z(k) => k,
            Self::S(k) => k,
            Self::SInv(k) => k,
            Self::XRot(k, _) => k,
            Self::ZRot(k, _) => k,
            Self::CX(k) => k,
            Self::CXRev(k) => k,
            Self::CZ(k) => k,
            Self::Haar2(k) => k,
        }
    }

    /// Return `self` as a matrix with the relevant target qubit index.
    ///
    /// Note that the `Haar2` variant will require random sampling, for which
    /// the local thread generator will need to be acquired here and then
    /// released immediately afterward. If you are creating matrices for many of
    /// these variants, this will be inefficient and you may wish to use
    /// [`into_matrix_rng`][Self::into_matrix_rng] instead, which takes a cached
    /// generator as argument; this is also useful for fixed-seed applications.
    pub fn into_matrix(self) -> (usize, Cow<'static, nd::Array2<C64>>) {
        match self {
            Self::U(k, a, b, c) => (k, Cow::Owned(make_u(a, b, c))),
            Self::H(k) => (k, Cow::Borrowed(Lazy::force(&HMAT))),
            Self::X(k) => (k, Cow::Borrowed(Lazy::force(&XMAT))),
            Self::Z(k) => (k, Cow::Borrowed(Lazy::force(&ZMAT))),
            Self::S(k) => (k, Cow::Borrowed(Lazy::force(&SMAT))),
            Self::SInv(k) => (k, Cow::Borrowed(Lazy::force(&SINVMAT))),
            Self::XRot(k, ang) => (k, Cow::Owned(make_xrot(ang))),
            Self::ZRot(k, ang) => (k, Cow::Owned(make_zrot(ang))),
            Self::CX(k) => (k, Cow::Borrowed(Lazy::force(&CXMAT))),
            Self::CXRev(k) => (k, Cow::Borrowed(Lazy::force(&CXREVMAT))),
            Self::CZ(k) => (k, Cow::Borrowed(Lazy::force(&CZMAT))),
            Self::Haar2(k) => (k, Cow::Owned(haar(2, &mut thread_rng()))),
        }
    }

    /// Like [`into_matrix`][Self::into_matrix], but taking a cached random
    /// generator as an argument.
    pub fn into_matrix_rng<R>(self, rng: &mut R)
        -> (usize, Cow<'static, nd::Array2<C64>>)
    where R: Rng + ?Sized
    {
        match self {
            Self::U(k, a, b, c) => (k, Cow::Owned(make_u(a, b, c))),
            Self::H(k) => (k, Cow::Borrowed(Lazy::force(&HMAT))),
            Self::X(k) => (k, Cow::Borrowed(Lazy::force(&XMAT))),
            Self::Z(k) => (k, Cow::Borrowed(Lazy::force(&ZMAT))),
            Self::S(k) => (k, Cow::Borrowed(Lazy::force(&SMAT))),
            Self::SInv(k) => (k, Cow::Borrowed(Lazy::force(&SINVMAT))),
            Self::XRot(k, ang) => (k, Cow::Owned(make_xrot(ang))),
            Self::ZRot(k, ang) => (k, Cow::Owned(make_zrot(ang))),
            Self::CX(k) => (k, Cow::Borrowed(Lazy::force(&CXMAT))),
            Self::CXRev(k) => (k, Cow::Borrowed(Lazy::force(&CXREVMAT))),
            Self::CZ(k) => (k, Cow::Borrowed(Lazy::force(&CZMAT))),
            Self::Haar2(k) => (k, Cow::Owned(haar(2, rng))),
        }
    }
}

/// Identifier for a single one-qubit gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G1 {
    /// Euler angles
    U,
    /// Hadamard
    H,
    /// π rotation about X
    X,
    /// π rotation about Z
    Z,
    /// π/2 rotation about Z
    S,
    /// -π/2 rotation about Z
    SInv,
    /// Arbitrary rotation about X
    XRot,
    /// Arbitrary rotation about Z
    ZRot,
}

impl G1 {
    /// Returns `true` if `self` is `U`.
    pub fn is_u(&self) -> bool { matches!(self, Self::U) }

    /// Returns `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::H) }

    /// Returns `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::X) }

    /// Returns `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Z) }

    /// Returns `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::S) }

    /// Returns `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv) }

    /// Returns `true` if `self` is `XRot`.
    pub fn is_xrot(&self) -> bool { matches!(self, Self::XRot) }

    /// Returns `true` if `self` is `ZRot`.
    pub fn is_zrot(&self) -> bool { matches!(self, Self::ZRot) }
}

/// Identifier for a single two-qubit gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G2 {
    /// Z-controlled π rotation about X.
    CX,
    /// Z-controlled π rotation about X with control and target reversed.
    CXRev,
    /// Z-controlled π rotation about Z.
    CZ,
    /// A Haar-random two-qubit unitary.
    Haar2,
}

impl G2 {
    /// Returns `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX) }

    /// Returns `true` if `self` is `CXRev`.
    pub fn is_cxrev(&self) -> bool { matches!(self, Self::CXRev) }

    /// Returns `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::CZ) }

    /// Returns `true` if `self` is `Haar2`.
    pub fn is_haar2(&self) -> bool { matches!(self, Self::Haar2) }
}

/// Identifier for a single gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G {
    /// A one-qubit gate.
    Q1(G1),
    /// A two-qubit gate.
    Q2(G2),
}

impl From<G1> for G {
    fn from(kind1: G1) -> Self { Self::Q1(kind1) }
}

impl From<G2> for G {
    fn from(kind2: G2) -> Self { Self::Q2(kind2) }
}

impl G {
    /// Returns `true` if `self` is `Q1`.
    pub fn is_q1(&self) -> bool { matches!(self, Self::Q1(..)) }

    /// Returns `true` if `self` is `U`.
    pub fn is_u(&self) -> bool { matches!(self, Self::Q1(g) if g.is_u()) }

    /// Returns `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::Q1(g) if g.is_h()) }

    /// Returns `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::Q1(g) if g.is_x()) }

    /// Returns `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Q1(g) if g.is_z()) }

    /// Returns `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::Q1(g) if g.is_s()) }

    /// Returns `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::Q1(g) if g.is_sinv()) }

    /// Returns `true` if `self` is `XRot`.
    pub fn is_xrot(&self) -> bool { matches!(self, Self::Q1(g) if g.is_xrot()) }

    /// Returns `true` if `self` is `ZRot`.
    pub fn is_zrot(&self) -> bool { matches!(self, Self::Q1(g) if g.is_zrot()) }

    /// Returns `true` if `self` is `Q2`.
    pub fn is_q2(&self) -> bool { matches!(self, Self::Q2(..)) }

    /// Returns `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::Q2(g) if g.is_cx()) }

    /// Returns `true` if `self` is `CXRev`.
    pub fn is_cxrev(&self) -> bool { matches!(self, Self::Q2(g) if g.is_cxrev()) }

    /// Returns `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::Q2(g) if g.is_cz()) }

    /// Returns `true` if `self` is `Haar2`.
    pub fn is_haar2(&self) -> bool { matches!(self, Self::Q2(g) if g.is_haar2()) }
}

/// An exact one- or two-qubit unitary.
#[derive(Clone, Debug, PartialEq)]
pub enum ExactGate {
    /// Array is 2x2, applied to a single qubit.
    Q1(usize, nd::Array2<C64>),
    /// Array is 4x4, applied to neighboring qubits with the index identifying
    /// the left.
    Q2(usize, nd::Array2<C64>),
}

impl ExactGate {
    /// Sample a random gate uniformly from a set.
    pub fn sample_from<'a, I, G, R>(kinds: I, arg: G::QubitArg, rng: &mut R)
        -> Self
    where
        I: IntoIterator<Item = &'a G>,
        G: GateToken<Self> + 'a,
        R: Rng + ?Sized
    {
        let kinds: Vec<&G> = kinds.into_iter().collect();
        let n = kinds.len();
        kinds[rng.gen_range(0..n)].sample(arg, rng)
    }

    /// Return the index of the (left-most) qubit that the unitary acts on.
    pub fn idx(&self) -> usize {
        match self {
            Self::Q1(k, _) => *k,
            Self::Q2(k, _) => *k,
        }
    }
}

/// Describes the general behavior for a gate identifier token, e.g. [`G1`] and
/// [`G2`].
pub trait GateToken<G> {
    /// Operand(s) of the gate. Usually this is just the index of the qubit(s)
    /// the gate acts on.
    type QubitArg;

    /// Given a particular kind of gate and a target qubit or qubits, randomly
    /// sample any remaining data to construct a gate object.
    fn sample<R>(&self, op: Self::QubitArg, rng: &mut R) -> G
    where R: Rng + ?Sized;

    /// Return a random element of the token set.
    fn random<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized;

    /// Shortcut to the composition of `random` and `sample`.
    fn sample_random<R>(op: Self::QubitArg, rng: &mut R) -> G
    where
        Self: Sized,
        R: Rng + ?Sized,
    {
        Self::random(rng).sample(op, rng)
    }
}

impl GateToken<Gate> for G1 {
    type QubitArg = usize;

    /// General rotation gates will sample their angles uniformly from `[0,
    /// 2π)`. [`U`][Self::U] gates will sample all three rotation angles
    /// independently.
    fn sample<R>(&self, op: usize, rng: &mut R) -> Gate
    where R: Rng + ?Sized
    {
        use std::f64::consts::TAU;
        match self {
            Self::U => {
                let alpha: f64 = TAU * rng.gen::<f64>();
                let beta: f64 = TAU * rng.gen::<f64>();
                let gamma: f64 = TAU * rng.gen::<f64>();
                Gate::U(op, alpha, beta, gamma)
            },
            Self::H => Gate::H(op),
            Self::X => Gate::X(op),
            Self::Z => Gate::Z(op),
            Self::S => Gate::S(op),
            Self::SInv => Gate::SInv(op),
            Self::XRot => Gate::XRot(op, TAU * rng.gen::<f64>()),
            Self::ZRot => Gate::ZRot(op, TAU * rng.gen::<f64>()),
        }
    }

    fn random<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..8_usize) {
            0 => Self::U,
            1 => Self::H,
            2 => Self::X,
            3 => Self::Z,
            4 => Self::S,
            5 => Self::SInv,
            6 => Self::XRot,
            7 => Self::ZRot,
            _ => unreachable!(),
        }
    }
}

impl GateToken<ExactGate> for G1 {
    type QubitArg = usize;

    /// General rotation gates will sample their angles uniformly from `[0,
    /// 2π)`. [`U`][Self::U] gates will sample all three rotation angles
    /// independently.
    fn sample<R>(&self, op: usize, rng: &mut R) -> ExactGate
    where R: Rng + ?Sized
    {
        use std::f64::consts::TAU;
        match self {
            Self::U => {
                let alpha: f64 = TAU * rng.gen::<f64>();
                let beta: f64 = TAU * rng.gen::<f64>();
                let gamma: f64 = TAU * rng.gen::<f64>();
                ExactGate::Q1(op, make_u(alpha, beta, gamma))
            },
            Self::H => ExactGate::Q1(op, Lazy::force(&HMAT).clone()),
            Self::X => ExactGate::Q1(op, Lazy::force(&XMAT).clone()),
            Self::Z => ExactGate::Q1(op, Lazy::force(&ZMAT).clone()),
            Self::S => ExactGate::Q1(op, Lazy::force(&SMAT).clone()),
            Self::SInv => ExactGate::Q1(op, Lazy::force(&SINVMAT).clone()),
            Self::XRot => ExactGate::Q1(op, make_xrot(TAU * rng.gen::<f64>())),
            Self::ZRot => ExactGate::Q1(op, make_zrot(TAU * rng.gen::<f64>())),
        }
    }

    fn random<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..8_usize) {
            0 => Self::U,
            1 => Self::H,
            2 => Self::X,
            3 => Self::Z,
            4 => Self::S,
            5 => Self::SInv,
            6 => Self::XRot,
            7 => Self::ZRot,
            _ => unreachable!(),
        }
    }
}


impl GateToken<Gate> for G2 {
    type QubitArg = usize;

    fn sample<R>(&self, op: Self::QubitArg, _rng: &mut R) -> Gate
    where R: Rng + ?Sized
    {
        match self {
            Self::CX => Gate::CX(op),
            Self::CXRev => Gate::CXRev(op),
            Self::CZ => Gate::CZ(op),
            Self::Haar2 => Gate::Haar2(op),
        }
    }

    fn random<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..4_usize) {
            0 => Self::CX,
            1 => Self::CXRev,
            2 => Self::CZ,
            3 => Self::Haar2,
            _ => unreachable!(),
        }
    }
}

impl GateToken<ExactGate> for G2 {
    type QubitArg = usize;

    fn sample<R>(&self, op: Self::QubitArg, rng: &mut R) -> ExactGate
    where R: Rng + ?Sized
    {
        match self {
            Self::CX => ExactGate::Q2(op, Lazy::force(&CXMAT).clone()),
            Self::CXRev => ExactGate::Q2(op, Lazy::force(&CXREVMAT).clone()),
            Self::CZ => ExactGate::Q2(op, Lazy::force(&CZMAT).clone()),
            Self::Haar2 => ExactGate::Q2(op, haar(2, rng)),
        }
    }

    fn random<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..4_usize) {
            0 => Self::CX,
            1 => Self::CXRev,
            2 => Self::CZ,
            3 => Self::Haar2,
            _ => unreachable!(),
        }
    }
}

/// Make a single-qubit unitary from its Euler angles.
///
/// This gate is equivalent to `Z(γ) × X(β) × Z(α)`.
pub fn make_u<A>(alpha: A::Real, beta: A::Real, gamma: A::Real)
    -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    let b2 = beta / (A::Real::one() + A::Real::one());
    let ag = alpha + gamma;
    let prefactor = A::cis(b2);
    let ondiag0 = A::from_re(b2.cos());
    let ondiag1 = A::cis(ag) * ondiag0;
    let offdiag = A::from_re(b2.sin());
    let offdiag0 = -A::i() * A::cis(gamma) * offdiag;
    let offdiag1 = -A::i() * A::cis(alpha) * offdiag;
    nd::array![
        [prefactor * ondiag0,  prefactor * offdiag0],
        [prefactor * offdiag1, prefactor * ondiag1 ],
    ]
}

/// Make a Hadamard gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`HMAT`] instead.
pub fn make_h<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    let h = A::from_re((A::Real::one() + A::Real::one()).recip().sqrt());
    nd::array![
        [h,  h],
        [h, -h],
    ]
}

/// Lazy-static version of [`make_h`] for a [`Complex64`][C64] element type.
pub static HMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_h);

/// Make an X gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`XMAT`] instead.
pub fn make_x<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(), A::zero()],
        [A::zero(), A::one()],
    ]
}

/// Lazy-static version of [`make_x`] for a [`Complex64`][C64] element type.
pub static XMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_x);

/// Make a Z gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`ZMAT`] instead.
pub fn make_z<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(),   A::zero()],
        [A::zero(), -A::one() ],
    ]
}

/// Lazy-static version of [`make_z`] for a [`Complex64`][C64] element type.
pub static ZMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_z);

/// Make an S gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`SMAT`] instead.
pub fn make_s<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(),  A::zero()],
        [A::zero(), A::i()   ],
    ]
}

/// Lazy-static version of [`make_s`] for a [`Complex64`][C64] element type.
pub static SMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_s);

/// Make an S<sup>†</sup> gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`SINVMAT`] instead.
pub fn make_sinv<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(),   A::zero()],
        [A::zero(), -A::i()   ],
    ]
}

/// Lazy-static version of [`make_sinv`] for a [`Complex64`][C64] element type.
pub static SINVMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_sinv);

/// Make an X-rotation gate.
pub fn make_xrot<A>(angle: A::Real) -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    let ang2 = angle / (A::Real::one() + A::Real::one());
    let prefactor = A::cis(ang2);
    let ondiag = A::from_re(ang2.cos());
    let offdiag = -A::i() * A::from_re(ang2.sin());
    nd::array![
        [prefactor * ondiag,  prefactor * offdiag],
        [prefactor * offdiag, prefactor * ondiag ],
    ]
}

/// Make a Z-rotation gate.
pub fn make_zrot<A>(angle: A::Real) -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    let ph = A::cis(angle);
    nd::array![
        [A::one(),  A::zero()],
        [A::zero(), ph     ],
    ]
}

/// Make a CX gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`CXMAT`] instead.
pub fn make_cx<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(),  A::zero(), A::zero(), A::zero()],
        [A::zero(), A::one(),  A::zero(), A::zero()],
        [A::zero(), A::zero(), A::zero(), A::one() ],
        [A::zero(), A::zero(), A::one(),  A::zero()],
    ]
}

/// Lazy-static version of [`make_cx`] for a [`Complex64`][C64] element type.
pub static CXMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_cx);

/// Make a CX gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`CXTENS`] instead.
pub fn make_cx2<A>() -> nd::Array4<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    make_cx().into_shape((2, 2, 2, 2)).unwrap()
}

/// Lazy-static version of [`make_cx2`] for a [`Complex64`][C64] element type.
pub static CXTENS: Lazy<nd::Array4<C64>> = Lazy::new(make_cx2);

/// Make a CX gate with the control and target qubits reversed.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`CXREVMAT`] instead.
pub fn make_cxrev<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(),  A::zero(), A::zero(), A::zero()],
        [A::zero(), A::zero(), A::zero(), A::one() ],
        [A::zero(), A::zero(), A::one(),  A::zero()],
        [A::zero(), A::one(),  A::zero(), A::zero()],
    ]
}

/// Lazy-static version of [`make_cxrev`] for a [`Complex64`][C64] element type.
pub static CXREVMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_cxrev);

/// Make a CX gate with the control and target qubits reversed.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`CXREVTENS`] instead.
pub fn make_cxrev2<A>() -> nd::Array4<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    make_cxrev().into_shape((2, 2, 2, 2)).unwrap()
}

/// Lazy-static version of [`make_cxrev2`] for a [`Complex64`][C64] element
/// type.
pub static CXREVTENS: Lazy<nd::Array4<C64>> = Lazy::new(make_cxrev2);

/// Make a CZ gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`CZMAT`] instead.
pub fn make_cz<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(),  A::zero(), A::zero(),  A::zero()],
        [A::zero(), A::one(),  A::zero(),  A::zero()],
        [A::zero(), A::zero(), A::one(),   A::zero()],
        [A::zero(), A::zero(), A::zero(), -A::one() ],
    ]
}

/// Lazy-static version of [`make_cz`] for a [`Complex64`][C64] element type.
pub static CZMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_cz);

/// Make a CZ gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`CZTENS`] instead.
pub fn make_cz2<A>() -> nd::Array4<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    make_cz().into_shape((2, 2, 2, 2)).unwrap()
}

/// Lazy-static version of [`make_cz2`] for a [`Complex64`][C64] element type.
pub static CZTENS: Lazy<nd::Array4<C64>> = Lazy::new(make_cz2);

/// Make a swap gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`SWAPMAT`] instead.
pub fn make_swap<A>() -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    nd::array![
        [A::one(),  A::zero(), A::zero(), A::zero()],
        [A::zero(), A::zero(), A::one(),  A::zero()],
        [A::zero(), A::one(),  A::zero(), A::zero()],
        [A::zero(), A::zero(), A::zero(), A::one() ],
    ]
}

/// Lazy-static version of [`make_swap`] for a [`Complex64`][C64] element type.
pub static SWAPMAT: Lazy<nd::Array2<C64>> = Lazy::new(make_swap);

/// Make a swap gate.
///
/// Since this gate takes no arguments, consider using the lazily-constructed,
/// [`Complex64`][C64]-valued [`SWAPTENS`] instead.
pub fn make_swap2<A>() -> nd::Array4<A>
where
    A: ComplexFloat + ComplexFloatExt,
    <A as ComplexFloat>::Real: std::fmt::Debug,
{
    make_swap().into_shape((2, 2, 2, 2)).unwrap()
}

/// Lazy-static version of [`make_swap2`] for a [`Complex64`][C64] element type.
pub static SWAPTENS: Lazy<nd::Array4<C64>> = Lazy::new(make_swap2);

/// Generate an `n`-qubit Haar-random unitary matrix.
pub fn haar<A, R>(n: usize, rng: &mut R) -> nd::Array2<A>
where
    A: ComplexFloat + ComplexFloatExt,
    nd::Array2<A>: QRSquareInplace<R = nd::Array2<A>>,
    Normal: Distribution<<A as ComplexFloat>::Real>,
    R: Rng + ?Sized,
{
    let normal = Normal::standard();
    let mut z: nd::Array2<A>
        = nd::Array2::from_shape_simple_fn(
            (2_usize.pow(n as u32), 2_usize.pow(n as u32)),
            || A::from_components(normal.sample(rng), normal.sample(rng)),
        );
    let (_, r) = z.qr_square_inplace().unwrap();
    nd::Zip::from(z.columns_mut())
        .and(r.diag())
        .for_each(|mut z_j, rjj| {
            let renorm = *rjj / A::from_re(rjj.abs());
            z_j.map_inplace(|zij| { *zij = *zij / renorm; });
        });
    z
}

/// Description of a single Clifford gate for a register of qubits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CliffGate {
    /// Hadamard
    H(usize),
    /// π rotation about X
    X(usize),
    /// π rotation about Y
    Y(usize),
    /// π rotation about Z
    Z(usize),
    /// π/2 rotation about Z
    S(usize),
    /// -π/2 rotation about Z
    SInv(usize),
    /// Z-controlled π rotation about X.
    ///
    /// The first qubit index is the control.
    CX(usize, usize),
    /// Z-controlled π rotation about Z.
    ///
    /// The first qubit index is the control.
    CZ(usize, usize),
    /// Swap
    Swap(usize, usize),
}

impl CliffGate {
    /// Return `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::H(..)) }

    /// Return `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::X(..)) }

    /// Return `true` if `self` is `Y`.
    pub fn is_y(&self) -> bool { matches!(self, Self::Y(..)) }

    /// Return `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Z(..)) }

    /// Return `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::S(..)) }

    /// Return `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv(..)) }

    /// Return `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX(..)) }

    /// Return `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::CZ(..)) }

    /// Return `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Swap(..)) }

    /// Return `true` if `self` and `other` are inverses.
    pub fn is_inv(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::H(a), Self::H(b)) if a == b => true,
            (Self::X(a), Self::X(b)) if a == b => true,
            (Self::Y(a), Self::Y(b)) if a == b => true,
            (Self::Z(a), Self::Z(b)) if a == b => true,
            (Self::S(a), Self::SInv(b)) if a == b => true,
            (Self::CX(ca, ta), Self::CX(cb, tb)) if ca == cb && ta == tb => true,
            (Self::CZ(ca, ta), Self::CZ(cb, tb))
                if (ca == cb && ta == tb) || (ca == tb && cb == ta) => true,
            (Self::Swap(aa, ba), Self::Swap(ab, bb))
                if (aa == ba && ab == bb) || (aa == bb && ab == ba) => true,
            _ => false,
        }
    }

}

// A single-qubit Pauli operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Pauli {
    /// Identity
    I,
    /// σ<sub>*x*</sub>
    X,
    /// σ<sub>*y*</sub>
    Y,
    /// σ<sub>*z*</sub>
    Z,
}

impl Pauli {
    fn gen<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..4_usize) {
            0 => Self::I,
            1 => Self::X,
            2 => Self::Y,
            3 => Self::Z,
            _ => unreachable!(),
        }
    }

    fn commutes_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I, _) | (_, Self::I) => true,
            (l, r) => l == r,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NPauli(bool, Vec<Pauli>);

impl NPauli {
    fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        Self(rng.gen(), (0..n).map(|_| Pauli::gen(rng)).collect())
    }

    fn commutes_with(&self, other: &Self, skip: Option<usize>) -> bool {
        if self.1.len() != other.1.len() { panic!(); }
        self.1.iter().zip(&other.1)
            .skip(skip.unwrap_or(0))
            .filter(|(l, r)| !l.commutes_with(r))
            .count() % 2 == 0
    }

    fn sample_anticomm<R>(&self, skip: Option<usize>, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let n = self.1.len();
        let mut other: Self;
        loop {
            other = Self::gen(n, rng);
            if !self.commutes_with(&other, skip) {
                return other;
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Col { a: bool, b: bool }

#[derive(Clone, Debug, PartialEq, Eq)]
struct Tableau {
    n: usize,
    x: Vec<Col>,
    z: Vec<Col>,
    s: Col,
    gates: Vec<CliffGate>,
}

impl Tableau {
    fn new(n: usize) -> Self {
        Self {
            n,
            x: vec![Col { a: false, b: false }; n],
            z: vec![Col { a: false, b: false }; n],
            s: Col { a: false, b: false },
            gates: Vec::new(),
        }
    }

    fn set(&mut self, a: &NPauli, b: &NPauli) {
        assert_eq!(a.1.len(), b.1.len());
        assert_eq!(self.n, a.1.len());
        self.x.iter_mut().zip(self.z.iter_mut())
            .zip(a.1.iter().zip(b.1.iter()))
            .for_each(|((xj, zj), (aj, bj))| {
                match aj {
                    Pauli::I => { xj.a = false; zj.a = false; },
                    Pauli::X => { xj.a = true;  zj.a = false; },
                    Pauli::Y => { xj.a = true;  zj.a = true;  },
                    Pauli::Z => { xj.a = false; zj.a = true;  },
                }
                match bj {
                    Pauli::I => { xj.b = false; zj.b = false; },
                    Pauli::X => { xj.b = true;  zj.b = false; },
                    Pauli::Y => { xj.b = true;  zj.b = true;  },
                    Pauli::Z => { xj.b = false; zj.b = true;  },
                }
            });
        self.s.a = a.0;
        self.s.b = b.0;
    }

    fn h(&mut self, k: usize) {
        let xk = &mut self.x[k];
        let zk = &mut self.z[k];
        std::mem::swap(xk, zk);
        self.s.a ^= xk.a && zk.a;
        self.s.b ^= xk.b && zk.b;
        self.gates.push(CliffGate::H(k));
    }

    fn s(&mut self, k: usize) {
        let xk = &mut self.x[k];
        let zk = &mut self.z[k];
        self.s.a ^= xk.a && zk.a;
        self.s.b ^= xk.b && zk.b;
        zk.a ^= xk.a;
        zk.b ^= xk.b;
        self.gates.push(CliffGate::S(k));
    }

    fn cnot(&mut self, c: usize, j: usize) {
        self.x[j].a ^= self.x[c].a;
        self.x[j].b ^= self.x[c].b;
        self.z[c].a ^= self.z[j].a;
        self.z[c].b ^= self.z[j].b;
        self.s.a ^= self.x[c].a && self.z[j].a &&  self.x[j].a &&  self.z[c].a;
        self.s.a ^= self.x[c].a && self.z[j].a && !self.x[j].a && !self.z[c].a;
        self.s.b ^= self.x[c].b && self.z[j].b &&  self.x[j].b &&  self.z[c].b;
        self.s.b ^= self.x[c].b && self.z[j].b && !self.x[j].b && !self.z[c].b;
        self.gates.push(CliffGate::CX(c, j));
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.x.swap(a, b);
        self.z.swap(a, b);
        self.gates.push(CliffGate::Swap(a, b));
    }

    fn iter_xz(&self) -> impl Iterator<Item = (&Col, &Col)> + '_ {
        self.x.iter().zip(self.z.iter())
    }

    fn iter_xz_mut(&mut self) -> impl Iterator<Item = (&mut Col, &mut Col)> + '_ {
        self.x.iter_mut().zip(self.z.iter_mut())
    }

    fn sweep(&mut self, llim: usize) {
        let mut idx_scratch: Vec<usize> = Vec::with_capacity(self.n - llim);
        macro_rules! step_12 {
            ( $tab:ident, $llim:ident, $idx_scratch:ident, $row:ident )
            => {
                // (1)
                // clear top row of z: H
                $tab.iter_xz().enumerate().skip($llim)
                    .filter(|(_, (txj, tzj))| tzj.$row && !txj.$row)
                    .for_each(|(j, _)| { $idx_scratch.push(j); });
                $idx_scratch.drain(..)
                    .for_each(|j| { $tab.h(j); });
                // clear top row of z: S
                $tab.iter_xz().enumerate().skip($llim)
                    .filter(|(_, (txj, tzj))| tzj.$row && txj.$row)
                    .for_each(|(j, _)| { $idx_scratch.push(j); });
                $idx_scratch.drain(..)
                    .for_each(|j| { $tab.s(j); });

                // (2)
                // clear top row of x, all but one: CNOTs
                $tab.iter_xz().enumerate().skip($llim)
                    .filter(|(_, (txj, _))| txj.$row) // guaranteed at least 1 such
                    .for_each(|(j, _)| { $idx_scratch.push(j); });
                while $idx_scratch.len() > 1 {
                    $idx_scratch
                        = $idx_scratch.into_iter()
                        .chunks(2).into_iter()
                        .map(|mut chunk| {
                            let Some(a)
                                = chunk.next() else { unreachable!() };
                            if let Some(b) = chunk.next() {
                                $tab.cnot(a, b);
                            }
                            a
                        })
                        .collect();
                }
            }
        }
        step_12!(self, llim, idx_scratch, a);

        // (3)
        // move the remaining x in the top row to the leftmost column
        if let Some(j) = idx_scratch.first() {
            if *j != llim { self.swap(*j, llim); }
            idx_scratch.pop();
        }

        // (4)
        // apply a hadamard if p1 != Z1.I.I...
        if !self.z[llim].b
            || self.x[llim].b
            || self.iter_xz().skip(llim + 1).any(|(txj, tzj)| txj.b || tzj.b)
        {
            self.h(llim);
            // repeat (1) and (2) above for the bottom row
            step_12!(self, llim, idx_scratch, b);
            self.h(llim);
        }

        // (5)
        // clear signs
        match self.s {
            Col { a: false, b: false } => { },
            Col { a: false, b: true  } => { self.gates.push(CliffGate::X(llim)); },
            Col { a: true,  b: true  } => { self.gates.push(CliffGate::Y(llim)); },
            Col { a: true,  b: false } => { self.gates.push(CliffGate::Z(llim)); },
        }
    }
}

/// A series of [`CliffGate`]s implementing an element of the *N*-qubit Clifford
/// group.
///
/// All gates sourced from this type are guaranteed to apply to qubit indices
/// less than the output of [`Clifford::n`], and all two-qubit gate indices are
/// guaranteed to be non-equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clifford {
    n: usize,
    gates: Vec<CliffGate>,
}

impl IntoIterator for Clifford {
    type Item = CliffGate;
    type IntoIter = <Vec<CliffGate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.gates.into_iter() }
}

impl<'a> IntoIterator for &'a Clifford {
    type Item = &'a CliffGate;
    type IntoIter = <&'a Vec<CliffGate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.gates.iter() }
}

impl Clifford {
    /// Convert a series of gates to a new `n`-qubit Clifford element, verifying
    /// that all qubit indices are less than `n` and that all two-qubit gate
    /// indices are non-equal.
    ///
    /// If the above conditions do not hold, all gates are returned in a new
    /// vector as `Err`.
    pub fn new<I>(n: usize, gates: I) -> Result<Self, Vec<CliffGate>>
    where I: IntoIterator<Item = CliffGate>
    {
        let gates: Vec<CliffGate> = gates.into_iter().collect();
        if gates.iter()
            .all(|gate| {
                match gate {
                    CliffGate::H(k)
                    | CliffGate::X(k)
                    | CliffGate::Y(k)
                    | CliffGate::Z(k)
                    | CliffGate::S(k)
                    | CliffGate::SInv(k)
                    => *k < n,
                    CliffGate::CX(a, b)
                    | CliffGate::CZ(a, b)
                    | CliffGate::Swap(a, b)
                    => *a < n && *b < n && a != b,
                }
            })
        {
            Ok(Self { n, gates })
        } else {
            Err(gates)
        }
    }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.n }

    /// Return the number of gates.
    pub fn len(&self) -> usize { self.gates.len() }

    /// Return `true` if the number of gates is zero.
    pub fn is_empty(&self) -> bool { self.gates.is_empty() }

    /// Return an iterator over the gates implementing the Clifford group
    /// element.
    pub fn iter(&self) -> std::slice::Iter<'_, CliffGate> { self.gates.iter() }

    pub fn unpack(self) -> (Vec<CliffGate>, usize) { (self.gates, self.n) }

    pub fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let mut stab_a: NPauli;
        let mut stab_b: NPauli;
        let mut tab = Tableau::new(n);
        for llim in 0..n {
            stab_a = NPauli::gen(n, rng);
            while stab_a.1.iter().skip(llim).all(|p| *p == Pauli::I) {
                stab_a = NPauli::gen(n, rng);
            }
            stab_b = stab_a.sample_anticomm(Some(llim), rng);
            tab.set(&stab_a, &stab_b);
            tab.sweep(llim);
        }
        Self { n, gates: reduce(tab.gates) }
    }
}

fn reduce(gates: Vec<CliffGate>) -> Vec<CliffGate> {
    let mut reduced: Vec<CliffGate> = Vec::with_capacity(gates.len());
    for gate in gates.into_iter() {
        if let Some(g) = reduced.last() {
            if g.is_inv(&gate) {
                reduced.pop();
            } else {
                reduced.push(gate);
            }
        } else {
            reduced.push(gate);
        }
    }
    reduced
}

