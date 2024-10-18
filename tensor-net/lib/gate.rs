//! Definitions of common one- and two-qubit gates for use with
//! [`MPS`][crate::mps::MPS] and [`MPSCircuit`][crate::circuit::MPSCircuit].

use ndarray as nd;
use ndarray_linalg::QRSquareInplace;
use num_complex::{ ComplexFloat, Complex64 as C64 };
use num_traits::One;
use once_cell::sync::Lazy;
use rand::{
    Rng,
    distributions::Distribution,
};
use statrs::distribution::Normal;
use crate::ComplexFloatExt;

/// A gate in a quantum circuit.
///
/// Two-qubit gates are limited to nearest neighbors, with the held value always
/// referring to the leftmost of the two relevant qubit indices.
#[derive(Copy, Clone, Debug, PartialEq)]
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

