//! Definitions of common one- and two-qubit gates for use with
//! [`MPS`][crate::mps::MPS].
//!
//! Note that all multi-qubit unitary matrices conform to *column-major* (i.e.
//! little-endian) ordering of basis elements.

use std::borrow::Cow;
use itertools::Itertools;
use nalgebra as na;
use num_complex::Complex64 as C64;
use num_traits::{ Float, One };
use once_cell::sync::Lazy;
use rand::{
    Rng,
    thread_rng,
    distributions::Distribution,
};
use serde::{ Serialize, Deserialize };
use statrs::distribution::Normal;
use crate::ComplexScalar;

/// A gate in a quantum circuit.
///
/// Two-qubit gates are limited to nearest neighbors, with the held value always
/// referring to the leftmost of the two relevant qubit indices.
///
/// Here, we take "rotation about *n*" (where *n* is one of {*x*, *y*, *z*}, as
/// in `XRot`, `S`, etc.) to mean the application of a *relative* phase between
/// the qubit states in the *n* basis, i.e.
/// ```text
/// X(θ) = ∣+ ⟩⟨+ ∣ + exp(i θ) ∣– ⟩⟨– ∣
/// Y(θ) = ∣+i⟩⟨+i∣ + exp(i θ) ∣–i⟩⟨–i∣
/// Z(θ) = ∣ 0⟩⟨ 0∣ + exp(i θ) ∣ 1⟩⟨ 1∣
/// ```
/// as represented in the Z basis, where
/// ```text
/// ∣± ⟩ = (∣0⟩ ±   ∣1⟩) / √2
/// ∣±i⟩ = (∣0⟩ ± i ∣1⟩) / √2
/// ```
///
/// These definitions are distinct from the alternative forms of these
/// operations, which are defined by exponentiating the corresponding Pauli
/// matrix:
/// ```text
/// RX(θ) = exp(-i X θ / 2)
/// RY(θ) = exp(-i Y θ / 2)
/// RZ(θ) = exp(-i Z θ / 2)
/// ```
/// The exponentiated versions are available as `*Exp` variants.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Gate {
    /// Hadamard gate.
    H(usize),
    /// π rotation about *x*.
    X(usize),
    /// π rotation about *y*.
    Y(usize),
    /// π rotation about *z*.
    Z(usize),
    /// π/4 rotation about *z*.
    T(usize),
    /// –π/4 rotation about *z*.
    TInv(usize),
    /// π/2 rotation about *z*.
    S(usize),
    /// –π/2 rotation about *z*.
    SInv(usize),
    /// An arbitrary rotation about *x*.
    XRot(usize, f64),
    /// An arbitrary rotation about *y*.
    YRot(usize, f64),
    /// An arbitrary rotation about *z*.
    ZRot(usize, f64),
    /// Exponentiation of `X` to an arbitrary angle.
    XExp(usize, f64),
    /// Exponentiation of `Y` to an arbitrary angle.
    YExp(usize, f64),
    /// Exponentiation of `Z` to an arbitrary angle.
    ZExp(usize, f64),
    /// A gate formed from an Euler angle decomposition.
    ///
    /// ```text
    /// U(α, β, γ) = Z(γ) X(β) Z(α)
    /// ```
    U(usize, f64, f64, f64),
    /// A random element from the one-qubit Clifford group.
    Cliff1(usize),
    /// A Haar-random one-qubit unitary.
    Haar1(usize),
    /// Z-controlled π rotation about X.
    ///
    /// `CX(k)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CX(usize),
    /// Like `CX`, but with the control and target qubits reversed.
    CXRev(usize),
    /// Z-controlled π rotation about Y.
    ///
    /// `CY(k)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CY(usize),
    /// Like `CY`, but with the control and target qubits reversed.
    CYRev(usize),
    /// Z-controlled π rotation about Z.
    ///
    /// `CZ(k)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CZ(usize),
    /// Z-controlled arbitrary rotation about *x*.
    ///
    /// `CXRot(k, _)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CXRot(usize, f64),
    /// Like `CXRot`, but with the control and target qubits reversed.
    CXRotRev(usize, f64),
    /// Z-controlled arbitrary rotation about *y*.
    ///
    /// `CYRot(k, _)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CYRot(usize, f64),
    /// Like `CYRot`, but with the control and target qubits reversed.
    CYRotRev(usize, f64),
    /// Z-controlled arbitrary rotation about *z*.
    ///
    /// `CZRot(k, _)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CZRot(usize, f64),
    /// Z-controlled exponentiation of `X` to an arbitrary angle.
    ///
    /// `CXExp(k, _)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CXExp(usize, f64),
    /// Like `CXRot`, but with the control and target qubits reversed.
    CXExpRev(usize, f64),
    /// Z-controlled exponentiation of `Y` to an arbitrary angle.
    ///
    /// `CYExp(k, _)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CYExp(usize, f64),
    /// Like `CYRot`, but with the control and target qubits reversed.
    CYExpRev(usize, f64),
    /// Z-controlled exponentiation of `Z` to an arbitrary angle.
    ///
    /// `CZExp(k, _)` rotates the `k + 1`-th qubit with the `k`-th qubit as the
    /// control.
    CZExp(usize, f64),
    /// Mølmer-Sørensen gate. This gate uses the *xx* definition.
    MS(usize),
    /// Root-swap gate.
    SqrtSwap(usize),
    /// Swap gate.
    Swap(usize),
    /// A random element from the two-qubit Clifford group.
    ///
    /// `Cliff2(k)` applies the unitary to the subspace corresponding to the
    /// `k`-th` and `k + 1`-th qubits.
    Cliff2(usize),
    /// A Haar-random two-qubit unitary.
    ///
    /// `Haar2(k)` applies the unitary to the subspace corresponding to the
    /// `k`-th and `k + 1`-th qubits.
    Haar2(usize),
}

impl Gate {
    /// Return `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::H(..)) }

    /// Return `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::X(..)) }

    /// Return `true` if `self` is `Y`.
    pub fn is_y(&self) -> bool { matches!(self, Self::Y(..)) }

    /// Return `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Z(..)) }

    /// Return `true` if `self` is `T`.
    pub fn is_t(&self) -> bool { matches!(self, Self::T(..)) }

    /// Return `true` if `self` is `TInv`.
    pub fn is_tinv(&self) -> bool { matches!(self, Self::TInv(..)) }

    /// Return `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::S(..)) }

    /// Return `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv(..)) }

    /// Return `true` if `self` is `XRot`.
    pub fn is_xrot(&self) -> bool { matches!(self, Self::XRot(..)) }

    /// Return `true` if `self` is `YRot`.
    pub fn is_yrot(&self) -> bool { matches!(self, Self::YRot(..)) }

    /// Return `true` if `self` is `ZRot`.
    pub fn is_zrot(&self) -> bool { matches!(self, Self::ZRot(..)) }

    /// Return `true` if `self` is `XExp`.
    pub fn is_xexp(&self) -> bool { matches!(self, Self::XExp(..)) }

    /// Return `true` if `self` is `YExp`.
    pub fn is_yexp(&self) -> bool { matches!(self, Self::YExp(..)) }

    /// Return `true` if `self` is `ZExp`.
    pub fn is_zexp(&self) -> bool { matches!(self, Self::ZExp(..)) }

    /// Return `true` if `self` is `U`.
    pub fn is_u(&self) -> bool { matches!(self, Self::U(..)) }

    /// Return `true` if `self` is `Cliff1`.
    pub fn is_cliff1(&self) -> bool { matches!(self, Self::Cliff1(..)) }

    /// Return `true` if `self` is `Haar1`.
    pub fn is_haar1(&self) -> bool { matches!(self, Self::Haar1(..)) }

    /// Return `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX(..)) }

    /// Return `true` if `self` is `CXRev`.
    pub fn is_cxrev(&self) -> bool { matches!(self, Self::CXRev(..)) }

    /// Return `true` if `self` is `CY`.
    pub fn is_cy(&self) -> bool { matches!(self, Self::CY(..)) }

    /// Return `true` if `self` is `CYRev`.
    pub fn is_cyrev(&self) -> bool { matches!(self, Self::CYRev(..)) }

    /// Return `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::CZ(..)) }

    /// Return `true` if `self` is `CXRot`.
    pub fn is_cxrot(&self) -> bool { matches!(self, Self::CXRot(..)) }

    /// Return `true` if `self` is `CXRotRev`.
    pub fn is_cxrotrev(&self) -> bool { matches!(self, Self::CXRotRev(..)) }

    /// Return `true` if `self` is `CYRot`.
    pub fn is_cyrot(&self) -> bool { matches!(self, Self::CYRot(..)) }

    /// Return `true` if `self` is `CYRotRev`.
    pub fn is_cyrotrev(&self) -> bool { matches!(self, Self::CYRotRev(..)) }

    /// Return `true` if `self` is `CZRot`.
    pub fn is_czrot(&self) -> bool { matches!(self, Self::CZRot(..)) }

    /// Return `true` if `self` is `CXExp`.
    pub fn is_cxexp(&self) -> bool { matches!(self, Self::CXExp(..)) }

    /// Return `true` if `self` is `CXExpRev`.
    pub fn is_cxexprev(&self) -> bool { matches!(self, Self::CXExpRev(..)) }

    /// Return `true` if `self` is `CYExp`.
    pub fn is_cyexp(&self) -> bool { matches!(self, Self::CYExp(..)) }

    /// Return `true` if `self` is `CYExpRev`.
    pub fn is_cyexprev(&self) -> bool { matches!(self, Self::CYExpRev(..)) }

    /// Return `true` if `self` is `CZExp`.
    pub fn is_czexp(&self) -> bool { matches!(self, Self::CZExp(..)) }

    /// Return `true` if `self` is `MS`.
    pub fn is_ms(&self) -> bool { matches!(self, Self::MS(..)) }

    /// Return `true` if `self` is `SqrtSwap`.
    pub fn is_sqrtswap(&self) -> bool { matches!(self, Self::SqrtSwap(..)) }

    /// Return `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Swap(..)) }

    /// Return `true` if `self` is `Cliff2`.
    pub fn is_cliff2(&self) -> bool { matches!(self, Self::Cliff2(..)) }

    /// Return `true` if `self` is `Haar2`.
    pub fn is_haar2(&self) -> bool { matches!(self, Self::Haar2(..)) }

    /// Return `true` if `self` is a one-qubit gate.
    pub fn is_q1(&self) -> bool {
        matches!(
            self,
            Self::H(..)
            | Self::X(..)
            | Self::Y(..)
            | Self::Z(..)
            | Self::T(..)
            | Self::TInv(..)
            | Self::S(..)
            | Self::SInv(..)
            | Self::XRot(..)
            | Self::YRot(..)
            | Self::ZRot(..)
            | Self::XExp(..)
            | Self::YExp(..)
            | Self::ZExp(..)
            | Self::U(..)
            | Self::Cliff1(..)
            | Self::Haar1(..)
        )
    }

    /// Return `true` if `self` is a two-qubit gate.
    pub fn is_q2(&self) -> bool {
        matches!(
            self,
            Self::CX(..)
            | Self::CXRev(..)
            | Self::CY(..)
            | Self::CYRev(..)
            | Self::CZ(..)
            | Self::CXRot(..)
            | Self::CXRotRev(..)
            | Self::CYRot(..)
            | Self::CYRotRev(..)
            | Self::CZRot(..)
            | Self::CXExp(..)
            | Self::CXExpRev(..)
            | Self::CYExp(..)
            | Self::CYExpRev(..)
            | Self::CZExp(..)
            | Self::MS(..)
            | Self::SqrtSwap(..)
            | Self::Swap(..)
            | Self::Cliff2(..)
            | Self::Haar2(..)
        )
    }

    /// Return the [kind][G] of `self`.
    pub fn kind(&self) -> G {
        use G::*;
        use G1::*;
        use G2::*;
        match self {
            Self::H(..) => Q1(H),
            Self::X(..) => Q1(X),
            Self::Y(..) => Q1(Y),
            Self::Z(..) => Q1(Z),
            Self::T(..) => Q1(T),
            Self::TInv(..) => Q1(TInv),
            Self::S(..) => Q1(S),
            Self::SInv(..) => Q1(SInv),
            Self::XRot(..) => Q1(XRot),
            Self::YRot(..) => Q1(YRot),
            Self::ZRot(..) => Q1(ZRot),
            Self::XExp(..) => Q1(XExp),
            Self::YExp(..) => Q1(YExp),
            Self::ZExp(..) => Q1(ZExp),
            Self::U(..) => Q1(U),
            Self::Cliff1(..) => Q1(Cliff1),
            Self::Haar1(..) => Q1(Haar1),
            Self::CX(..) => Q2(CX),
            Self::CXRev(..) => Q2(CXRev),
            Self::CY(..) => Q2(CY),
            Self::CYRev(..) => Q2(CYRev),
            Self::CZ(..) => Q2(CZ),
            Self::CXRot(..) => Q2(CXRot),
            Self::CXRotRev(..) => Q2(CXRotRev),
            Self::CYRot(..) => Q2(CYRot),
            Self::CYRotRev(..) => Q2(CYRotRev),
            Self::CZRot(..) => Q2(CZRot),
            Self::CXExp(..) => Q2(CXExp),
            Self::CXExpRev(..) => Q2(CXExpRev),
            Self::CYExp(..) => Q2(CYExp),
            Self::CYExpRev(..) => Q2(CYExpRev),
            Self::CZExp(..) => Q2(CZExp),
            Self::MS(..) => Q2(MS),
            Self::SqrtSwap(..) => Q2(SqrtSwap),
            Self::Swap(..) => Q2(Swap),
            Self::Cliff2(..) => Q2(Cliff2),
            Self::Haar2(..) => Q2(Haar2),
        }
    }

    /// Return the index of the (left-most) qubit that the unitary acts on.
    pub fn idx(&self) -> usize {
        match self {
            Self::H(k, ..) => *k,
            Self::X(k, ..) => *k,
            Self::Y(k, ..) => *k,
            Self::Z(k, ..) => *k,
            Self::T(k, ..) => *k,
            Self::TInv(k, ..) => *k,
            Self::S(k, ..) => *k,
            Self::SInv(k, ..) => *k,
            Self::XRot(k, ..) => *k,
            Self::YRot(k, ..) => *k,
            Self::ZRot(k, ..) => *k,
            Self::XExp(k, ..) => *k,
            Self::YExp(k, ..) => *k,
            Self::ZExp(k, ..) => *k,
            Self::U(k, ..) => *k,
            Self::Cliff1(k, ..) => *k,
            Self::Haar1(k, ..) => *k,
            Self::CX(k, ..) => *k,
            Self::CXRev(k, ..) => *k,
            Self::CY(k, ..) => *k,
            Self::CYRev(k, ..) => *k,
            Self::CZ(k, ..) => *k,
            Self::CXRot(k, ..) => *k,
            Self::CXRotRev(k, ..) => *k,
            Self::CYRot(k, ..) => *k,
            Self::CYRotRev(k, ..) => *k,
            Self::CZRot(k, ..) => *k,
            Self::CXExp(k, ..) => *k,
            Self::CXExpRev(k, ..) => *k,
            Self::CYExp(k, ..) => *k,
            Self::CYExpRev(k, ..) => *k,
            Self::CZExp(k, ..) => *k,
            Self::MS(k, ..) => *k,
            Self::SqrtSwap(k, ..) => *k,
            Self::Swap(k, ..) => *k,
            Self::Cliff2(k, ..) => *k,
            Self::Haar2(k, ..) => *k,
        }
    }

    /// Return a mutable reference to the index of the (left-most) qubit that
    /// the unitary acts on.
    pub fn idx_mut(&mut self) -> &mut usize {
        match self {
            Self::H(k, ..) => k,
            Self::X(k, ..) => k,
            Self::Y(k, ..) => k,
            Self::Z(k, ..) => k,
            Self::T(k, ..) => k,
            Self::TInv(k, ..) => k,
            Self::S(k, ..) => k,
            Self::SInv(k, ..) => k,
            Self::XRot(k, ..) => k,
            Self::YRot(k, ..) => k,
            Self::ZRot(k, ..) => k,
            Self::XExp(k, ..) => k,
            Self::YExp(k, ..) => k,
            Self::ZExp(k, ..) => k,
            Self::U(k, ..) => k,
            Self::Cliff1(k, ..) => k,
            Self::Haar1(k, ..) => k,
            Self::CX(k, ..) => k,
            Self::CXRev(k, ..) => k,
            Self::CY(k, ..) => k,
            Self::CYRev(k, ..) => k,
            Self::CZ(k, ..) => k,
            Self::CXRot(k, ..) => k,
            Self::CXRotRev(k, ..) => k,
            Self::CYRot(k, ..) => k,
            Self::CYRotRev(k, ..) => k,
            Self::CZRot(k, ..) => k,
            Self::CXExp(k, ..) => k,
            Self::CXExpRev(k, ..) => k,
            Self::CYExp(k, ..) => k,
            Self::CYExpRev(k, ..) => k,
            Self::CZExp(k, ..) => k,
            Self::MS(k, ..) => k,
            Self::SqrtSwap(k, ..) => k,
            Self::Swap(k, ..) => k,
            Self::Cliff2(k, ..) => k,
            Self::Haar2(k, ..) => k,
        }
    }

    /// Apply a mapping function to the index of the (left-most) qubit that the
    /// unitary acts on.
    pub fn map_idx<F>(mut self, f: F) -> Self
    where F: FnOnce(usize) -> usize
    {
        let k = self.idx_mut();
        *k = f(*k);
        self
    }

    pub(crate) fn from_cliff2(cliff: CliffGate) -> Self {
        match cliff {
            CliffGate::H(k) => Self::H(k),
            CliffGate::X(k) => Self::X(k),
            CliffGate::Y(k) => Self::Y(k),
            CliffGate::Z(k) => Self::Z(k),
            CliffGate::S(k) => Self::S(k),
            CliffGate::SInv(k) => Self::SInv(k),
            CliffGate::CX(c, t) => {
                assert!((c == 0 || c == 1) && (t == 0 || t == 1));
                if c < t { Self::CX(c) } else { Self::CXRev(t) }
            },
            CliffGate::CZ(a, b) => {
                assert!((a == 0 || a == 1) && (b == 0 || b == 1));
                Self::CZ(a.min(b))
            },
            CliffGate::Swap(a, b) => {
                assert!((a == 0 || a == 1) && (b == 0 || b == 1));
                Self::Swap(a.min(b))
            },
        }
    }

    /// Return `self` as a matrix with the relevant target qubit index.
    ///
    /// Note that the `Cliff*` and `Haar*` variants will require random
    /// sampling, for which the local thread generator will need to be acquired
    /// here and then released immediately afterward. If you are creating
    /// matrices for many of these variants, this will be inefficient and you
    /// may wish to use [`into_matrix_rng`][Self::into_matrix_rng] instead,
    /// which takes a cached generator as argument; this is also useful for
    /// fixed-seed applications.
    pub fn into_matrix(self) -> (usize, Cow<'static, na::DMatrix<C64>>) {
        match self {
            Self::H(k) => (k, Cow::Borrowed(Lazy::force(&HMAT))),
            Self::X(k) => (k, Cow::Borrowed(Lazy::force(&XMAT))),
            Self::Y(k) => (k, Cow::Borrowed(Lazy::force(&YMAT))),
            Self::Z(k) => (k, Cow::Borrowed(Lazy::force(&ZMAT))),
            Self::T(k) => (k, Cow::Borrowed(Lazy::force(&TMAT))),
            Self::TInv(k) => (k, Cow::Borrowed(Lazy::force(&TINVMAT))),
            Self::S(k) => (k, Cow::Borrowed(Lazy::force(&SMAT))),
            Self::SInv(k) => (k, Cow::Borrowed(Lazy::force(&SINVMAT))),
            Self::XRot(k, ang) => (k, Cow::Owned(make_xrot(ang))),
            Self::YRot(k, ang) => (k, Cow::Owned(make_yrot(ang))),
            Self::ZRot(k, ang) => (k, Cow::Owned(make_zrot(ang))),
            Self::XExp(k, ang) => (k, Cow::Owned(make_xexp(ang))),
            Self::YExp(k, ang) => (k, Cow::Owned(make_yexp(ang))),
            Self::ZExp(k, ang) => (k, Cow::Owned(make_zexp(ang))),
            Self::U(k, a, b, c) => (k, Cow::Owned(make_u(a, b, c))),
            Self::Cliff1(k) => (k, Cow::Owned(make_cliff(1, &mut thread_rng()))),
            Self::Haar1(k) => (k, Cow::Owned(make_haar(1, &mut thread_rng()))),
            Self::CX(k) => (k, Cow::Borrowed(Lazy::force(&CXMAT))),
            Self::CXRev(k) => (k, Cow::Borrowed(Lazy::force(&CXREVMAT))),
            Self::CY(k) => (k, Cow::Borrowed(Lazy::force(&CYMAT))),
            Self::CYRev(k) => (k, Cow::Borrowed(Lazy::force(&CYREVMAT))),
            Self::CZ(k) => (k, Cow::Borrowed(Lazy::force(&CZMAT))),
            Self::CXRot(k, ang) => (k, Cow::Owned(make_cxrot(ang))),
            Self::CXRotRev(k, ang) => (k, Cow::Owned(make_cxrotrev(ang))),
            Self::CYRot(k, ang) => (k, Cow::Owned(make_cyrot(ang))),
            Self::CYRotRev(k, ang) => (k, Cow::Owned(make_cyrotrev(ang))),
            Self::CZRot(k, ang) => (k, Cow::Owned(make_czrot(ang))),
            Self::CXExp(k, ang) => (k, Cow::Owned(make_cxexp(ang))),
            Self::CXExpRev(k, ang) => (k, Cow::Owned(make_cxexprev(ang))),
            Self::CYExp(k, ang) => (k, Cow::Owned(make_cyexp(ang))),
            Self::CYExpRev(k, ang) => (k, Cow::Owned(make_cyexprev(ang))),
            Self::CZExp(k, ang) => (k, Cow::Owned(make_czexp(ang))),
            Self::MS(k) => (k, Cow::Borrowed(Lazy::force(&MSMAT))),
            Self::SqrtSwap(k) => (k, Cow::Borrowed(Lazy::force(&SQRTSWAPMAT))),
            Self::Swap(k) => (k, Cow::Borrowed(Lazy::force(&SWAPMAT))),
            Self::Cliff2(k) => (k, Cow::Owned(make_cliff(2, &mut thread_rng()))),
            Self::Haar2(k) => (k, Cow::Owned(make_haar(2, &mut thread_rng()))),
        }
    }

    /// Like [`into_matrix`][Self::into_matrix], but taking a cached random
    /// generator as an argument.
    pub fn into_matrix_rng<R>(self, rng: &mut R)
        -> (usize, Cow<'static, na::DMatrix<C64>>)
    where R: Rng + ?Sized
    {
        match self {
            Self::H(k) => (k, Cow::Borrowed(Lazy::force(&HMAT))),
            Self::X(k) => (k, Cow::Borrowed(Lazy::force(&XMAT))),
            Self::Y(k) => (k, Cow::Borrowed(Lazy::force(&YMAT))),
            Self::Z(k) => (k, Cow::Borrowed(Lazy::force(&ZMAT))),
            Self::T(k) => (k, Cow::Borrowed(Lazy::force(&TMAT))),
            Self::TInv(k) => (k, Cow::Borrowed(Lazy::force(&TINVMAT))),
            Self::S(k) => (k, Cow::Borrowed(Lazy::force(&SMAT))),
            Self::SInv(k) => (k, Cow::Borrowed(Lazy::force(&SINVMAT))),
            Self::XRot(k, ang) => (k, Cow::Owned(make_xrot(ang))),
            Self::YRot(k, ang) => (k, Cow::Owned(make_yrot(ang))),
            Self::ZRot(k, ang) => (k, Cow::Owned(make_zrot(ang))),
            Self::XExp(k, ang) => (k, Cow::Owned(make_xexp(ang))),
            Self::YExp(k, ang) => (k, Cow::Owned(make_yexp(ang))),
            Self::ZExp(k, ang) => (k, Cow::Owned(make_zexp(ang))),
            Self::U(k, a, b, c) => (k, Cow::Owned(make_u(a, b, c))),
            Self::Cliff1(k) => (k, Cow::Owned(make_cliff(1, rng))),
            Self::Haar1(k) => (k, Cow::Owned(make_haar(1, rng))),
            Self::CX(k) => (k, Cow::Borrowed(Lazy::force(&CXMAT))),
            Self::CXRev(k) => (k, Cow::Borrowed(Lazy::force(&CXREVMAT))),
            Self::CY(k) => (k, Cow::Borrowed(Lazy::force(&CYMAT))),
            Self::CYRev(k) => (k, Cow::Borrowed(Lazy::force(&CYREVMAT))),
            Self::CZ(k) => (k, Cow::Borrowed(Lazy::force(&CZMAT))),
            Self::CXRot(k, ang) => (k, Cow::Owned(make_cxrot(ang))),
            Self::CXRotRev(k, ang) => (k, Cow::Owned(make_cxrotrev(ang))),
            Self::CYRot(k, ang) => (k, Cow::Owned(make_cyrot(ang))),
            Self::CYRotRev(k, ang) => (k, Cow::Owned(make_cyrotrev(ang))),
            Self::CZRot(k, ang) => (k, Cow::Owned(make_czrot(ang))),
            Self::CXExp(k, ang) => (k, Cow::Owned(make_cxexp(ang))),
            Self::CXExpRev(k, ang) => (k, Cow::Owned(make_cxexprev(ang))),
            Self::CYExp(k, ang) => (k, Cow::Owned(make_cyexp(ang))),
            Self::CYExpRev(k, ang) => (k, Cow::Owned(make_cyexprev(ang))),
            Self::CZExp(k, ang) => (k, Cow::Owned(make_czexp(ang))),
            Self::MS(k) => (k, Cow::Borrowed(Lazy::force(&MSMAT))),
            Self::SqrtSwap(k) => (k, Cow::Borrowed(Lazy::force(&SQRTSWAPMAT))),
            Self::Swap(k) => (k, Cow::Borrowed(Lazy::force(&SWAPMAT))),
            Self::Cliff2(k) => (k, Cow::Owned(make_cliff(2, rng))),
            Self::Haar2(k) => (k, Cow::Owned(make_haar(2, rng))),
        }
    }
}

/// Identifier for a single one-qubit gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G1 {
    /// Hadamard gate.
    H,
    /// π rotation about *x*.
    X,
    /// π rotation about *y*.
    Y,
    /// π rotation about *z*.
    Z,
    /// π/4 rotation about *z*.
    T,
    /// –π/4 rotation about *z*.
    TInv,
    /// π/2 rotation about *z*.
    S,
    /// –π/2 rotation about *z*.
    SInv,
    /// An arbitrary rotation about *x*.
    XRot,
    /// An arbitrary rotation about *y*.
    YRot,
    /// An arbitrary rotation about *z*.
    ZRot,
    /// Exponentiation of `X` to an arbitrary angle.
    XExp,
    /// Exponentiation of `Y` to an arbitrary angle.
    YExp,
    /// Exponentiation of `Z` to an arbitrary angle.
    ZExp,
    /// A gate formed from an Euler angle decomposition.
    U,
    /// A random element from the one-qubit Clifford group.
    Cliff1,
    /// A Haar-random one-qubit unitary.
    Haar1,
}

impl G1 {
    /// Return `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::H) }

    /// Return `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::X) }

    /// Return `true` if `self` is `Y`.
    pub fn is_y(&self) -> bool { matches!(self, Self::Y) }

    /// Return `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Z) }

    /// Return `true` if `self` is `T`.
    pub fn is_t(&self) -> bool { matches!(self, Self::T) }

    /// Return `true` if `self` is `TInv`.
    pub fn is_tinv(&self) -> bool { matches!(self, Self::TInv) }

    /// Return `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::S) }

    /// Return `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv) }

    /// Return `true` if `self` is `XRot`.
    pub fn is_xrot(&self) -> bool { matches!(self, Self::XRot) }

    /// Return `true` if `self` is `YRot`.
    pub fn is_yrot(&self) -> bool { matches!(self, Self::YRot) }

    /// Return `true` if `self` is `ZRot`.
    pub fn is_zrot(&self) -> bool { matches!(self, Self::ZRot) }

    /// Return `true` if `self` is `XExp`.
    pub fn is_xexp(&self) -> bool { matches!(self, Self::XExp) }

    /// Return `true` if `self` is `YExp`.
    pub fn is_yexp(&self) -> bool { matches!(self, Self::YExp) }

    /// Return `true` if `self` is `ZExp`.
    pub fn is_zexp(&self) -> bool { matches!(self, Self::ZExp) }

    /// Return `true` if `self` is `U`.
    pub fn is_u(&self) -> bool { matches!(self, Self::U) }

    /// Return `true` if `self` is `Cliff1`.
    pub fn is_cliff1(&self) -> bool { matches!(self, Self::Cliff1) }

    /// Return `true` if `self` is `Haar1`.
    pub fn is_haar1(&self) -> bool { matches!(self, Self::Haar1) }
}

/// Identifier for a single two-qubit gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G2 {
    /// Z-controlled π rotation about X.
    CX,
    /// Like `CX`, but with the control and target qubits reversed.
    CXRev,
    /// Z-controlled π rotation about Y.
    CY,
    /// Like `CY`, but with the control and target qubits reversed.
    CYRev,
    /// Z-controlled π rotation about Z.
    CZ,
    /// Z-controlled arbitrary rotation about *x*.
    CXRot,
    /// Like `CXRot`, but with the control and target qubits reversed.
    CXRotRev,
    /// Z-controlled arbitrary rotation about *y*.
    CYRot,
    /// Like `CYRot`, but with the control and target qubits reversed.
    CYRotRev,
    /// Z-controlled arbitrary rotation about *z*.
    CZRot,
    /// Z-controlled exponentiation of `X` to an arbitrary angle.
    CXExp,
    /// Like `CXRot`, but with the control and target qubits reversed.
    CXExpRev,
    /// Z-controlled exponentiation of `Y` to an arbitrary angle.
    CYExp,
    /// Like `CYRot`, but with the control and target qubits reversed.
    CYExpRev,
    /// Z-controlled exponentiation of `Z` to an arbitrary angle.
    CZExp,
    /// Mølmer-Sørensen gate.
    MS,
    /// Root-swap gate.
    SqrtSwap,
    /// Swap gate.
    Swap,
    /// A random element from the two-qubit Clifford group.
    Cliff2,
    /// A Haar-random two-qubit unitary.
    Haar2,
}

impl G2 {
    /// Return `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX) }

    /// Return `true` if `self` is `CXRev`.
    pub fn is_cxrev(&self) -> bool { matches!(self, Self::CXRev) }

    /// Return `true` if `self` is `CY`.
    pub fn is_cy(&self) -> bool { matches!(self, Self::CY) }

    /// Return `true` if `self` is `CYRev`.
    pub fn is_cyrev(&self) -> bool { matches!(self, Self::CYRev) }

    /// Return `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::CZ) }

    /// Return `true` if `self` is `CXRot`.
    pub fn is_cxrot(&self) -> bool { matches!(self, Self::CXRot) }

    /// Return `true` if `self` is `CXRotRev`.
    pub fn is_cxrotrev(&self) -> bool { matches!(self, Self::CXRotRev) }

    /// Return `true` if `self` is `CYRot`.
    pub fn is_cyrot(&self) -> bool { matches!(self, Self::CYRot) }

    /// Return `true` if `self` is `CYRotRev`.
    pub fn is_cyrotrev(&self) -> bool { matches!(self, Self::CYRotRev) }

    /// Return `true` if `self` is `CZRot`.
    pub fn is_czrot(&self) -> bool { matches!(self, Self::CZRot) }

    /// Return `true` if `self` is `CXExp`.
    pub fn is_cxexp(&self) -> bool { matches!(self, Self::CXExp) }

    /// Return `true` if `self` is `CXExpRev`.
    pub fn is_cxexprev(&self) -> bool { matches!(self, Self::CXExpRev) }

    /// Return `true` if `self` is `CYExp`.
    pub fn is_cyexp(&self) -> bool { matches!(self, Self::CYExp) }

    /// Return `true` if `self` is `CYExpRev`.
    pub fn is_cyexprev(&self) -> bool { matches!(self, Self::CYExpRev) }

    /// Return `true` if `self` is `CZExp`.
    pub fn is_czexp(&self) -> bool { matches!(self, Self::CZExp) }

    /// Return `true` if `self` is `MS`.
    pub fn is_ms(&self) -> bool { matches!(self, Self::MS) }

    /// Return `true` if `self` is `SqrtSwap`.
    pub fn is_sqrtswap(&self) -> bool { matches!(self, Self::SqrtSwap) }

    /// Return `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Swap) }

    /// Return `true` if `self` is `Cliff2`.
    pub fn is_cliff2(&self) -> bool { matches!(self, Self::Cliff2) }

    /// Return `true` if `self` is `Haar2`.
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

    /// Return `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::Q1(G1::H)) }

    /// Return `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::Q1(G1::X)) }

    /// Return `true` if `self` is `Y`.
    pub fn is_y(&self) -> bool { matches!(self, Self::Q1(G1::Y)) }

    /// Return `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Q1(G1::Z)) }

    /// Return `true` if `self` is `T`.
    pub fn is_t(&self) -> bool { matches!(self, Self::Q1(G1::T)) }

    /// Return `true` if `self` is `TInv`.
    pub fn is_tinv(&self) -> bool { matches!(self, Self::Q1(G1::TInv)) }

    /// Return `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::Q1(G1::S)) }

    /// Return `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::Q1(G1::SInv)) }

    /// Return `true` if `self` is `XRot`.
    pub fn is_xrot(&self) -> bool { matches!(self, Self::Q1(G1::XRot)) }

    /// Return `true` if `self` is `YRot`.
    pub fn is_yrot(&self) -> bool { matches!(self, Self::Q1(G1::YRot)) }

    /// Return `true` if `self` is `ZRot`.
    pub fn is_zrot(&self) -> bool { matches!(self, Self::Q1(G1::ZRot)) }

    /// Return `true` if `self` is `XExp`.
    pub fn is_xexp(&self) -> bool { matches!(self, Self::Q1(G1::XExp)) }

    /// Return `true` if `self` is `YExp`.
    pub fn is_yexp(&self) -> bool { matches!(self, Self::Q1(G1::YExp)) }

    /// Return `true` if `self` is `ZExp`.
    pub fn is_zexp(&self) -> bool { matches!(self, Self::Q1(G1::ZExp)) }

    /// Return `true` if `self` is `U`.
    pub fn is_u(&self) -> bool { matches!(self, Self::Q1(G1::U)) }

    /// Return `true` if `self` is `Cliff1`.
    pub fn is_cliff1(&self) -> bool { matches!(self, Self::Q1(G1::Cliff1)) }

    /// Return `true` if `self` is `Haar1`.
    pub fn is_haar1(&self) -> bool { matches!(self, Self::Q1(G1::Haar1)) }

    /// Returns `true` if `self` is `Q2`.
    pub fn is_q2(&self) -> bool { matches!(self, Self::Q2(..)) }

    /// Return `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::Q2(G2::CX)) }

    /// Return `true` if `self` is `CXRev`.
    pub fn is_cxrev(&self) -> bool { matches!(self, Self::Q2(G2::CXRev)) }

    /// Return `true` if `self` is `CY`.
    pub fn is_cy(&self) -> bool { matches!(self, Self::Q2(G2::CY)) }

    /// Return `true` if `self` is `CYRev`.
    pub fn is_cyrev(&self) -> bool { matches!(self, Self::Q2(G2::CYRev)) }

    /// Return `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::Q2(G2::CZ)) }

    /// Return `true` if `self` is `CXRot`.
    pub fn is_cxrot(&self) -> bool { matches!(self, Self::Q2(G2::CXRot)) }

    /// Return `true` if `self` is `CXRotRev`.
    pub fn is_cxrotrev(&self) -> bool { matches!(self, Self::Q2(G2::CXRotRev)) }

    /// Return `true` if `self` is `CYRot`.
    pub fn is_cyrot(&self) -> bool { matches!(self, Self::Q2(G2::CYRot)) }

    /// Return `true` if `self` is `CYRotRev`.
    pub fn is_cyrotrev(&self) -> bool { matches!(self, Self::Q2(G2::CYRotRev)) }

    /// Return `true` if `self` is `CZRot`.
    pub fn is_czrot(&self) -> bool { matches!(self, Self::Q2(G2::CZRot)) }

    /// Return `true` if `self` is `CXExp`.
    pub fn is_cxexp(&self) -> bool { matches!(self, Self::Q2(G2::CXExp)) }

    /// Return `true` if `self` is `CXExpRev`.
    pub fn is_cxexprev(&self) -> bool { matches!(self, Self::Q2(G2::CXExpRev)) }

    /// Return `true` if `self` is `CYExp`.
    pub fn is_cyexp(&self) -> bool { matches!(self, Self::Q2(G2::CYExp)) }

    /// Return `true` if `self` is `CYExpRev`.
    pub fn is_cyexprev(&self) -> bool { matches!(self, Self::Q2(G2::CYExpRev)) }

    /// Return `true` if `self` is `CZExp`.
    pub fn is_czexp(&self) -> bool { matches!(self, Self::Q2(G2::CZExp)) }

    /// Return `true` if `self` is `MS`.
    pub fn is_ms(&self) -> bool { matches!(self, Self::Q2(G2::MS)) }

    /// Return `true` if `self` is `SqrtSwap`.
    pub fn is_sqrtswap(&self) -> bool { matches!(self, Self::Q2(G2::SqrtSwap)) }

    /// Return `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Q2(G2::Swap)) }

    /// Return `true` if `self` is `Cliff2`.
    pub fn is_cliff2(&self) -> bool { matches!(self, Self::Q2(G2::Cliff2)) }

    /// Return `true` if `self` is `Haar2`.
    pub fn is_haar2(&self) -> bool { matches!(self, Self::Q2(G2::Haar2)) }
}

// /// An exact one- or two-qubit unitary.
// #[derive(Clone, Debug, PartialEq)]
// pub enum ExactGate {
//     /// Array is 2x2, applied to a single qubit.
//     Q1(usize, na::DMatrix<C64>),
//     /// Array is 4x4, applied to neighboring qubits with the index identifying
//     /// the left.
//     Q2(usize, na::DMatrix<C64>),
// }

/// Make an identity gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`IDMAT`] instead.
pub fn make_id<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::DMatrix::identity(2, 2)
}

/// Lazy-static version of [`make_id`] for a [`Complex64`][C64] element type.
pub static IDMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_id);

/// Make a projection matrix that takes the ∣0⟩ component of a qubit.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`PROJ0MAT`] instead.
pub fn make_proj0<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero();
        A::zero(), A::zero();
    ]
}

/// Lazy-static version of [`make_proj0`] for a [`Complex64`][C64] element type.
pub static PROJ0MAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_proj0);

/// Make a projection matrix that takes the ∣1⟩ component of a qubit.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`PROJ1MAT`] instead.
pub fn make_proj1<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::zero(), A::zero();
        A::zero(), A::one();
    ]
}

/// Lazy-static version of [`make_proj1`] for a [`Complex64`][C64] element type.
pub static PROJ1MAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_proj1);

/// Make a Hadamard gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`HMAT`] instead.
pub fn make_h<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    let h = A::from_re((A::Re::one() + A::Re::one()).recip().sqrt());
    na::dmatrix![
        h,  h;
        h, -h;
    ]
}

/// Lazy-static version of [`make_h`] for a [`Complex64`][C64] element type.
pub static HMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_h);

/// Make an X gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`XMAT`] instead.
pub fn make_x<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(), A::zero();
        A::zero(), A::one();
    ]
}

/// Lazy-static version of [`make_x`] for a [`Complex64`][C64] element type.
pub static XMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_x);

/// Make a Y gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`YMAT`] instead.
pub fn make_y<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::zero(), -A::i();
        A::i(),  A::zero();
    ]
}

/// Lazy-static version of [`make_y`] for a [`Complex64`][C64] element type.
pub static YMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_y);

/// Make a Z gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`ZMAT`] instead.
pub fn make_z<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),   A::zero();
        A::zero(), -A::one();
    ]
}

/// Lazy-static version of [`make_z`] for a [`Complex64`][C64] element type.
pub static ZMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_z);

/// Make a T gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`TMAT`] instead.
pub fn make_t<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero();
        A::zero(), na::ComplexField::sqrt(A::i());
    ]
}

/// Lazy-static version of [`make_t`] for a [`Complex64`][C64] element type.
pub static TMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_t);

/// Make a T<sup>†</sup> gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`TINVMAT`] instead.
pub fn make_tinv<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero();
        A::zero(), na::ComplexField::sqrt(-A::i());
    ]
}

/// Lazy-static version of [`make_tinv`] for a [`Complex64`][C64] element type.
pub static TINVMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_tinv);

/// Make an S gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`SMAT`] instead.
pub fn make_s<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero();
        A::zero(), A::i();
    ]
}

/// Lazy-static version of [`make_s`] for a [`Complex64`][C64] element type.
pub static SMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_s);

/// Make an S<sup>†</sup> gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`SINVMAT`] instead.
pub fn make_sinv<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),   A::zero();
        A::zero(), -A::i();
    ]
}

/// Lazy-static version of [`make_sinv`] for a [`Complex64`][C64] element type.
pub static SINVMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_sinv);

/// Make an X-rotation gate.
pub fn make_xrot<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let prefactor = A::cis(ang2);
    let ondiag = A::from_re(ang2.cos());
    let offdiag = -A::i() * A::from_re(ang2.sin());
    na::dmatrix![
        prefactor * ondiag,  prefactor * offdiag;
        prefactor * offdiag, prefactor * ondiag;
    ]
}

/// Make a Y-rotation gate.
pub fn make_yrot<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let prefactor = A::cis(ang2);
    let ondiag = A::from_re(ang2.cos());
    let offdiag = A::from_re(ang2.sin());
    na::dmatrix![
        prefactor * ondiag, -prefactor * offdiag;
        prefactor * offdiag, prefactor * ondiag;
    ]
}

/// Make a Z-rotation gate.
pub fn make_zrot<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let ph = A::cis(angle);
    na::dmatrix![
        A::one(),  A::zero();
        A::zero(), ph;
    ]
}

/// Make an X-exponentiation gate.
pub fn make_xexp<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let ondiag = A::from_re(ang2.cos());
    let offdiag = -A::i() * A::from_re(ang2.sin());
    na::dmatrix![
        ondiag,  offdiag;
        offdiag, ondiag;
    ]
}

/// Make a Y-exponentiation gate.
pub fn make_yexp<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let ondiag = A::from_re(ang2.cos());
    let offdiag = A::from_re(ang2.sin());
    na::dmatrix![
        ondiag, -offdiag;
        offdiag, ondiag;
    ]
}

/// Make a Z-exponentiation gate.
pub fn make_zexp<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let ph = A::cis(angle / (A::Re::one() + A::Re::one()));
    na::dmatrix![
        ph.conj(), A::zero();
        A::zero(), ph;
    ]
}

/// Make a single-qubit unitary from its Euler angles.
///
/// This gate is equivalent to `Z(γ) × X(β) × Z(α)`.
pub fn make_u<A>(alpha: A::Re, beta: A::Re, gamma: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let b2 = beta / (A::Re::one() + A::Re::one());
    let ag = alpha + gamma;
    let prefactor = A::cis(b2);
    let ondiag0 = A::from_re(b2.cos());
    let ondiag1 = A::cis(ag) * ondiag0;
    let offdiag = A::from_re(b2.sin());
    let offdiag0 = -A::i() * A::cis(gamma) * offdiag;
    let offdiag1 = -A::i() * A::cis(alpha) * offdiag;
    na::dmatrix![
        prefactor * ondiag0,  prefactor * offdiag0;
        prefactor * offdiag1, prefactor * ondiag1;
    ]
}

/// Make a CX gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`CXMAT`] instead.
pub fn make_cx<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero(), A::zero(), A::zero();
        A::zero(), A::zero(), A::zero(), A::one();
        A::zero(), A::zero(), A::one(),  A::zero();
        A::zero(), A::one(),  A::zero(), A::zero();
    ]
}

/// Lazy-static version of [`make_cx`] for a [`Complex64`][C64] element type.
pub static CXMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_cx);

/// Make a CX gate with the control and target qubits reversed.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`CXREVMAT`] instead.
pub fn make_cxrev<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero(), A::zero(), A::zero();
        A::zero(), A::one(),  A::zero(), A::zero();
        A::zero(), A::zero(), A::zero(), A::one();
        A::zero(), A::zero(), A::one(),  A::zero();
    ]
}

/// Lazy-static version of [`make_cxrev`] for a [`Complex64`][C64] element type.
pub static CXREVMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_cxrev);

/// Make a CY gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`CYMAT`] instead.
pub fn make_cy<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero(), A::zero(),  A::zero();
        A::zero(), A::zero(), A::zero(), -A::i();
        A::zero(), A::zero(), A::one(),   A::zero();
        A::zero(), A::i(),    A::zero(),  A::zero();
    ]
}

/// Lazy-static version of [`make_cy`] for a [`Complex64`][C64] element type.
pub static CYMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_cy);

/// Make a CY gate with the control and target qubits reversed.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`CYREVMAT`] instead.
pub fn make_cyrev<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero(), A::zero(),  A::zero();
        A::zero(), A::one(),  A::zero(),  A::zero();
        A::zero(), A::zero(), A::zero(), -A::i();
        A::zero(), A::zero(), A::i(),     A::zero();
    ]
}

/// Lazy-static version of [`make_cyrev`] for a [`Complex64`][C64] element type.
pub static CYREVMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_cyrev);

/// Make a CZ gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`CZMAT`] instead.
pub fn make_cz<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero(), A::zero(),  A::zero();
        A::zero(), A::one(),  A::zero(),  A::zero();
        A::zero(), A::zero(), A::one(),   A::zero();
        A::zero(), A::zero(), A::zero(), -A::one();
    ]
}

/// Lazy-static version of [`make_cz`] for a [`Complex64`][C64] element type.
pub static CZMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_cz);

/// Make a controlled X-rotation gate.
pub fn make_cxrot<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let prefactor = A::cis(ang2);
    let ondiag = A::from_re(ang2.cos());
    let offdiag = -A::i() * A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero,                zero, zero;
        zero, prefactor * ondiag,  zero, prefactor * offdiag;
        zero, zero,                one,  zero;
        zero, prefactor * offdiag, zero, prefactor * ondiag;
    ]
}

/// Make a controlled X-rotation gate with the control and target qubits
/// reversed.
pub fn make_cxrotrev<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let prefactor = A::cis(ang2);
    let ondiag = A::from_re(ang2.cos());
    let offdiag = -A::i() * A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero, zero,                zero;
        zero, one,  zero,                zero;
        zero, zero, prefactor * ondiag,  prefactor * offdiag;
        zero, zero, prefactor * offdiag, prefactor * ondiag;
    ]
}

/// Make a controlled Y-rotation gate.
pub fn make_cyrot<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let prefactor = A::cis(ang2);
    let ondiag = A::from_re(ang2.cos());
    let offdiag = A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero,                zero,  zero;
        zero, prefactor * ondiag,  zero, -prefactor * offdiag;
        zero, zero,                one,   zero;
        zero, prefactor * offdiag, zero,  prefactor * ondiag;
    ]
}

/// Make a controlled Y-rotation gate with the control and target qubits
/// reversed.
pub fn make_cyrotrev<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let prefactor = A::cis(ang2);
    let ondiag = A::from_re(ang2.cos());
    let offdiag = A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero, zero,                zero;
        zero, one,  zero,                zero;
        zero, zero, prefactor * ondiag, -prefactor * offdiag;
        zero, zero, prefactor * offdiag, prefactor * ondiag;
    ]
}

/// Make a controlled Z-rotation gate.
pub fn make_czrot<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ph = A::cis(angle);
    na::dmatrix![
        one,  zero, zero, zero;
        zero, one,  zero, zero;
        zero, zero, one,  zero;
        zero, zero, zero, ph;
    ]
}

/// Make a controlled X-exponentiation gate.
pub fn make_cxexp<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let ondiag = A::from_re(ang2.cos());
    let offdiag = -A::i() * A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero,    zero, zero;
        zero, ondiag,  zero, offdiag;
        zero, zero,    one,  zero;
        zero, offdiag, zero, ondiag;
    ]
}

/// Make a controlled X-exponentiation gate with the control and target qubits
/// reversed.
pub fn make_cxexprev<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let ondiag = A::from_re(ang2.cos());
    let offdiag = -A::i() * A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero, zero,    zero;
        zero, one,  zero,    zero;
        zero, zero, ondiag,  offdiag;
        zero, zero, offdiag, ondiag;
    ]
}

/// Make a controlled Y-exponentiation gate.
pub fn make_cyexp<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let ondiag = A::from_re(ang2.cos());
    let offdiag = A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero,    zero,  zero;
        zero, ondiag,  zero, -offdiag;
        zero, zero,    one,   zero;
        zero, offdiag, zero,  ondiag;
    ]
}

/// Make a controlled Y-exponentiation gate with the control and target qubits
/// reversed.
pub fn make_cyexprev<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ang2 = angle / (A::Re::one() + A::Re::one());
    let ondiag = A::from_re(ang2.cos());
    let offdiag = A::from_re(ang2.sin());
    na::dmatrix![
        one,  zero, zero,     zero;
        zero, one,  zero,     zero;
        zero, zero, ondiag,  -offdiag;
        zero, zero, offdiag,  ondiag;
    ]
}

/// Make a controlled Z-exponentiation gate.
pub fn make_czexp<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ph = A::cis(angle / (A::Re::one() + A::Re::one()));
    na::dmatrix![
        one,  zero,      zero, zero;
        zero, ph.conj(), zero, zero;
        zero, zero,      one,  zero;
        zero, zero,      zero, ph;
    ]
}

/// Make a controlled Z-exponentiation gate with the control and target qubits
/// reversed.
pub fn make_czexprev<A>(angle: A::Re) -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let ph = A::cis(angle / (A::Re::one() + A::Re::one()));
    na::dmatrix![
        one,  zero, zero,      zero;
        zero, one,  zero,      zero;
        zero, zero, ph.conj(), zero;
        zero, zero, zero,      ph;
    ]
}

/// Make a Mølmer-Sørensen gate. This gate uses the *xx* definition.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`MSMAT`] instead.
pub fn make_ms<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    use num_complex::ComplexFloat;
    let zero = A::zero();
    let two = A::one() + A::one();
    let ort2 = ComplexFloat::recip(ComplexFloat::sqrt(two));
    let iort2 = A::i() * ort2;
    na::dmatrix![
        ort2, zero, zero, -iort2;
        zero, ort2, -iort2, zero;
        zero, -iort2, ort2, zero;
        -iort2, zero, zero, ort2;
    ]
}

/// Lazy-static version of [`make_ms`] for a [`Complex64`][C64] element type.
pub static MSMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_ms);

/// Make a root-swap gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`SQRTSWAPMAT`] instead.
pub fn make_sqrtswap<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    let zero = A::zero();
    let one = A::one();
    let two = one + one;
    let t = (one + A::i()) / two;
    na::dmatrix![
        one,  zero,     zero,     zero;
        zero, t,        t.conj(), zero;
        zero, t.conj(), t,        zero;
        zero, zero,     zero,     one;
    ]
}

/// Lazy-static version of [`make_sqrtswap`] for a [`Complex64`][C64] element
/// type.
pub static SQRTSWAPMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_sqrtswap);

/// Make a swap gate.
///
/// Since this gate takes no arguments, consider using the lazily constructed,
/// [`Complex64`][C64]-valued [`SWAPMAT`] instead.
pub fn make_swap<A>() -> na::DMatrix<A>
where A: ComplexScalar
{
    na::dmatrix![
        A::one(),  A::zero(), A::zero(), A::zero();
        A::zero(), A::zero(), A::one(),  A::zero();
        A::zero(), A::one(),  A::zero(), A::zero();
        A::zero(), A::zero(), A::zero(), A::one();
    ]
}

/// Lazy-static version of [`make_swap`] for a [`Complex64`][C64] element type.
pub static SWAPMAT: Lazy<na::DMatrix<C64>> = Lazy::new(make_swap);

/// Generate a random element of the `n`-qubit Clifford group as a matrix.
pub fn make_cliff<A, R>(n: usize, rng: &mut R) -> na::DMatrix<A>
where
    A: ComplexScalar,
    R: Rng + ?Sized,
{
    use num_complex::ComplexFloat;
    if n == 1 {
        let zero = A::zero();
        let one = A::one();
        let two = one + one;
        let ort2 = ComplexFloat::recip(ComplexFloat::sqrt(two));
        let i = A::i();
        let iort2 = i * ort2;
        let t = (one + i) / two;
        match rng.gen_range(0..24) {
            0  => na::DMatrix::identity(2, 2),                      // I
            1  => make_h(),                                         // H
            2  => make_s(),                                         // S
            3  => make_x(),                                         // X
            4  => make_y(),                                         // Y
            5  => make_z(),                                         // Z
            6  => na::dmatrix!(ort2, -ort2; ort2, ort2),            // X H
            7  => na::dmatrix!(-iort2, iort2; iort2, iort2),        // Y H
            8  => na::dmatrix!(ort2, ort2; -ort2, ort2),            // Z H
            9  => na::dmatrix!(ort2, iort2; ort2, -iort2),          // H S
            10 => na::dmatrix!(zero, i; one, zero),                 // X S
            11 => na::dmatrix!(zero, one; i, zero),                 // Y S
            12 => na::dmatrix!(one, zero; zero, -i),                // Z S
            13 => na::dmatrix!(t, t.conj(); t.conj(), t),           // H S H
            14 => na::dmatrix!(ort2, -iort2; ort2, iort2),          // X H S
            15 => na::dmatrix!(-iort2, -ort2; iort2, -ort2),         // Y H S
            16 => na::dmatrix!(ort2, iort2; -ort2, iort2),          // Z H S
            17 => na::dmatrix!(t, i * t.conj(); t.conj(), i * t),   // H S H S
            18 => na::dmatrix!(t.conj(), t; t, t.conj()),           // X H S H
            19 => na::dmatrix!(-t, t.conj(); -t.conj(), t),         // Y H S H
            20 => na::dmatrix!(t, t.conj(); -t.conj(), -t),         // Z H S H
            21 => na::dmatrix!(t.conj(), i * t; t, i * t.conj()),   // X H S H S
            22 => na::dmatrix!(-t, t.conj(); -t.conj(), t),         // Y H S H S
            23 => na::dmatrix!(t, i * t.conj(); -t.conj(), -i * t), // Z H S H S
            _ => unreachable!(),
        }
    } else {
        Clifford::gen(n, rng)
            .unpack().0
            .into_iter()
            .map(|cg| cg.into_matrix::<A>(n))
            .fold(qembed(n, []), |acc, mat| mat * acc)
    }
}

/// Generate an `n`-qubit Haar-random unitary matrix.
pub fn make_haar<A, R>(n: usize, rng: &mut R) -> na::DMatrix<A>
where
    A: ComplexScalar,
    Normal: Distribution<<A as ComplexScalar>::Re>,
    R: Rng + ?Sized,
{
    let normal = Normal::standard();
    let z: na::DMatrix<A> =
        na::DMatrix::from_fn(
            1_usize << n as u32, 1_usize << n as u32,
            |_, _| A::from_components(normal.sample(rng), normal.sample(rng)),
        );
    let (mut q, r) = z.qr().unpack();
    q.column_iter_mut().zip(&r.diagonal())
        .for_each(|(mut z_j, rjj)| {
            let renorm = *rjj / A::from_re(rjj.modulus());
            z_j.iter_mut().for_each(|zij| { *zij /= renorm; });
        });
    q
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
    /// The left qubit index is the control.
    CX(usize, usize),
    /// Z-controlled π rotation about Z.
    ///
    /// The left qubit index is the control.
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

    fn into_matrix<A>(self, n: usize) -> na::DMatrix<A>
    where A: ComplexScalar
    {
        match self {
            Self::H(k) => qembed(n, [(k, make_h())]),
            Self::X(k) => qembed(n, [(k, make_x())]),
            Self::Y(k) => qembed(n, [(k, make_y())]),
            Self::Z(k) => qembed(n, [(k, make_z())]),
            Self::S(k) => qembed(n, [(k, make_s())]),
            Self::SInv(k) => qembed(n, [(k, make_sinv())]),
            Self::CX(c, t) => {
                let p0 = qembed(n, [(c, make_proj0())]);
                let p1 = qembed(
                    n,
                    if c < t { [(c, make_proj1()), (t, make_x())] }
                    else { [(t, make_x()), (c, make_proj1())] },
                );
                p0 + p1
            },
            Self::CZ(a, b) => {
                let p0 = qembed(n, [(a, make_proj0())]);
                let p1 = qembed(n, [(a, make_proj1()), (b, make_z())]);
                p0 + p1
            },
            Self::Swap(a, b) => {
                let ii = na::DMatrix::<A>::identity(
                    1_usize << n as u32, 1_usize << n as u32);
                let xx = qembed(n, [(a, make_x()), (b, make_x())]);
                let yy = qembed(n, [(a, make_y()), (b, make_y())]);
                let zz = qembed(n, [(a, make_z()), (b, make_z())]);
                (ii + xx + yy + zz) / (A::one() + A::one())
            },
        }
    }
}

// requires that qubit indices be in ascending order, all operators are
// single-qubit, and that all indices are less than `n`
//
// since nalgebra is column-major, we accumulate tensor products from the right
fn qembed<I, A>(n: usize, ops: I) -> na::DMatrix<A>
where
    I: IntoIterator<Item = (usize, na::DMatrix<A>)>,
    A: ComplexScalar,
{
    let mut last: Option<usize> = None;
    let mut acc: na::DMatrix<A> = na::dmatrix!(A::one());
    for (j, mat) in ops.into_iter() {
        if let Some(k) = last.as_mut() {
            if j - *k > 1 {
                let dq = (j - 1 - *k) as u32;
                acc =
                    mat.kronecker(&na::DMatrix::identity(1 << dq, 1 << dq))
                    .kronecker(&acc);
            } else {
                acc = mat.kronecker(&acc);
            }
            *k = j;
        } else {
            let dq = j as u32;
            acc =
                mat.kronecker(&na::DMatrix::identity(1 << dq, 1 << dq))
                .kronecker(&acc);
            last = Some(j);
        }
    }
    if let Some(k) = last {
        if k < n - 1 {
            let dq = (n - 1 - k) as u32;
            acc = na::DMatrix::identity(1 << dq, 1 << dq).kronecker(&acc);
        }
    } else {
        let dq = n as u32;
        acc = na::DMatrix::identity(1 << dq, 1 << dq);
    }
    acc
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

    #[allow(dead_code)]
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

    /// Unpack `self` into a bare pair of gates and a number of qubits.
    pub fn unpack(self) -> (Vec<CliffGate>, usize) { (self.gates, self.n) }

    /// Sample a random element of the `n`-qubit Clifford group.
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

