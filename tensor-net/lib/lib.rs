#![allow(non_snake_case)]

//! Tools for simulating measurement-induced phase transitions in registers of
//! qubits using tensor networks.

use nalgebra as na;
use num_complex::{ ComplexFloat, Complex };
use num_traits::{ Float, Zero };

// pub mod tensor;
// pub mod network;
// pub mod pool;
//
// pub mod mps;
// pub mod pmps;
// pub mod lazymps;
// pub mod gate;
// pub mod circuit;

pub mod id_set;
pub mod tensor;
pub mod network;

pub mod mps;
pub mod gate;
pub mod circuit;

/// Extension trait for [`ComplexFloat`].
pub trait ComplexFloatExt: ComplexFloat /*+ std::fmt::Debug*/
/*where <Self as ComplexFloat>::Real: std::fmt::Debug*/
{
    // /// The imaginary unit, *i*.
    // const I: Self;
    //
    // /// The additive identity, 0.
    // const ZERO: Self;
    //
    // /// The multiplicative identity, 1.
    // const ONE: Self;

    /// Return the imaginary unit, *i*.
    fn i() -> Self;

    /// Convert from `Self::Real`.
    ///
    /// Should adhere to the usual relationship between ordinary complex and
    /// real numbers, i.e. the result should have imaginary part equal to zero.
    fn from_re(x: Self::Real) -> Self;

    /// Construct from real and imaginary components.
    fn from_components(re: Self::Real, im: Self::Real) -> Self;

    /// Create a new value of unit magnitude with a given phase angle.
    fn cis(angle: Self::Real) -> Self;

    /// Convert to a polar representation `(r, θ)`.
    fn to_polar(self) -> (Self::Real, Self::Real);

    /// Convert a polar representation into a complex number.
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;

    #[cfg(feature = "openblas")]
    /// Construct from real and imaginary components encoded in `f64`s.
    fn from_f64(re: f64, im: f64) -> Self;

    #[cfg(feature = "openblas")]
    /// Return the real component encoded in an `f64`.
    fn real_f64(self) -> f64;

    #[cfg(feature = "openblas")]
    /// Return the imaginary component encoded in an `f64`.
    fn imag_f64(self) -> f64;

    #[cfg(feature = "openblas")]
    /// Convert a real value to a `f64`.
    fn real_into_f64(re: Self::Real) -> f64;

    #[cfg(feature = "openblas")]
    /// Convert a `f64` to `Self::Real`.
    fn real_from_f64(re: f64) -> Self::Real;
}

#[cfg(not(feature = "openblas"))]
impl<T> ComplexFloatExt for Complex<T>
where
    Complex<T>: ComplexFloat<Real = T>,
    T: Zero + Float /*+ std::fmt::Debug*/,
{
    // const I: Self = Complex::<T>::I;
    // const ZERO: Self = Complex::<T>::ZERO;
    // const ONE: Self = Complex::<T>::ONE;

    fn i() -> Self { Complex::i() }

    fn from_re(x: Self::Real) -> Self {
        Self { re: x, im: <Self::Real as Zero>::zero() }
    }

    fn from_components(re: Self::Real, im: Self::Real) -> Self {
        Self { re, im }
    }

    fn cis(angle: Self::Real) -> Self { Complex::cis(angle) }

    fn to_polar(self) -> (Self::Real, Self::Real) {
        self.to_polar()
    }

    fn from_polar(r: Self::Real, theta: Self::Real) -> Self {
        Complex::from_polar(r, theta)
    }
}

#[cfg(feature = "openblas")]
impl ComplexFloatExt for Complex<f32> {
    fn i() -> Self { Self::i() }

    fn from_re(x: f32) -> Self { Self { re: x, im: f32::zero() } }

    fn from_components(re: f32, im: f32) -> Self { Self { re, im } }

    fn cis(angle: f32) -> Self { Self::cis(angle) }

    fn to_polar(self) -> (f32, f32) { self.to_polar() }

    fn from_polar(r: f32, theta: f32) -> Self { Self::from_polar(r, theta) }

    fn from_f64(re: f64, im: f64) -> Self {
        Self { re: re as f32, im: im as f32 }
    }

    fn real_f64(self) -> f64 { self.re.into() }

    fn imag_f64(self) -> f64 { self.im.into() }

    fn real_into_f64(re: f32) -> f64 { re.into() }

    fn real_from_f64(re: f64) -> f32 { re as f32 }
}

#[cfg(feature = "openblas")]
impl ComplexFloatExt for Complex<f64> {
    fn i() -> Self { Self::i() }

    fn from_re(x: f64) -> Self { Self { re: x, im: f64::zero() } }

    fn from_components(re: f64, im: f64) -> Self { Self { re, im } }

    fn cis(angle: f64) -> Self { Self::cis(angle) }

    fn to_polar(self) -> (f64, f64) { self.to_polar() }

    fn from_polar(r: f64, theta: f64) -> Self { Self::from_polar(r, theta) }

    fn from_f64(re: f64, im: f64) -> Self { Self { re, im } }

    fn real_f64(self) -> f64 { self.re }

    fn imag_f64(self) -> f64 { self.im }

    fn real_into_f64(re: f64) -> f64 { re }

    fn real_from_f64(re: f64) -> f64 { re }
}

// NOTE: Dependence on ComplexFloat seems to be mostly redundant via
// ComplexField -- should check this

/// Convenience trait to identity complex number types that can be used in
/// linear-algebraic operations.
pub trait ComplexScalar
where
    Self:
        ComplexFloat<Real = Self::Re>
        + ComplexFloatExt
        + na::ComplexField<RealField = Self::Re>
{
    /// Type for associated real values.
    type Re: Float + na::RealField;
}

impl<A> ComplexScalar for A
where
    A:
        ComplexFloat<Real = <A as na::ComplexField>::RealField>
        + ComplexFloatExt
        + na::ComplexField,
    <A as na::ComplexField>::RealField: Float,
{
    type Re = <A as na::ComplexField>::RealField;
}

