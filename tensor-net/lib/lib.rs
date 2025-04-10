#![allow(dead_code, non_snake_case, non_upper_case_globals)]

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

pub mod tensor_na;

pub mod mps_na;
pub mod gate_na;
pub mod circuit_na;

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
}

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

