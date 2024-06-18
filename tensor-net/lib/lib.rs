#![allow(dead_code, non_snake_case, non_upper_case_globals)]

//! Tools for simulating measurement-induced phase transitions in registers of
//! qubits using tensor networks.

use num_complex::{ ComplexFloat, Complex };
use num_traits::{ Float, Zero };

pub mod tensor;
pub mod network;
pub mod pool;

pub mod mps;
pub mod gate;
pub mod circuit;

/// Extension trait for [`ComplexFloat`].
pub trait ComplexFloatExt: ComplexFloat {
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
    /// real numbers.
    fn from_real(x: Self::Real) -> Self;

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
    T: Zero + Float,
{
    // const I: Self = Complex::<T>::I;
    // const ZERO: Self = Complex::<T>::ZERO;
    // const ONE: Self = Complex::<T>::ONE;

    fn i() -> Self { Complex::i() }

    fn from_real(x: Self::Real) -> Self {
        Self { re: x, im: <Self::Real as Zero>::zero() }
    }

    fn cis(angle: Self::Real) -> Self { Complex::cis(angle) }

    fn to_polar(self) -> (Self::Real, Self::Real) {
        self.to_polar()
    }

    fn from_polar(r: Self::Real, theta: Self::Real) -> Self {
        Complex::from_polar(r, theta)
    }
}

