use std::ops::Range;
use rand::Rng;

/// Return the range of qubits in the reverse light cone of a target measurement
/// on qubit index `x` (out of `n` total qubits) occurring `dt` layers from
/// layer `t`.
pub fn rev_cone_range(n: usize, x: usize, t: usize, dt: usize) -> Range<usize> {
    if (t + x) % 2 == 0 {
        let start = x.saturating_sub(2 * (dt / 2));
        let end = (x + 2 * ((dt + 1) / 2)).min(n);
        start .. end
    } else {
        let start = x.saturating_sub(2 * ((dt + 1) / 2)) + 1;
        let end = (x + 2 * (dt / 2) + 1).min(n);
        start .. end
    }
}

/// A single qubit measurement operation, either naive or post-selected.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Meas {
    Rand(usize),
    Postsel(usize, bool),
}

impl Meas {
    /// Return the target qubit index.
    pub fn idx(&self) -> usize {
        match self {
            Self::Rand(k) => *k,
            Self::Postsel(k, _) => *k,
        }
    }
}

/// A single layer of measurement locations.
#[derive(Clone, Debug)]
pub struct MeasLayer(Vec<Meas>);

impl MeasLayer {
    /// Get a reference to all measurement operations in the layer.
    pub fn get(&self) -> &Vec<Meas> { &self.0 }

    /// Generate a layer of (`Rand`) measurements whose locations are chosen
    /// uniformly with probability `o`.
    pub fn gen<R>(n: usize, p: f64, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let data: Vec<Meas> =
            (0..n)
            .filter(|_| rng.gen::<f64>() < p)
            .map(Meas::Rand)
            .collect();
        Self(data)
    }
}

impl AsRef<Vec<Meas>> for MeasLayer {
    fn as_ref(&self) -> &Vec<Meas> { self.get() }
}

pub mod clifford;
pub mod haar;

pub mod systems;

