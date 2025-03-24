use ndarray::{ self as nd, linalg::kron };
use num_complex::Complex64 as C64;
use rand::Rng;
use clifford_sim::{
    gate::{ Clifford, Gate },
    stab::{ Stab, Postsel },
};
use tensor_net::{
    circuit::{ Q, TileQ2 },
    mps::MPS,
};
use crate::{ Meas, MeasLayer };

/// Convert a [`Gate`] to an ordinary two-qubit unitary.
///
/// The `Gate` must operate on a qubit index that is either `0` or `1`;
/// otherwise this function will panic.
pub fn gate_matrix(gate: &Gate) -> nd::Array2<C64> {
    use std::f64::consts::FRAC_1_SQRT_2;
    const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    const I: C64 = C64 { re: 0.0, im: 1.0 };
    const ORT2: C64 = C64 { re: FRAC_1_SQRT_2, im: 0.0 };
    let eye: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, ONE]];
    match gate {
        Gate::H(k) => {
            let mat: nd::Array2<C64> = nd::array![[ORT2, ORT2], [ORT2, -ORT2]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!("invalid qubit operand index {k}")
            }
        },
        Gate::X(k) => {
            let mat: nd::Array2<C64> = nd::array![[ZERO, ONE], [ONE, ZERO]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!("invalid qubit operand index {k}")
            }
        },
        Gate::Y(k) => {
            let mat: nd::Array2<C64> = nd::array![[ZERO, -I], [I, ZERO]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!("invalid qubit operand index {k}")
            }
        },
        Gate::Z(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, -ONE]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!("invalid qubit operand index {k}")
            }
        },
        Gate::S(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, I]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!("invalid qubit operand index {k}")
            }
        },
        Gate::SInv(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, -I]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!("invalid qubit operand index {k}")
            }
        },
        Gate::CX(a, b) => {
            if *a == 0 && *b == 1 {
                nd::array![
                    [ONE, ZERO, ZERO, ZERO],
                    [ZERO, ONE, ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE],
                    [ZERO, ZERO, ONE, ZERO],
                ]
            } else if *a == 1 && *b == 0 {
                nd::array![
                    [ONE, ZERO, ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE],
                    [ZERO, ZERO, ONE, ZERO],
                    [ZERO, ONE, ZERO, ZERO],
                ]
            } else {
                panic!("invalid qubit operand indicies {a}, {b}")
            }
        },
        Gate::CZ(a, b) => {
            if (*a == 0 && *b == 1) || (*a == 1 && *b == 0) {
                nd::array![
                    [ONE,  ZERO, ZERO,  ZERO],
                    [ZERO, ONE,  ZERO,  ZERO],
                    [ZERO, ZERO, ONE,   ZERO],
                    [ZERO, ZERO, ZERO, -ONE ],
                ]
            } else {
                panic!("invalid qubit operand indicies {a}, {b}")
            }
        },
        Gate::Swap(a, b) => {
            if (*a == 0 && *b == 1) || (*a == 1 && *b == 0) {
                nd::array![
                    [ONE, ZERO, ZERO, ZERO],
                    [ZERO, ZERO, ONE, ZERO],
                    [ZERO, ONE, ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE],
                ]
            } else {
                panic!("invalid qubit operand indicies {a}, {b}")
            }
        },
    }
}

/// A single unitary comprising several [`Gate`]s and their equivalent matrix.
#[derive(Clone, Debug)]
pub struct Uni {
    gates: Vec<Gate>,
    mat: (usize, nd::Array2<C64>),
}

impl Uni {
    /// Get a reference to the source [`Gate`] decomposition.
    pub fn gates(&self) -> &Vec<Gate> { &self.gates }

    /// Get a reference to the unitary matrix.
    pub fn mat(&self) -> (usize, &nd::Array2<C64>) { (self.mat.0, &self.mat.1) }
}

/// A single layer of [`Uni`]s.
#[derive(Clone, Debug)]
pub struct UniLayer(Vec<Uni>);

impl UniLayer {
    /// Get a reference to each [`Uni`] in the layer.
    pub fn get(&self) -> &Vec<Uni> { &self.0 }

    /// Generate a brickwork layer of (uniformly) random Clifford gates.
    pub fn gen<R>(n: usize, offs: bool, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let data: Vec<Uni> =
            TileQ2::new(offs, n)
            .map(|k| {
                let mut gates: Vec<Gate> =
                    Clifford::gen(2, rng).unpack().0;
                let mat: nd::Array2<C64> =
                    gates.iter()
                    .map(gate_matrix)
                    .fold(nd::Array2::eye(4), |acc, gate| gate.dot(&acc));
                gates.iter_mut().for_each(|g| { g.shift(k); });
                Uni { gates, mat: (k, mat) }
            })
            .collect();
        Self(data)
    }
}

/// Apply a single pair of unitary and measurement layers to a [`MPS`], with the
/// unitary layer applied first.
///
/// Measurement outcomes are returned as a list of all [`Meas::Postsel`].
pub fn apply_main_layer_mps<R>(
    state: &mut MPS<Q, C64>,
    unis: &UniLayer,
    meas: &MeasLayer,
    rng: &mut R,
) -> Vec<Meas>
where R: Rng + ?Sized
{
    unis.get().iter()
        .for_each(|uni| {
            let (k, mat) = uni.mat();
            state.apply_unitary2(k, mat).unwrap();
        });
    meas.get().iter()
        .map(|m| {
            match m {
                Meas::Rand(k) => {
                    let out: bool = state.measure(*k, rng).unwrap() == 1;
                    Meas::Postsel(*k, out)
                },
                Meas::Postsel(k, out) => {
                    state.measure_postsel(*k, *out as usize);
                    Meas::Postsel(*k, *out)
                },
            }
        })
        .collect()
}

/// Apply a single pair of unitary and measurement layers to a [`MPS`], with the
/// unitary layer applied first.
///
/// If `target` is `Some`, then no measurements are applied to the inner qubit
/// index; the single-qubit state probabilities at that site are instead
/// returned alongside measurement outcomes.
pub fn apply_probe_layer_mps<R>(
    state: &mut MPS<Q, C64>,
    unis: &UniLayer,
    meas: &MeasLayer,
    target: Option<usize>,
    rng: &mut R,
) -> (Vec<Meas>, Option<(f64, f64)>)
where R: Rng + ?Sized
{
    unis.get().iter()
        .for_each(|uni| {
            let (k, mat) = uni.mat();
            state.apply_unitary2(k, mat).unwrap();
        });
    let outs: Vec<Meas> =
        meas.get().iter()
            .filter(|m| !target.is_some_and(|j| j == m.idx()))
            .map(|m| {
                match m {
                    Meas::Rand(k) => {
                        let out: bool = state.measure(*k, rng).unwrap() == 1;
                        Meas::Postsel(*k, out)
                    },
                    Meas::Postsel(k, out) => {
                        state.measure_postsel(*k, *out as usize);
                        Meas::Postsel(*k, *out)
                    },
                }
            })
            .collect();
    let probs: Option<(f64, f64)> =
        target.and_then(|k| state.prob(k, 0).zip(state.prob(k, 1)));
    (outs, probs)
}

/// Apply a single pair of unitary and measurement layers to a [`Stab`], with
/// the unitary layer applied first.
///
/// Measurement outcomes are returned as a list of all [`Meas::Postsel`].
pub fn apply_main_layer_stab<R>(
    state: &mut Stab,
    unis: &UniLayer,
    meas: &MeasLayer,
    rng: &mut R,
) -> Vec<Meas>
where R: Rng + ?Sized
{
    unis.get().iter()
        .for_each(|uni| {
            let gates = uni.gates();
            state.apply_circuit(gates);
        });
    meas.get().iter()
        .map(|m| {
            match m {
                Meas::Rand(k) => {
                    let out: bool = state.measure(*k, rng).is_1();
                    Meas::Postsel(*k, out)
                },
                Meas::Postsel(k, out) => {
                    let postsel = 
                        if *out { Postsel::One } else { Postsel::Zero };
                    state.measure_postsel_checked(*k, postsel).unwrap();
                    Meas::Postsel(*k, *out)
                },
            }
        })
        .collect()
}

/// Apply a single pair of unitary and measurement layers to a [`Stab`], with
/// the unitary layer applied first.
///
/// If `target` is `Some`, then no measurements are applied to the inner qubit
/// index; the single-qubit state probabilities at that site are instead
/// returned alongside measurement outcomes.
pub fn apply_probe_layer_stab<R>(
    state: &mut Stab,
    unis: &UniLayer,
    meas: &MeasLayer,
    target: Option<usize>,
    rng: &mut R,
) -> (Vec<Meas>, Option<(f64, f64)>)
where R: Rng + ?Sized
{
    unis.get().iter()
        .for_each(|uni| {
            let gates = uni.gates();
            state.apply_circuit(gates);
        });
    let outs: Vec<Meas> =
        meas.get().iter()
            .map(|m| {
                match m {
                    Meas::Rand(k) => {
                        let out: bool = state.measure(*k, rng).is_1();
                        Meas::Postsel(*k, out)
                    },
                    Meas::Postsel(k, out) => {
                        let postsel = 
                            if *out { Postsel::One } else { Postsel::Zero };
                        state.measure_postsel_checked(*k, postsel).unwrap();
                        Meas::Postsel(*k, *out)
                    },
                }
            })
            .collect();
    let probs: Option<(f64, f64)> =
        target.map(|k| state.probs(k).into());
    (outs, probs)
}

