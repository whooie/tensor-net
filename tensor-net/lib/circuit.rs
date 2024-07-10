//! Abstractions for driving randomized circuits on qubits and measureing
//! measurement-induced phase transitions (MIPTs).

#![allow(unused_imports)]

use std::ops::Range;
use rand::{ rngs::StdRng, Rng, SeedableRng };
use rustc_hash::FxHashSet as HashSet;
use num_complex::Complex64 as C64;
use crate::{
    gate::{ self, GateToken, Gate, G1, G2 },
    mps::{ BondDim, MPS },
    tensor::Idx,
};

/// Type alias for [`HashSet`].
pub type Set<T> = HashSet<T>;

/// [Index type][Idx] for qubits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Q(
    /// The qubit index.
    pub(crate) usize
);

impl From<usize> for Q {
    fn from(k: usize) -> Self { Self(k) }
}

impl From<Q> for usize {
    fn from(q: Q) -> Self { q.0 }
}

impl Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String { format!("q:{}", self.0) }
}

impl crate::tensor2::Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String { format!("q:{}", self.0) }
}

impl crate::tensor3::Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String { format!("q:{}", self.0) }
}

/// The result of a measurement.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// The outcomes associated with a single measurement layer.
pub type MeasLayer = Vec<Option<Outcome>>;

/// The outcomes from each measurement layer in a full circuit.
pub type MeasRecord = Vec<MeasLayer>;

/// Main driver for running circuits of alternating unitary evolution and
/// measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct MPSCircuit {
    pub state: MPS<Q, C64>,
    n: usize,
    rng: StdRng,
}

impl MPSCircuit {
    /// Create a new `MPSCircuit` for a 1D chain of `n` qubits, with state
    /// initialized to ∣0...0⟩.
    ///
    /// `trunc` is an optional global truncation method for discarding singular
    /// values, defaulting to a [`Cutoff`][BondDim::Cutoff] at machine epsilon.
    /// Optionally also seed the internal random number generator.
    ///
    /// *Panics* if `n` is zero.
    pub fn new(
        n: usize,
        trunc: Option<BondDim<f64>>,
        seed: Option<u64>,
    ) -> Self
    {
        if n == 0 { panic!("cannot run a circuit on an empty state"); }
        let rng
            = seed.map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);
        let state = MPS::new((0..n).map(Q), trunc).unwrap();
        Self { state, n, rng }
    }

    /// Unwrap `self` into the current state.
    pub fn into_state(self) -> MPS<Q, C64> { self.state }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.n }

    fn sample_simple(
        &mut self,
        offs: bool,
        buf: &mut Vec<Gate>,
    ) {
        (0..self.n).for_each(|k| {
            buf.push(G1::sample_random(k, &mut self.rng));
        });
        TileQ2::new(offs, self.n).for_each(|a| {
            buf.push(Gate::CX(a));
        });
    }

    fn apply_haars(&mut self, offs: bool) {
        TileQ2::new(offs, self.n).for_each(|a| {
            let u = gate::haar(2, &mut self.rng);
            self.state.apply_unitary2(a, &u).unwrap();
        });
    }

    fn sample_gateset(
        &mut self,
        g1: &G1Set,
        g2: &G2Set,
        offs: bool,
        buf: &mut Vec<Gate>,
    ) {
        (0..self.n).for_each(|k| {
            buf.push(g1.sample(k, &mut self.rng));
        });
        TileQ2::new(offs, self.n).for_each(|a| {
            buf.push(g2.sample(a, &mut self.rng));
        });
    }

    fn measure(
        &mut self,
        d: usize,
        config: MeasureConfig,
        buf: &mut [Option<Outcome>],
    ) -> bool
    {
        use MeasLayerConfig::*;
        use MeasProbConfig::*;

        enum Pred<'a> {
            Never,
            Always,
            Prob(f64),
            Func(Box<dyn Fn(usize) -> bool + 'a>),
        }

        fn do_measure(
            circ: &mut MPSCircuit,
            pred: Pred,
            buf: &mut [Option<Outcome>],
        ) {
            buf.iter_mut()
                .enumerate()
                .for_each(|(k, outk)| {
                    match &pred {
                        Pred::Never => {
                            *outk = None;
                        },
                        Pred::Always => {
                            *outk = circ.state.measure(k, &mut circ.rng)
                                .map(Outcome::from);
                        },
                        Pred::Prob(p) => {
                            *outk = if circ.rng.gen::<f64>() < *p {
                                circ.state.measure(k, &mut circ.rng)
                                    .map(Outcome::from)
                            } else {
                                None
                            };
                        },
                        Pred::Func(f) => {
                            *outk = if f(k) {
                                circ.state.measure(k, &mut circ.rng)
                                    .map(Outcome::from)
                            } else {
                                None
                            };
                        },
                    }
                });

            // buf.fill(None);
            // let to_measure: Vec<usize>
            //     = (0..buf.len())
            //     .filter(|k| {
            //         match &pred {
            //             Pred::Never => false,
            //             Pred::Always => true,
            //             Pred::Prob(p) => circ.rng.gen::<f64>() < *p,
            //             Pred::Func(f) => f(*k),
            //         }
            //     })
            //     .collect();
            // let outcomes: Vec<usize>
            //     = circ.state.measure_multi(
            //         to_measure.iter().copied(), &mut circ.rng);
            // to_measure.into_iter()
            //     .zip(outcomes)
            //     .for_each(|(k, outk)| { buf[k] = Some(Outcome::from(outk)); });
        }

        let MeasureConfig { layer, prob } = config;
        match layer {
            Every | Period(1) => {
                let pred
                    = match prob {
                        Random(p) => Pred::Prob(p),
                        Cycling(0) => Pred::Never,
                        Cycling(1) => Pred::Always,
                        Cycling(n) => Pred::Func(
                            Box::new(move |k| k % n == d % n)),
                        CyclingInv(0) => Pred::Always,
                        CyclingInv(1) => Pred::Never,
                        CyclingInv(n) => Pred::Func(
                            Box::new(move |k| k % n != d % n)),
                        Block(0) => Pred::Never,
                        Block(b) => Pred::Func(
                            Box::new(move |k| k / b == d % b)),
                        Window(0) => Pred::Never,
                        Window(w) => {
                            let d = d as isize;
                            let w = w as isize;
                            let n = self.n as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d).rem_euclid(n) / w == 0
                                })
                            )
                        },
                    };
                do_measure(self, pred, buf);
                true
            },
            Period(m) if d % m == 0 => {
                let pred
                    = match prob {
                        Random(p) => Pred::Prob(p),
                        Cycling(0) => Pred::Never,
                        Cycling(1) => Pred::Always,
                        Cycling(n) => Pred::Func(
                            Box::new(move |k| k % n == (d / m) % n)),
                        CyclingInv(0) => Pred::Always,
                        CyclingInv(1) => Pred::Never,
                        CyclingInv(n) => Pred::Func(
                            Box::new(move |k| k % n != (d / m) % n)),
                        Block(0) => Pred::Never,
                        Block(b) => Pred::Func(
                            Box::new(move |k| k / b == (d / m) % b)),
                        Window(0) => Pred::Never,
                        Window(w) => {
                            let d = d as isize;
                            let w = w as isize;
                            let n = self.n as isize;
                            let m = m as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d / m).rem_euclid(n) / w == 0
                                })
                            )
                        },
                    };
                do_measure(self, pred, buf);
                true
            },
            _ => {
                buf.iter_mut().for_each(|outk| { *outk = None; });
                false
            },
        }
    }

    fn entropy(&self, config: &EntropyConfig) -> f64 {
        match config {
            EntropyConfig::VonNeumann(part) => {
                let Range { start, end } = part;
                let start = *start;
                let end = *end;
                if start == 0 {
                    self.state.entropy_vn(end).unwrap_or(0.0)
                } else {
                    (
                        self.state.entropy_vn(start).unwrap_or(0.0)
                        - self.state.entropy_vn(end).unwrap_or(0.0)
                    ).abs()
                }
            },
            EntropyConfig::RenyiSchmidt(part, a) => {
                let Range { start, end } = part;
                let start = *start;
                let end = *end;
                if start == 0 {
                    self.state.entropy_ry_schmidt(*a, end).unwrap_or(0.0)
                } else {
                    (
                        self.state.entropy_ry_schmidt(*a, start).unwrap_or(0.0)
                        - self.state.entropy_ry_schmidt(*a, end).unwrap_or(0.0)
                    ).abs()
                }
            },
        }
    }

    fn do_run(
        &mut self,
        config: &mut CircuitConfig,
        mut meas: Option<&mut MeasRecord>,
        mut entropy: Option<&mut Vec<f64>>,
        // mut mutinf: Option<(&mut Vec<f64>, Option<usize>)>,
    ) {
        let CircuitConfig {
            depth: depth_conf,
            gates: ref mut gate_conf,
            measurement: meas_conf,
            entropy: entropy_conf,
        } = config;
        let depth_conf = *depth_conf;
        let meas_conf = *meas_conf;

        let mut outcomes: MeasLayer = vec![None; self.n];
        let mut s: f64 = self.entropy(entropy_conf);
        if let Some(rcd) = entropy.as_mut() { rcd.push(s); }
        let mut sbar: f64 = 0.0;
        let mut check: f64;

        let mut gates: Vec<Gate> = Vec::new();
        let mut d: usize = 1;
        loop {
            gates.clear();
            match gate_conf {
                GateConfig::Simple => {
                    self.sample_simple(d % 2 == 1, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Haar2 => {
                    self.apply_haars(d % 2 == 1);
                },
                GateConfig::GateSet(ref g1, ref g2) => {
                    self.sample_gateset(g1, g2, d % 2 == 1, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Circuit(ref circ) => {
                    self.state.apply_circuit(circ);
                },
                GateConfig::Feedback(ref mut f) => {
                    match f(d, s, &outcomes) {
                        Feedback::Halt => { break; },
                        Feedback::Simple => {
                            self.sample_simple(d % 2 == 1, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::GateSet(g1, g2) => {
                            self.sample_gateset(
                                &g1, &g2, d % 2 == 1, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::Circuit(circ) => {
                            self.state.apply_circuit(&circ);
                        },
                    }
                },
            }

            self.measure(d, meas_conf, &mut outcomes);
            if let Some(rcd) = meas.as_mut() { rcd.push(outcomes.clone()); }

            s = self.entropy(entropy_conf);
            if let Some(rcd) = entropy.as_mut() { rcd.push(s); }

            match depth_conf {
                DepthConfig::Unlimited => { },
                DepthConfig::Converge(tol) => {
                    check = (
                        2.0 * (sbar - s) / ((2 * d + 3) as f64 * sbar + s)
                    ).abs();
                    if check < tol.unwrap_or(1e-6) { break; }
                    sbar = (sbar + (d + 1) as f64 + s) / (d + 2) as f64;
                },
                DepthConfig::Const(d0) => { if d >= d0 { break; } },
            }
            d += 1;
        }
    }

    /// Run the MIPT procedure for a general config.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_entropy(
        &mut self,
        mut config: CircuitConfig,
        meas: Option<&mut MeasRecord>,
    ) -> Vec<f64> {
        let mut entropy: Vec<f64> = Vec::new();
        self.do_run(&mut config, meas, Some(&mut entropy), /* None */);
        entropy
    }

    // /// Run the MIPT procedure for a general config.
    // ///
    // /// Returns the mutual information measured once before the first layer, and
    // /// then after each round of measurements.
    // pub fn run_mutinf(
    //     &mut self,
    //     mut config: CircuitConfig,
    //     part_size: Option<usize>,
    //     meas: Option<&mut MeasRecord>,
    // ) -> Vec<f64> {
    //     let mut mutinf: Vec<f64> = Vec::new();
    //     self.do_run(&mut config, meas, None, Some((&mut mutinf, part_size)));
    //     mutinf
    // }

    /// Run the MIPT procedure for a general config without recording any
    /// time-evolution data.
    pub fn run(&mut self, mut config: CircuitConfig) {
        self.do_run(&mut config, None, None, /* None */)
    }
}

/// Iterator type for qubit indices under a two-qubit gate tiling.
struct TileQ2(std::iter::StepBy<std::ops::Range<usize>>);

impl TileQ2 {
    fn new(offs: bool, stop: usize) -> Self {
        Self((if offs { 1 } else { 0 } .. stop - 1).step_by(2))
    }
}

impl Iterator for TileQ2 {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
}

/// Post-measurement determination of a mid-circuit action.
#[derive(Clone, Debug)]
pub enum Feedback {
    /// Immediately halt the circuit.
    Halt,
    /// Draw the next layer of gates from the "simple" set (all single-qubit
    /// gates and tiling CXs).
    Simple,
    /// Draw the next layer of gates uniformly from gate sets (two-qubit gates
    /// will still alternately tile).
    GateSet(G1Set, G2Set),
    /// Apply a specific sequence of gates.
    Circuit(Vec<Gate>),
}

/// A feedback function.
type FeedbackFn<'a> = Box<dyn FnMut(usize, f64, &[Option<Outcome>]) -> Feedback + 'a>;

/// Set the termination condition for a circuit.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DepthConfig {
    /// Run indefinitely. Only makes sense if using
    /// [feedback][GateConfig::Feedback].
    Unlimited,
    /// Run until the entanglement entropy converges on a steady-state value to
    /// within some tolerance (defaults to 10<sup>-6</sup>).
    ///
    /// The criterion for convergence is
    ///
    /// 2|(*μ*<sub>*k*+1</sub> - *μ*<sub>*k*</sub>)
    /// / (*μ*<sub>*k*+1</sub> + *μ*<sub>*k*</sub>)| < *tol*
    ///
    /// where *μ*<sub>*k*</sub> is the running average of the first *k* entropy
    /// measurements, including the first before the circuit has begun; i.e. the
    /// entropy has reached a steady state when the absolute difference between
    /// consecutive values of the running average divided by their mean is less
    /// than *tol*.
    Converge(Option<f64>),
    /// Run for a constant depth.
    Const(usize),
}

/// One-qubit gate set to draw from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum G1Set {
    /// All available single-qubit gates (see below).
    All,
    /// Only Euler decompositions.
    U,
    /// Only Hadamards.
    H,
    /// Only X.
    X,
    /// Only Z.
    Z,
    /// Only S.
    S,
    /// Only S<sup>†</sup>.
    SInv,
    /// Only R<sub>X</sub>.
    XRot,
    /// Only R<sub>Z</sub>.
    ZRot,
    /// Only Hadamards and S.
    HS,
    /// A particular gate set.
    Set(Set<G1>),
}

impl G1Set {
    /// Sample a single gate.
    pub fn sample<R>(&self, k: usize, rng: &mut R) -> Gate
    where R: Rng + ?Sized
    {
        use std::f64::consts::TAU;
        match self {
            Self::All => match rng.gen_range(0..8_u8) {
                0 => {
                    let alpha: f64 = TAU * rng.gen::<f64>();
                    let beta: f64 = TAU * rng.gen::<f64>();
                    let gamma: f64 = TAU * rng.gen::<f64>();
                    Gate::U(k, alpha, beta, gamma)
                },
                1 => Gate::H(k),
                2 => Gate::X(k),
                3 => Gate::Z(k),
                4 => Gate::S(k),
                5 => Gate::SInv(k),
                6 => Gate::XRot(k, TAU * rng.gen::<f64>()),
                7 => Gate::ZRot(k, TAU * rng.gen::<f64>()),
                _ => unreachable!(),
            },
            Self::U => {
                let alpha: f64 = TAU * rng.gen::<f64>();
                let beta: f64 = TAU * rng.gen::<f64>();
                let gamma: f64 = TAU * rng.gen::<f64>();
                Gate::U(k, alpha, beta, gamma)
            },
            Self::H => Gate::H(k),
            Self::X => Gate::X(k),
            Self::Z => Gate::Z(k),
            Self::S => Gate::S(k),
            Self::SInv => Gate::SInv(k),
            Self::XRot => Gate::XRot(k, TAU * rng.gen::<f64>()),
            Self::ZRot => Gate::ZRot(k, TAU * rng.gen::<f64>()),
            Self::HS => if rng.gen::<bool>() {
                Gate::H(k)
            } else {
                Gate::S(k)
            },
            Self::Set(set) => Gate::sample_from(set, k, rng),
        }
    }
}

/// Two-qubit gate set to draw from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum G2Set {
    /// All available two-qubit gates (see below).
    All,
    /// Only CXs.
    CX,
    /// Only reverse CXs.
    CXRev,
    /// Only CZs.
    CZ,
    /// Only CXs and CZs.
    CXCZ,
    /// A particular gate set.
    Set(Set<G2>),
}

impl G2Set {
    /// Sample a single gate.
    pub fn sample<R>(&self, a: usize, rng: &mut R) -> Gate
    where R: Rng + ?Sized
    {
        match self {
            Self::All => match rng.gen_range(0..3_u8) {
                0 => Gate::CX(a),
                1 => Gate::CXRev(a),
                2 => Gate::CZ(a),
                _ => unreachable!(),
            },
            Self::CX => Gate::CX(a),
            Self::CXRev => Gate::CXRev(a),
            Self::CZ => Gate::CZ(a),
            Self::CXCZ => if rng.gen::<bool>() {
                Gate::CX(a)
            } else {
                Gate::CZ(a)
            },
            Self::Set(set) => Gate::sample_from(set, a, rng),
        }
    }
}

/// Define one- and two-qubit gate sets to draw from.
pub enum GateConfig<'a> {
    /// The "simple" set (all single-qubit gates and tiling CXs).
    Simple,
    /// Replace distinct, overlapped one- and two-qubit unitaries with tiled
    /// Haar-random two-qubit unitaries.
    Haar2,
    /// A particular gate set.
    GateSet(G1Set, G2Set),
    /// A particular sequence of gates.
    Circuit(Vec<Gate>),
    /// Gates based on a feedback function on measurement outcomes.
    Feedback(FeedbackFn<'a>),
}

/// Define the conditions for when measurements are applied.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MeasureConfig {
    /// Application of measurement layers.
    pub layer: MeasLayerConfig,
    /// Application of measurements within a single layer.
    pub prob: MeasProbConfig,
}

/// Define the conditions for when measurement layers are applied.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MeasLayerConfig {
    /// Perform measurements on every layer.
    Every,
    /// Perform measurements every `n` layers.
    ///
    /// `Period(1)` is equivalent to `Every`, and `Period(0)` applies no
    /// measurements.
    Period(usize),
}

/// Define the conditions for when measurements are applied within a single
/// layer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MeasProbConfig {
    /// Perform measurements randomly, as normal, with fixed probability.
    Random(f64),
    /// Perform a measurement on every `n`-th qubit, shifting by 1 on every
    /// measurement layer. `Cycling(0)` means to never measure any qubit and
    /// `Cycling(1)` means to always measure every qubit.
    Cycling(usize),
    /// Inverse of `Cycling`: Perform a measurement on every qubit *except*
    /// every `n`-th qubit, shifting by 1 on every measurement layer.
    /// `CyclingInv`(0)` means to always measure every qubit and `CyclingInv(1)`
    /// means to never measure any qubit.
    CyclingInv(usize),
    /// Perform measurements in blocks of `n` qubits that slide without overlap
    /// across the array.
    Block(usize),
    /// Perform measurements in sliding windows of `n` qubits.
    Window(usize),
}

impl MeasProbConfig {
    /// Convert a measurement probability to `Cycling(round(1 / p))` if `p <
    /// 0.5`, otherwise `CyclingInv(round(1 / (1 - p)))`.
    pub fn cycling_prob(p: f64) -> Self {
        if p.abs() < f64::EPSILON {
            Self::Cycling(0)
        } else if (1.0 - p).abs() < f64::EPSILON {
            Self::Cycling(1)
        } else if p.abs() < 0.5 {
            Self::Cycling(p.recip().round() as usize)
        } else {
            Self::CyclingInv((1.0 - p).recip().round() as usize)
        }
    }
}

/// Define the entropy to calculate and the subsystem on which to calculate it.
#[derive(Clone, Debug, PartialEq)]
pub enum EntropyConfig {
    /// The Von Neumann entropy.
    VonNeumann(Range<usize>),
    /// The Rényi entropy using a density matrix in the local Schmidt basis.
    RenyiSchmidt(Range<usize>, f64),
    // /// The Rényi entropy in the computational basis.
    // ///
    // /// **Warning**: this entropy is calculated by contracting the state into a
    // /// bona fide density matrix, which is *very* slow. Consider using either of
    // /// the other two entropies instead.
    // Renyi(Range<usize>, u32),
}

/// Top-level config for a circuit.
pub struct CircuitConfig<'a> {
    /// Set the depth of the circuit.
    pub depth: DepthConfig,
    /// Set available gates to draw from.
    pub gates: GateConfig<'a>,
    /// Set conditions for measurements.
    pub measurement: MeasureConfig,
    /// Set the entropy to calculate.
    pub entropy: EntropyConfig,
}

