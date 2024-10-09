//! Abstractions for driving randomized circuits on qubits and measureing
//! measurement-induced phase transitions (MIPTs).

use std::ops::Range;
use rand::{ rngs::StdRng, Rng, SeedableRng };
use rustc_hash::FxHashSet as HashSet;
use num_complex::Complex64 as C64;
use once_cell::sync::Lazy;
use thiserror::Error;
use crate::{
    gate::{ self, GateToken, Gate, G1, G2, ExactGate },
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
            reset: bool,
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
                            *outk = if reset {
                                circ.state.measure_reset(k, &mut circ.rng)
                            } else {
                                circ.state.measure(k, &mut circ.rng)
                            }.map(Outcome::from)
                        },
                        Pred::Prob(p) => {
                            *outk = if circ.rng.gen::<f64>() < *p {
                                if reset {
                                    circ.state.measure_reset(k, &mut circ.rng)
                                } else {
                                    circ.state.measure(k, &mut circ.rng)
                                }.map(Outcome::from)
                            } else {
                                None
                            };
                        },
                        Pred::Func(f) => {
                            *outk = if f(k) {
                                if reset {
                                    circ.state.measure_reset(k, &mut circ.rng)
                                } else {
                                    circ.state.measure(k, &mut circ.rng)
                                }.map(Outcome::from)
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

        let MeasureConfig { layer, prob, reset } = config;
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
                do_measure(self, pred, reset, buf);
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
                do_measure(self, pred, reset, buf);
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
                let end = end.saturating_sub(1);
                if start == 0 {
                    self.state.entropy_vn(end).unwrap_or(0.0)
                } else {
                    (
                        self.state.entropy_vn(start - 1).unwrap_or(0.0)
                        - self.state.entropy_vn(end).unwrap_or(0.0)
                    ).abs()
                }
            },
            EntropyConfig::RenyiSchmidt(part, a) => {
                let Range { start, end } = part;
                let start = *start;
                let end = end.saturating_sub(1);
                if start == 0 {
                    self.state.entropy_ry_schmidt(*a, end).unwrap_or(0.0)
                } else {
                    (
                        self.state.entropy_ry_schmidt(*a, start - 1).unwrap_or(0.0)
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
                    self.sample_simple(d % 2 == 0, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Haar2 => {
                    self.apply_haars(d % 2 == 0);
                },
                GateConfig::GateSet(ref g1, ref g2) => {
                    self.sample_gateset(g1, g2, d % 2 == 0, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Circuit(ref circ) => {
                    self.state.apply_circuit(circ);
                },
                GateConfig::Feedback(ref mut f) => {
                    match f(d, s, &outcomes) {
                        Feedback::Halt => { break; },
                        Feedback::Simple => {
                            self.sample_simple(d % 2 == 0, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::Haar2 => {
                            self.apply_haars(d % 2 == 0);
                        },
                        Feedback::GateSet(g1, g2) => {
                            self.sample_gateset(
                                &g1, &g2, d % 2 == 0, &mut gates);
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
    ) -> Vec<f64>
    {
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
    pub fn run(
        &mut self,
        mut config: CircuitConfig,
        meas: Option<&mut MeasRecord>,
    ) {
        self.do_run(&mut config, meas, None, /* None */);
    }

    fn do_run_fixed(
        &mut self,
        circ: &Circuit,
        mut meas: Option<&mut MeasRecord>,
        mut entropy: Option<&mut Vec<f64>>,
    ) {
        let Circuit { n: _, ops, reset, entropy: entropy_conf } = circ;
        let reset = *reset;
        let mut outcomes: MeasLayer = vec![None; self.n];
        let mut s: f64;
        if let Some(rcd) = entropy.as_mut() {
            s = self.entropy(entropy_conf);
            rcd.push(s);
        }
        for layer in ops.iter() {
            for uni in layer.unis.iter() {
                match uni {
                    Unitary::Gate(gate) => {
                        self.state.apply_gate(*gate);
                    },
                    Unitary::ExactGate(ExactGate::Q1(k, gate)) => {
                        self.state.apply_unitary1(*k, gate)
                            .expect("invalid q1 unitary application");
                    },
                    Unitary::ExactGate(ExactGate::Q2(k, gate)) => {
                        self.state.apply_unitary2(*k, gate)
                            .expect("invalid q2 unitary application");
                    },
                }
            }

            for &m in layer.meas.iter() {
                match m {
                    Measurement::Rand(k) => {
                        if let Some(outk) = outcomes.get_mut(k) {
                            if reset {
                                *outk
                                    = self.state.measure_reset(k, &mut self.rng)
                                    .map(Outcome::from);
                            } else {
                                *outk
                                    = self.state.measure(k, &mut self.rng)
                                    .map(Outcome::from);
                            }
                        }
                    },
                    Measurement::Proj(k, out) => {
                        if let Some(outk) = outcomes.get_mut(k) {
                            let p = out as usize;
                            if reset {
                                self.state.measure_postsel_reset(k, p);
                            } else {
                                self.state.measure_postsel(k, p);
                            }
                            *outk = Some(out);
                        }
                    }
                }
            }

            if let Some(rcd) = meas.as_mut() {
                let mut tmp: MeasLayer = vec![None; self.n];
                std::mem::swap(&mut tmp, &mut outcomes);
                rcd.push(tmp);
            }

            if let Some(rcd) = entropy.as_mut() {
                s = self.entropy(entropy_conf);
                rcd.push(s);
            }
        }
    }

    /// Run the MIPT procedure for a completely fixed circuit.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_entropy_fixed(
        &mut self,
        circ: &Circuit,
        meas: Option<&mut MeasRecord>,
    ) -> Vec<f64>
    {
        let mut entropy: Vec<f64> = Vec::new();
        self.do_run_fixed(circ, meas, Some(&mut entropy));
        entropy
    }

    /// Run the MIPT procedure for a completely fixed circuit without recording
    /// any time-evolution data.
    pub fn run_fixed(
        &mut self,
        circ: &Circuit,
        meas: Option<&mut MeasRecord>,
    ) {
        self.do_run_fixed(circ, meas, None);
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
    /// Draw the next layer of gates uniformly from the two-qubit Haar measure.
    Haar2,
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
            Self::All => G1::sample_random(k, rng),
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

    /// Sample an exact gate.
    pub fn sample_exact<R>(&self, k: usize, rng: &mut R) -> ExactGate
    where R: Rng + ?Sized
    {
        use std::f64::consts::TAU;
        match self {
            Self::All => G1::sample_random(k, rng),
            Self::U => {
                let alpha: f64 = TAU * rng.gen::<f64>();
                let beta: f64 = TAU * rng.gen::<f64>();
                let gamma: f64 = TAU * rng.gen::<f64>();
                ExactGate::Q1(k, gate::make_u(alpha, beta, gamma))
            },
            Self::H => ExactGate::Q1(k, Lazy::force(&gate::HMAT).clone()),
            Self::X => ExactGate::Q1(k, Lazy::force(&gate::XMAT).clone()),
            Self::Z => ExactGate::Q1(k, Lazy::force(&gate::ZMAT).clone()),
            Self::S => ExactGate::Q1(k, Lazy::force(&gate::SMAT).clone()),
            Self::SInv => ExactGate::Q1(k, Lazy::force(&gate::SINVMAT).clone()),
            Self::XRot => ExactGate::Q1(k, gate::make_xrot(TAU * rng.gen::<f64>())),
            Self::ZRot => ExactGate::Q1(k, gate::make_zrot(TAU * rng.gen::<f64>())),
            Self::HS => if rng.gen::<bool>() {
                ExactGate::Q1(k, Lazy::force(&gate::HMAT).clone())
            } else {
                ExactGate::Q1(k, Lazy::force(&gate::SMAT).clone())
            },
            Self::Set(set) => ExactGate::sample_from(set, k, rng),
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
            Self::All => G2::sample_random(a, rng),
            // Self::All => match rng.gen_range(0..3_u8) {
            //     0 => Gate::CX(a),
            //     1 => Gate::CXRev(a),
            //     2 => Gate::CZ(a),
            //     _ => unreachable!(),
            // },
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

    /// Sample an exact gate.
    pub fn sample_exact<R>(&self, a: usize, rng: &mut R) -> ExactGate
    where R: Rng + ?Sized
    {
        match self {
            Self::All => G2::sample_random(a, rng),
            Self::CX => ExactGate::Q2(a, Lazy::force(&gate::CXMAT).clone()),
            Self::CXRev => ExactGate::Q2(a, Lazy::force(&gate::CXREVMAT).clone()),
            Self::CZ => ExactGate::Q2(a, Lazy::force(&gate::CZMAT).clone()),
            Self::CXCZ => if rng.gen::<bool>() {
                ExactGate::Q2(a, Lazy::force(&gate::CXMAT).clone())
            } else {
                ExactGate::Q2(a, Lazy::force(&gate::CZMAT).clone())
            },
            Self::Set(set) => ExactGate::sample_from(set, a, rng),
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
    /// Apply a deterministic reset back to ∣0⟩ after each measurement.
    pub reset: bool,
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

/// A single unitary operation in a [`Circuit`].
///
/// Can be either a general gate description to leave two-qubit Haar gates
/// random, or a literal unitary matrix to fix the action of the gate exactly.
#[derive(Clone, Debug, PartialEq)]
pub enum Unitary {
    /// A general gate description.
    Gate(Gate),
    /// An exact one- or two-qubit unitary.
    ///
    /// **Note**: This variant stores the exact 2x2 or 4x4 matrix representation
    /// of the unitary; if your circuit only contains non-Haar-random gates
    /// (e.g. `H`, `XRot`, `S`), then consider using `Gate` instead.
    ExactGate(ExactGate),
}

impl From<Gate> for Unitary {
    fn from(gate: Gate) -> Self { Self::Gate(gate) }
}

impl From<ExactGate> for Unitary {
    fn from(gate: ExactGate) -> Self { Self::ExactGate(gate) }
}

/// A measurement operation in a [`Circuit`].
///
/// Measurements are either naive (fully randomized) or projectors with
/// pre-determined outcomes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Measurement {
    /// A fully randomized measurement on a particular qubit.
    Rand(usize),
    /// A measurement that is post-selected to a particular outcome.
    Proj(usize, Outcome),
}

impl Measurement {
    /// Return the qubit index of the measurement.
    pub fn idx(&self) -> usize {
        match self {
            Self::Rand(k) => *k,
            Self::Proj(k, _) => *k,
        }
    }
}

/// The operations in a single layer of a [`Circuit`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct OpLayer {
    /// All unitary operations.
    pub unis: Vec<Unitary>,
    /// The indices of the qubits to be measured.
    // pub meas: Vec<usize>,
    pub meas: Vec<Measurement>,
}

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("non-constant depths are not supported by Circuit")]
    NoDynDepth,

    #[error("active-feedback circuits are not supported by Circuit")]
    NoFeedback,
}
use CircuitError::*;
pub type CircuitResult<T> = Result<T, CircuitError>;

/// A fixed set of circuit operations.
#[derive(Clone, Debug, PartialEq)]
pub struct Circuit {
    n: usize,
    ops: Vec<OpLayer>,
    reset: bool,
    entropy: EntropyConfig,
}

impl Circuit {
    /// Return the number of qubits in the circuit.
    pub fn nqubits(&self) -> usize { self.n }

    /// Return the depth of the circuit.
    pub fn depth(&self) -> usize { self.ops.len() }

    /// Return a reference to a particular layer in the circuit.
    pub fn get_layer(&self, k: usize) -> Option<&OpLayer> {
        self.ops.get(k)
    }

    /// Return a mutable reference to a particular layer in the circuit.
    pub fn get_layer_mut(&mut self, k: usize) -> Option<&mut OpLayer> {
        self.ops.get_mut(k)
    }

    fn sample_simple<R>(
        n: usize,
        offs: bool,
        exact: bool,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        if exact {
            (0..n).for_each(|k| {
                let uni = Unitary::ExactGate(G1::sample_random(k, rng));
                ops.unis.push(uni);
            });
            TileQ2::new(offs, n).for_each(|a| {
                let uni = ExactGate::Q2(a, Lazy::force(&gate::CXMAT).clone());
                ops.unis.push(uni.into());
            });
        } else {
            (0..n).for_each(|k| {
                let uni = Unitary::Gate(G1::sample_random(k, rng));
                ops.unis.push(uni);
            });
            TileQ2::new(offs, n).for_each(|a| {
                let uni = Unitary::Gate(Gate::CX(a));
                ops.unis.push(uni);
            });
        }
    }

    fn sample_haars<R>(
        n: usize,
        offs: bool,
        exact: bool,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        if exact {
            TileQ2::new(offs, n).for_each(|a| {
                let gate = ExactGate::Q2(a, gate::haar(2, rng));
                ops.unis.push(Unitary::ExactGate(gate))
            });
        } else {
            TileQ2::new(offs, n).for_each(|a| {
                ops.unis.push(Unitary::Gate(Gate::Haar2(a)));
            });
        }
    }

    fn sample_gateset<R>(
        n: usize,
        g1: &G1Set,
        g2: &G2Set,
        offs: bool,
        exact: bool,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        if exact {
            (0..n).for_each(|k| {
                ops.unis.push(Unitary::ExactGate(g1.sample_exact(k, rng)));
            });
            TileQ2::new(offs, n).for_each(|a| {
                ops.unis.push(Unitary::ExactGate(g2.sample_exact(a, rng)));
            });
        } else {
            (0..n).for_each(|k| {
                ops.unis.push(Unitary::Gate(g1.sample(k, rng)));
            });
            TileQ2::new(offs, n).for_each(|a| {
                ops.unis.push(Unitary::Gate(g2.sample(a, rng)));
            });
        }
    }

    fn sample_measurements<R>(
        n: usize,
        d: usize,
        config: MeasureConfig,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        use MeasLayerConfig::*;
        use MeasProbConfig::*;

        enum Pred<'a> {
            Never,
            Always,
            Prob(f64),
            Func(Box<dyn Fn(usize) -> bool + 'a>),
        }

        fn do_measure<R>(
            n: usize,
            pred: Pred,
            ops: &mut OpLayer,
            rng: &mut R
        )
        where R: Rng + ?Sized
        {
            (0..n)
                .filter(|k| {
                    match &pred {
                        Pred::Never => false,
                        Pred::Always => true,
                        Pred::Prob(p) => rng.gen::<f64>() < *p,
                        Pred::Func(f) => f(*k),
                    }
                })
                .for_each(|k| { ops.meas.push(Measurement::Rand(k)); });
        }

        let MeasureConfig { layer, prob, reset: _ } = config;
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
                            let n = n as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d).rem_euclid(n) / w == 0
                                })
                            )
                        },
                    };
                do_measure(n, pred, ops, rng);
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
                            let n = n as isize;
                            let m = m as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d / m).rem_euclid(n) / w == 0
                                })
                            )
                        },
                    };
                do_measure(n, pred, ops, rng);
            },
            _ => { }
        }
    }

    /// Generate a fixed circuit description for `n` qubits from a
    /// [`CircuitConfig`].
    ///
    /// If `exact == true`, then all gates are stored
    /// [exactly][Unitary::ExactGate`]s.
    pub fn gen<R>(n: usize, config: CircuitConfig, exact: bool, rng: &mut R)
        -> CircuitResult<Self>
    where R: Rng + ?Sized
    {
        let CircuitConfig {
            depth: depth_conf,
            gates: gate_conf,
            measurement: meas_conf,
            entropy: entropy_conf,
        } = config;
        let DepthConfig::Const(depth) = depth_conf
            else { return Err(NoDynDepth); };
        if let GateConfig::Feedback(..) = &gate_conf { return Err(NoFeedback); }

        let mut ops: Vec<OpLayer> = Vec::new();
        let mut layer = OpLayer::default();
        for d in 0..depth {
            match &gate_conf {
                GateConfig::Simple => {
                    Self::sample_simple(n, d % 2 == 1, exact, &mut layer, rng);
                },
                GateConfig::Haar2 => {
                    Self::sample_haars(n, d % 2 == 1, exact, &mut layer, rng);
                },
                GateConfig::GateSet(ref g1, ref g2) => {
                    Self::sample_gateset(
                        n, g1, g2, d % 2 == 1, exact, &mut layer, rng);
                },
                GateConfig::Circuit(circ) => {
                    circ.iter().copied()
                        .for_each(|gate| { layer.unis.push(gate.into()); });
                },
                GateConfig::Feedback(..) => unreachable!(),
            }
            Self::sample_measurements(n, d, meas_conf, &mut layer, rng);
            ops.push(std::mem::take(&mut layer));
        }
        Ok(Self { n, ops, reset: meas_conf.reset, entropy: entropy_conf })
    }

    /// Increase the effective measurement probability to a new value
    /// "conservatively" by sampling new ([randomized][Measurement::Rand])
    /// measurement locations uniformly, *only* from positions that do not
    /// already have a measurement.
    ///
    /// This sorts the measurement indices in each layer and does not respect
    /// any non-probabilistic measurement patterns. Does nothing if `p_new` is
    /// less than the current effective value.
    pub fn upsample_measurements<R>(&mut self, p_new: f64, rng: &mut R)
    where R: Rng + ?Sized
    {
        let n: f64 = self.n as f64;
        let p0: f64
            = self.ops.iter()
            .enumerate()
            .map(|(d, layer)| (d as f64, layer.meas.len() as f64))
            .fold(0.0, |acc, (d, m)| (acc * d * n + m) / ((d + 1.0) * n));
        if p_new <= p0 { return; }
        let p_sample = (p_new - p0) / (1.0 - p0);
        let mut new_pos: Vec<Measurement> = Vec::with_capacity(self.n);
        for OpLayer { unis: _, meas } in self.ops.iter_mut() {
            if meas.is_empty() {
                (0..self.n).filter(|_| rng.gen::<f64>() < p_sample)
                    .for_each(|k| { meas.push(Measurement::Rand(k)); });
            } else {
                meas.sort_by_key(Measurement::idx);
                let mut k0: usize = 0;
                meas.iter().copied()
                    .flat_map(|m| {
                        let k = m.idx();
                        let start = k0;
                        k0 = k + 1;
                        start..k
                    })
                    .filter(|_| rng.gen::<f64>() < p_sample)
                    .for_each(|k| { new_pos.push(Measurement::Rand(k)); });
                (k0..self.n)
                    .filter(|_| rng.gen::<f64>() < p_sample)
                    .for_each(|k| { new_pos.push(Measurement::Rand(k)); });
                meas.append(&mut new_pos);
                meas.sort_by_key(Measurement::idx);
            }
        }
    }
}

