//! Lazily evaluated version of [`MPS`] for quantum circuits with
//! nearest-neightbor couplings.

#![allow(unused_imports)]

use std::{ collections::VecDeque, thread };
use crossbeam::channel;
use ndarray as nd;
use num_complex::Complex64 as C64;
use thiserror::Error;
use crate::{
    mps::{ BondDim, MPS, MPSError },
    circuit::Q,
    gate::{ self, Gate },
};

#[derive(Debug, Error)]
pub enum LazyError {
    #[error("failed to enqueue eval work: dead thread")]
    DeadThread,

    #[error("failed to enqueue eval work: closed sender channel")]
    ClosedSenderChannel,

    #[error("failed to receive eval result: receiver error: {0}")]
    ClosedReceiverChannel(channel::RecvError),

    #[error("encountered receiver error from within a thread: receiver error: {0}")]
    WorkerReceiverError(channel::RecvError),

    #[error("MPS error: {0}")]
    MPSError(#[from] MPSError),
}
use LazyError::*;
pub type LazyResult<T> = Result<T, LazyError>;

// one-qubit gate
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum G1 {
    U(f64, f64, f64),
    H,
    X,
    Z,
    S,
    SInv,
    XRot(f64),
    ZRot(f64),
}

// two-qubit gate
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum G2 {
    CX,
    CXRev,
    CZ,
    Haar2,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Operation {
    G1(usize, G1),
    G2(usize, G2),
    M(usize),
}

impl From<Gate> for Operation {
    fn from(g: Gate) -> Self {
        match g {
            Gate::U(k, alpha, beta, gamma)
                => Self::G1(k, G1::U(alpha, beta, gamma)),
            Gate::H(k) => Self::G1(k, G1::H),
            Gate::X(k) => Self::G1(k, G1::X),
            Gate::Z(k) => Self::G1(k, G1::Z),
            Gate::S(k) => Self::G1(k, G1::S),
            Gate::SInv(k) => Self::G1(k, G1::SInv),
            Gate::XRot(k, ang) => Self::G1(k, G1::XRot(ang)),
            Gate::ZRot(k, ang) => Self::G1(k, G1::ZRot(ang)),
            Gate::CX(c) => Self::G2(c, G2::CX),
            Gate::CXRev(t) => Self::G2(t, G2::CXRev),
            Gate::CZ(c) => Self::G2(c, G2::CZ),
            Gate::Haar2(k) => Self::G2(k, G2::Haar2),
        }
    }
}

impl Operation {
    fn is_g1(&self) -> bool { matches!(self, Self::G1(..)) }

    fn is_g2(&self) -> bool { matches!(self, Self::G2(..)) }

    fn is_m(&self) -> bool { matches!(self, Self::M(..)) }

    fn operand(&self) -> usize {
        match self {
            Self::G1(k, _) => *k,
            Self::G2(k, _) => *k,
            Self::M(k) => *k,
        }
    }

    fn is_inverse(&self, other: &Self) -> bool {
        use std::f64::consts::TAU;
        match (self, other) {
            (Self::G1(a, gate_a), Self::G1(b, gate_b)) if a == b => {
                match (gate_a, gate_b) {
                    (G1::U(a_a, b_a, g_a), G1::U(a_b, b_b, g_b)) => {
                        (g_a + a_b).rem_euclid(TAU) < f64::EPSILON
                            && (b_a + b_b).rem_euclid(TAU) < f64::EPSILON
                            && (a_a + g_b).rem_euclid(TAU) < f64::EPSILON
                    },
                    (G1::H, G1::H) => true,
                    (G1::X, G1::X) => true,
                    (G1::Z, G1::Z) => true,
                    (G1::S, G1::SInv) => true,
                    (G1::SInv, G1::S) => true,
                    (G1::XRot(ang_a), G1::XRot(ang_b)) => {
                        (ang_a + ang_b).rem_euclid(TAU) < f64::EPSILON
                    },
                    (G1::ZRot(ang_a), G1::ZRot(ang_b)) => {
                        (ang_a + ang_b).rem_euclid(TAU) < f64::EPSILON
                    },
                    _ => false,
                }
            },
            (Self::G2(a, gate_a), Self::G2(b, gate_b)) if a == b => {
                matches!(
                    (gate_a, gate_b),
                    (G2::CX, G2::CX)
                    | (G2::CXRev, G2::CXRev)
                    | (G2::CZ, G2::CZ)
                )
            },
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum EvalState {
    Working,
    Idle(nd::Array3<C64>),
    G2L(nd::Array3<C64>),
    G2R(nd::Array3<C64>),
    Measure(nd::Array3<C64>),
}

/// Lazy gate applicator model for [`MPS<Q, C64>`].
#[derive(Clone, Debug, PartialEq)]
pub struct LazyMPS {
    pub(crate) n: usize, // number of qubits
    pub(crate) state: MPS<Q, C64>, // n qubits
    pub(crate) ops: VecDeque<Operation>,
    pub(crate) depth: usize,
    pub(crate) last: Vec<Option<(usize, Operation)>>, // length n
    pub(crate) eval: bool,
}

#[allow(unused_variables, unused_mut)]
impl LazyMPS {
    /// Initialize to a separable product state of `n` qubits with each particle
    /// in ∣0⟩.
    ///
    /// Optionally provide a global truncation method for discarding singular
    /// values. Defaults to a [`Cutoff`][BondDim::Cutoff] at machine epsilon.
    ///
    /// Fails if `n == 0`.
    pub fn new(n: usize, trunc: Option<BondDim<f64>>) -> LazyResult<Self> {
        let state = MPS::new((0..n).map(Q), trunc)?;
        let last: Vec<Option<(usize, Operation)>>
            = (0..n).map(|_| None).collect();
        Ok(Self { n, state, ops: VecDeque::new(), depth: 0, last, eval: false })
    }

    /// Create a new `LazyMPS` from an arbitrary [`MPS`].
    pub fn from_mps(mps: MPS<Q, C64>) -> Self {
        let n = mps.n();
        let last: Vec<Option<(usize, Operation)>>
            = (0..n).map(|_| None).collect();
        Self { n, state: mps, ops: VecDeque::new(), depth: 0, last, eval: false }
    }

    /// Unwrap `self` into a bare [`MPS<Q, C64>`], evaluating all accumulated
    /// operations if necessary.
    pub fn into_mps(mut self) -> MPS<Q, C64> {
        if self.eval { self.eval(); }
        self.state
    }

    pub fn apply_gate(&mut self, gate: Gate) -> &mut Self {
        let op: Operation = gate.into();
        let k = op.operand();
        if k >= self.n || (op.is_g2() && k >= self.n - 1) { return self; }
        if let Some((o_index, o)) = self.last[k].as_ref() {
            if op.is_inverse(o) {
                self.ops.remove(*o_index);
                let last: Option<(usize, Operation)>
                    = self.ops.iter().enumerate().rev()
                    .find(|(_, o)| o.operand() == k)
                    .map(|(j, o)| (j, *o));
                let j = *o_index; // drop the reference to self.last
                self.last[j] = last;
                self.depth -= 1;
            } else {
                self.ops.push_back(op);
                self.last[k] = Some((self.depth, op));
                if op.is_g2() { self.last[k + 1] = Some((self.depth, op)); }
                self.depth += 1;
            }
        } else {
            self.ops.push_back(op);
            self.last[k] = Some((self.depth, op));
            if op.is_g2() { self.last[k + 1] = Some((self.depth, op)); }
            self.depth += 1;
        }
        self.eval = self.depth > 0;
        self
    }

    fn eval(&mut self) { todo!() }
}

impl From<MPS<Q, C64>> for LazyMPS {
    fn from(mps: MPS<Q, C64>) -> Self { Self::from_mps(mps) }
}

impl From<LazyMPS> for MPS<Q, C64> {
    fn from(lazy: LazyMPS) -> Self { lazy.into_mps() }
}

#[derive(Clone, Debug)]
enum ToWorker {
    Stop,
    Gate1q(usize, nd::Array3<C64>, G1),
    Gate2q(usize, nd::Array3<C64>, G2),
}

#[derive(Clone, Debug)]
enum FromWorker {
    RecvError(channel::RecvError),
    Output1q(usize, nd::Array3<C64>),
    Output2q(usize, nd::Array3<C64>, nd::Array1<C64>, nd::Array3<C64>),
}

/// A simple thread pool to apply gates to qubit MPS tensors in parallel.
///
/// Workload between threads is automatically balanced at a coarse level by
/// means of a single-producer, multiple-consumer channel. Contracted pairs are
/// returned in the order in which the operations are completed, but are tagged
/// with qubit indices. The pool as a whole is meant to be reused between
/// batches of contractions, and is **not** thread-safe.
#[derive(Debug)]
pub struct GatePool {
    thread: Vec<thread::JoinHandle<()>>,
    workers_in: channel::Sender<ToWorker>,
    workers_out: channel::Receiver<FromWorker>,
}

#[allow(unused_variables, unused_mut)]
impl GatePool {
    /// Create a new thread pool of `nthreads` threads.
    pub fn new(nthreads: usize) -> Self {
        todo!()
    }

    /// Create a new thread pool with the number of threads equal to the number
    /// of logical CPU cores available in the current system.
    pub fn new_cpus() -> Self { Self::new(num_cpus::get()) }

    /// Create a new thread pool with the number of threads equal to the number
    /// of physical CPU cores available in the current system.
    pub fn new_physical() -> Self { Self::new(num_cpus::get_physical()) }


}

