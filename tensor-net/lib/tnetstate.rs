//! *N*-qubit register states in a tensor network-based density matrix
//! representation.
//!
//! - defined the [`Q`] index type
//! - defined gates

use std::{ cmp::Ordering, rc::Rc };
use ndarray as nd;
use ndarray_linalg::{ Eigh, UPLO };
use num_complex::Complex32 as C32;
use crate::{
    network::Network,
    pool::ContractorPool,
    tensor::{ Idx, Tensor },
    gate::tnet::{ self as gate, Gate },
};

/// An [index][Idx] representing a qubit degree of freedom.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Q(
    /// Qubit index.
    pub usize,
    /// Bra- (`true`) or ket- (`false`) side of a density matrix.
    pub bool,
    // Index to keep track of gate input/output indices in a state before
    // contractions are performed. Upon evaluation, this field is reset to zero.
    pub(crate) usize,
);

impl PartialOrd for Q {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Q {
    fn cmp(&self, other: &Self) -> Ordering {
        self.2.cmp(&other.2)
            .then(self.1.cmp(&other.1))
            .then(self.0.cmp(&other.0))
    }
}

impl From<(usize, bool)> for Q {
    fn from(x: (usize, bool)) -> Self { Self(x.0, x.1, 0) }
}

impl From<Q> for (usize, bool) {
    fn from(q: Q) -> Self { (q.0, q.1) }
}

impl Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String {
        format!("q:{}{}", self.0, if self.1 { "'" } else { "" })
    }
}

impl Q {
    fn reset_depth(&mut self) { self.2 = 0; }
}

/// A node in a qubit tensor network.
pub type QTensor = Tensor<Q, C32>;

/// A tensor network comprising [`QTensor`]s.
pub type QNetwork = Network<Q, C32>;

/// A [`ContractorPool`] for [`QTensor`]s.
pub type QPool = ContractorPool<Q, C32>;

/// A general *N*-qbit state represented as a tensor network.
#[derive(Clone, Debug)]
pub struct TNet {
    pub(crate) n: usize,
    pub(crate) net: QNetwork,
    pub(crate) qdepth: Vec<usize>, // length == n
}

impl TNet {
    /// Create a new `n`-qubit state initialized to ∣0...0⟩.
    pub fn new(n: usize) -> Self {
        let mut net = QNetwork::new();
        let indices: Vec<Q>
            = (0..n).map(|q| Q(q, false, 0))
            .chain((0..n).map(|q| Q(q, true, 0)))
            .collect();
        let mut shape = vec![2_usize; 2 * n];
        let mut arr: nd::ArrayD<C32> = nd::ArrayD::zeros(shape.as_slice());
        shape.fill(0);
        arr[shape.as_slice()] = C32::from(1.0);
        let init = unsafe { QTensor::from_array_unchecked(indices, arr) };
        net.push(init).unwrap();
        let qdepth: Vec<usize> = vec![0; n];
        Self { n, net, qdepth }
    }

    /// Apply a gate.
    ///
    /// This method does not compute the action of the gate directly, it only
    /// inserts the appropriate tensor into the internal tensor network. To
    /// actually evaluate the output state, call [`Self::eval`] or apply a
    /// measurement. If any qubit indices in the gate are invalid, no gate is
    /// added.
    pub fn apply_gate(&mut self, gate: &Gate) -> &mut Self {
        match *gate {
            Gate::U(q, alpha, beta, gamma) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_u_tens(q, false, d, alpha, beta, gamma);
                let g_bra = gate::make_u_tens(q, true,  d, alpha, beta, gamma);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::H(q) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_h_tens(q, false, d);
                let g_bra = gate::make_h_tens(q, true,  d);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::X(q) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_x_tens(q, false, d);
                let g_bra = gate::make_x_tens(q, true,  d);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::Y(q) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_y_tens(q, false, d);
                let g_bra = gate::make_y_tens(q, true,  d);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::Z(q) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_z_tens(q, false, d);
                let g_bra = gate::make_z_tens(q, true,  d);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::S(q) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_s_tens(q, false, d);
                let g_bra = gate::make_s_tens(q, true,  d);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::SInv(q) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_sinv_tens(q, false, d);
                let g_bra = gate::make_sinv_tens(q, true,  d);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::XRot(q, ang) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_xrot_tens(q, false, d, ang);
                let g_bra = gate::make_xrot_tens(q, true,  d, ang);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::YRot(q, ang) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_yrot_tens(q, false, d, ang);
                let g_bra = gate::make_yrot_tens(q, true,  d, ang);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },  
            Gate::ZRot(q, ang) if q < self.n => {
                let d = self.qdepth[q];
                let g_ket = gate::make_zrot_tens(q, false, d, ang);
                let g_bra = gate::make_zrot_tens(q, true,  d, ang);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[q] += 1;
            },
            Gate::CX(c, t) if c < self.n && t < self.n => {
                let dc = self.qdepth[c];
                let dt = self.qdepth[t];
                let g_ket = gate::make_cx_tens(c, t, false, dc, dt);
                let g_bra = gate::make_cx_tens(c, t, true,  dc, dt);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[c] += 1;
                self.qdepth[t] += 1;
            },
            Gate::CY(c, t) if c < self.n && t < self.n => {
                let dc = self.qdepth[c];
                let dt = self.qdepth[t];
                let g_ket = gate::make_cy_tens(c, t, false, dc, dt);
                let g_bra = gate::make_cy_tens(c, t, true,  dc, dt);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[c] += 1;
                self.qdepth[t] += 1;
            },
            Gate::CZ(c, t) if c < self.n && t < self.n => {
                let dc = self.qdepth[c];
                let dt = self.qdepth[t];
                let g_ket = gate::make_cz_tens(c, t, false, dc, dt);
                let g_bra = gate::make_cz_tens(c, t, true,  dc, dt);
                self.net.push(g_ket).unwrap();
                self.net.push(g_bra).unwrap();
                self.qdepth[c] += 1;
                self.qdepth[t] += 1;
            },
            _ => { },
        }
        self
    }

    /// Apply a series of gates.
    ///
    /// This method does not compute the action of the gate directly, it only
    /// inserts the appropriate tensors into the internal tensor network. To
    /// actually evaluate the output state, call [`Self::eval`] or apply a
    /// measurement. If any qubit indices in the gate are invalid, no gate is
    /// added.
    pub fn apply_circuit<'a, I>(&mut self, gates: I) -> &mut Self
    where I: IntoIterator<Item = &'a Gate>
    {
        gates.into_iter().for_each(|g| { self.apply_gate(g); });
        self
    }

    fn reset_depth_idx(&mut self) {
        unsafe { self.net.update_indices(|idx| { idx.reset_depth(); }); }
        self.qdepth.iter_mut().for_each(|d| { *d = 0; });
    }

    /// Contract the network to evalute the output state.
    pub fn eval(&mut self, pool: &QPool) -> &mut Self {
        self.net.contract_network_par(pool).unwrap();
        self.reset_depth_idx();
        self
    }

    /// Apply a projective measurement to the `k`-th qubit.
    ///
    /// If `k` is out of bounds, do nothing.
    pub fn measure(&mut self, k: usize, pool: &QPool) {
        if k >= self.n { return; }
        self.eval(pool);
        let mut rho = self.net.contract_remove_par(pool).unwrap();
        if rho.rank() != 2 * self.n { panic!("extra dimensions in measure"); }
        rho.sort_indices();
        let rho_0 = rho.map(|idx, a| {
            if idx[k] == 0 && idx[self.n + k] == 0 { *a } else { 0.0.into() }
        });
        rho.map_inplace(|idx, a| {
            if idx[k] == 1 && idx[self.n + k] == 1 { *a } else { 0.0.into() }
        });
        self.net.push(rho + rho_0).unwrap();
    }

    /// Convert to a full density matrix, with row and column indices collapsed
    /// into the standard `n`-qubit computational basis.
    pub fn into_matrix(self, pool: &QPool) -> nd::Array2<C32> {
        let mut rho = self.net.contract_par(pool).unwrap();
        if rho.rank() != 2 * self.n {
            panic!("invalid dimensions in into_matrix ({})", rho.rank());
        }
        unsafe { rho.indices_mut().for_each(|idx| { idx.reset_depth(); }); }
        rho.sort_indices();
        rho.indices().for_each(|idx| { println!("{:?}", idx); });
        let n = self.n;
        rho.into_flat().1
            .into_shape((1 << n, 1 << n)).unwrap()
            .into_dimensionality::<nd::Ix2>().unwrap()
    }

    /// Like [`Self::into_matrix`], but with all qubits outside of `part` traced
    /// out.
    ///
    /// `part` defaults to the leftmost `floor(n / 2)` qubits.
    pub fn into_matrix_part(mut self, part: Option<Partition>, pool: &QPool)
        -> nd::Array2<C32>
    {
        const ONE: C32 = C32 { re: 1.0, im: 0.0 };
        const ZERO: C32 = C32 { re: 0.0, im: 0.0 };
        let mut tracer: QTensor;
        let mut d: usize;
        let part = part.unwrap_or(Partition::Left(self.n / 2 - 1));
        for k in 0..self.n {
            if part.contains(k) { continue; }
            d = self.qdepth[k];
            tracer = unsafe {
                QTensor::from_array_unchecked(
                    [Q(k, true, d), Q(k, false, d)],
                    nd::array![ [ONE,  ZERO], [ZERO, ONE ] ],
                )
            };
            self.net.push(tracer).unwrap();
        }
        self.n = part.size(self.n);
        self.into_matrix(pool)
    }
}

// compute the matrix (natural) logarithm via eigendecomp
// panics of the matrix is not Hermitian
pub(crate) fn mat_ln(x: &nd::Array2<C32>) -> nd::Array2<C32> {
    let (e, v) = x.eigh(UPLO::Upper).expect("mat_ln: non-Hermitian matrix");
    let u = v.t().mapv(|a| a.conj());
    let log_e = nd::Array2::from_diag(&e.mapv(|ek| C32::from(ek.ln())));
    v.dot(&log_e).dot(&u)
}

// compute an integer power of a matrix
// panics if the matrix is not square
// expecting most cases to use n < 5
pub(crate) fn mat_pow(x: &nd::Array2<C32>, n: u32) -> nd::Array2<C32> {
    if !x.is_square() { panic!("mat_pow: non-square matrix"); }
    match n {
        0 => nd::Array2::eye(x.shape()[0]),
        1 => x.clone(),
        m if m < 4 => {
            let mut acc = x.clone();
            for _ in 1..m { acc = acc.dot(x); }
            acc
        },
        m if m < 12 => {
            // exponent on the largest power of 2 that's ≤ m
            let k: u32 = (1..).find(|p| 1 << (p + 1) > m).unwrap();
            let r: u32 = m - (1 << k); // remainder
            let mut acc = x.clone();
            for _ in 0..k { acc = acc.dot(&acc); }
            for _ in 0..r { acc = acc.dot(x); }
            acc
        },
        m => {
            // exponent on the largest power of 2 that's ≤ m
            let k1: u32 = (1..).find(|p| 1 << (p + 1) > m).unwrap();
            let r1: u32 = m - (1 << k1); // remainder
            // expoment on the largest power of 2 that's ≤ r1
            let k2: u32 = (1..).find(|p| 1 << (p + 1) > r1).unwrap();
            let r: u32 = r1 - (1 << k2); // remainder
            let mut acc1 = x.clone();
            for _ in 0..k2 { acc1 = acc1.dot(&acc1); } // k2 ≤ k1
            let acc2 = acc1.clone(); // save the intermediate result
            for _ in 0..k1 - k2 { acc1 = acc1.dot(&acc1); }
            acc1 = acc1.dot(&acc2);
            for _ in 0..r { acc1 = acc1.dot(x); }
            acc1
        }
    }
}

/// Compute the Von Neumann entropy of a density matrix.
///
/// The input must be a valid density matrix, but can be pure or mixed.
pub fn entropy_vn(rho: &nd::Array2<C32>) -> f32 {
    rho.dot(&mat_ln(rho))
        .diag().iter()
        .map(|r| r.re)
        .sum::<f32>() * (-1.0)
}

/// Compute the *n*-th Rényi entropy of a density matrix.
///
/// Returns the [Von Neumann entropy][entropy_vn] if `n == 1`. The input must be
/// a valid density matrix, but can be pure or mixed.
pub fn entropy_renyi(rho: &nd::Array2<C32>, n: u32) -> f32 {
    if n == 1 {
        entropy_vn(rho)
    } else {
        let pref = (1.0 - n as f32).recip();
        mat_pow(rho, n)
            .diag().iter()
            .map(|r| r.re)
            .sum::<f32>()
            .ln() * pref
    }
}

/// Describes a subset of a [`TNet`] (assumed as a linear chain with periodic
/// boundary conditions) for which to calculate the entanglement entropy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Partition {
    /// The leftmost `n` qubits.
    Left(usize),
    /// The rightmost `n` qubits.
    Right(usize),
    /// A contiguous range of qubit indices, beginning at the left argument and
    /// ending (non-inclusively) at the right.
    Range(usize, usize),
    /// The union of two partitions.
    Union(Rc<Partition>, Rc<Partition>),
}

impl Partition {
    /// Return the union of two partitions.
    pub fn union(self, other: Self) -> Self {
        Self::Union(Rc::new(self), Rc::new(other))
    }

    /// Return the intersection of two partitions.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Left(r1), Self::Left(r2))
                => Some(Self::Left(*r1.min(r2))),
            (Self::Left(r1), Self::Right(l2))
                => (l2 < r1).then_some(Self::Range(*l2, *r1)),
            (Self::Left(r1), Self::Range(l2, r2))
                => if l2 >= r2 {
                    None
                } else if r2 <= r1 {
                    Some(Self::Range(*l2, *r2))
                } else if l2 <  r1 {
                    Some(Self::Range(*l2, *r1))
                } else {
                    None
                },
            (Self::Right(l1), Self::Left(r2))
                => (l1 < r2).then_some(Self::Range(*l1, *r2)),
            (Self::Right(l1), Self::Right(l2))
                => Some(Self::Right(*l1.max(l2))),
            (Self::Right(l1), Self::Range(l2, r2))
                => if l2 >= r2 {
                    None
                } else if l1 <= l2 {
                    Some(Self::Range(*l2, *r2))
                } else if l1 <  r2 {
                    Some(Self::Range(*l1, *r2))
                } else {
                    None
                },
            (Self::Range(l1, r1), Self::Left(r2))
                => if l1 >= r1 {
                    None
                } else if r1 <= r2 {
                    Some(Self::Range(*l1, *r1))
                } else if l1 <  r2 {
                    Some(Self::Range(*l1, *r2))
                } else {
                    None
                },
            (Self::Range(l1, r1), Self::Right(l2))
                => if l1 >= r1 {
                    None
                } else if l2 <= l1 {
                    Some(Self::Range(*l1, *r1))
                } else if l2 <  r1 {
                    Some(Self::Range(*l2, *r1))
                } else {
                    None
                },
            (Self::Range(l1, r1), Self::Range(l2, r2))
                => if l1 >= r1 || l2 >= r2 {
                    None
                } else if l1 <= l2 && r1 >= r2 {
                    Some(Self::Range(*l2, *r2))
                } else if l2 <= l1 && r2 >= r1 {
                    Some(Self::Range(*l1, *r1))
                } else if r1 >  l2 {
                    Some(Self::Range(*l2, *r1))
                } else if r2 >  l1 {
                    Some(Self::Range(*l1, *r2))
                } else {
                    None
                },
            (_, Self::Union(ul, ur))
                => match (self.intersection(ul), self.intersection(ur)) {
                    (None, None) => None,
                    (None, Some(r)) => Some(r),
                    (Some(l), None) => Some(l),
                    (Some(l), Some(r)) => Some(Self::Union(l.into(), r.into())),
                },
            (Self::Union(ul, ur), _)
                => match (ul.intersection(other), ur.intersection(other)) {
                    (None, None) => None,
                    (None, Some(r)) => Some(r),
                    (Some(l), None) => Some(l),
                    (Some(l), Some(r)) => Some(Self::Union(l.into(), r.into())),
                },
        }
    }

    /// Return `true` if `self` contains `k`.
    pub fn contains(&self, k: usize) -> bool {
        match self {
            Self::Left(r) => k < *r,
            Self::Right(l) => *l <= k,
            Self::Range(l, r) => *l <= k && k < *r,
            Self::Union(l, r) => l.contains(k) || r.contains(k),
        }
    }

    /// Returns the number of qubits contained in `self`, with consideration for
    /// fixed total system size `n`.
    pub fn size(&self, n: usize) -> usize {
        match self {
            Self::Left(r) => (*r).min(n),
            Self::Right(l) => n.saturating_sub(*l),
            Self::Range(l, r) => (*r).min(n).saturating_sub(*l),
            Self::Union(l, r)
                => l.size(n) + r.size(n)
                - l.intersection(r).map(|ix| ix.size(n)).unwrap_or(0),
        }
    }
}

