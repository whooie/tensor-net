#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray::{ self as nd, linalg::kron };
use num_complex::Complex64 as C64;
use rand::{ thread_rng, Rng };
use clifford_sim::{
    gate::{ Clifford, Gate },
    stab::{ Stab, Postsel },
};
use tensor_net::{
    circuit::{ Q, TileQ2 },
    gate::haar,
    mps::{ BondDim, MPS },
};

// small-endian
fn bits(n: usize, k: usize) -> Vec<usize> {
    (0..n).map(|b| (k >> (n - 1 - b)) % 2).collect()
}

fn gate_matrix(gate: &Gate) -> nd::Array2<C64> {
    use std::f64::consts::FRAC_1_SQRT_2;
    const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    const I: C64 = C64 { re: 0.0, im: 1.0 };
    const ORT2: C64 = C64 { re: FRAC_1_SQRT_2, im: 0.0 };
    const IORT2: C64 = C64 { re: 0.0, im: FRAC_1_SQRT_2 };
    let eye: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, ONE]];
    match gate {
        Gate::H(k) => {
            let mat: nd::Array2<C64> = nd::array![[ORT2, ORT2], [ORT2, -ORT2]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::X(k) => {
            let mat: nd::Array2<C64> = nd::array![[ZERO, ONE], [ONE, ZERO]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::Y(k) => {
            let mat: nd::Array2<C64> = nd::array![[ZERO, -I], [I, ZERO]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::Z(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, -ONE]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::S(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, I]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::SInv(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, -I]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::CX(a, _) => {
            if *a == 0 {
                nd::array![
                    [ONE,  ZERO, ZERO, ZERO],
                    [ZERO, ONE,  ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE ],
                    [ZERO, ZERO, ONE,  ZERO],
                ]
            } else {
                nd::array![
                    [ONE,  ZERO, ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE ],
                    [ZERO, ZERO, ONE,  ZERO],
                    [ZERO, ONE,  ZERO, ZERO],
                ]
            }
        },
        Gate::CZ(..) => {
            nd::array![
                [ONE,  ZERO, ZERO,  ZERO],
                [ZERO, ONE,  ZERO,  ZERO],
                [ZERO, ZERO, ONE,   ZERO],
                [ZERO, ZERO, ZERO, -ONE ],
            ]
        },
        Gate::Swap(..) => {
            nd::array![
                [ONE,  ZERO, ZERO, ZERO],
                [ZERO, ZERO, ONE,  ZERO],
                [ZERO, ONE,  ZERO, ZERO],
                [ZERO, ZERO, ZERO, ONE ],
            ]
        },
    }
}

fn get_probs_stab<R>(state: &Stab, target: usize, mc: u32, rng: &mut R)
    -> (f64, f64)
where R: Rng + ?Sized
{
    let mut n0: f64 = 0.0;
    let mut n1: f64 = 0.0;
    for _ in 0..mc {
        let outcome = state.clone().measure(target, rng);
        if outcome.is_0() { n0 += 1.0; } else { n1 += 1.0; }
    }
    (n0 / mc as f64, n1 / mc as f64)
}

fn simple() {
    let mut rng = thread_rng();

    let circuit: Vec<Gate> = 
        vec![
            Gate::H(0),
            Gate::S(1),
            Gate::CX(0, 1),
            Gate::H(0),
            Gate::CX(0, 1),
            Gate::H(0),
            Gate::S(1),
        ];
    // let circuit =
    //     Clifford::gen(2, &mut rng)
    //     .unpack().0;
    println!("{:?}", circuit);

    let circuit_mat: nd::Array2<C64> =
        circuit.iter()
        .map(gate_matrix)
        .fold(nd::Array2::eye(4), |acc, gate| gate.dot(&acc));
    println!("{:+.3}", circuit_mat);

    println!();

    let mut mps: MPS<Q, C64> = MPS::new_qubits(2, None).unwrap();
    mps.apply_unitary2(0, &circuit_mat).unwrap();
    // mps.measure_postsel(0, 1);
    // mps.apply_unitary2(0, &circuit_mat).unwrap();
    // mps.measure_postsel(0, 0);
    // mps.apply_unitary2(0, &circuit_mat).unwrap();
    println!("{:.3?}", mps.probs(0).unwrap());
    println!("{:.3?}", mps.probs(1).unwrap());
    let mps_arr = mps.contract();
    for (k, ak) in mps_arr.iter().enumerate() {
        println!("{:?}: {:+.5}", bits(2, k), ak);
    }

    println!();

    let mut stab = Stab::new(2);
    // println!("{}", stab.as_group());
    stab.apply_circuit(&circuit);
    // stab = stab.measure_postsel(0, Postsel::One).unwrap();
    // stab.apply_circuit(&circuit);
    // stab = stab.measure_postsel(0, Postsel::Zero).unwrap();
    // stab.apply_circuit(&circuit);
    println!("{}", stab.as_group());
    let stab_kets = stab.as_kets_cloned().unwrap();
    println!("{}", stab_kets);
    println!("{:?}", stab.probs(0));
    println!("{:?}", stab.probs(1));

}

fn cliff2<R>(offs: usize, rng: &mut R) -> (Vec<Gate>, nd::Array2<C64>)
where R: Rng + ?Sized
{
    let mut gates: Vec<Gate> = Clifford::gen(2, rng).unpack().0;
    let mat: nd::Array2<C64> =
        gates.iter()
        .map(gate_matrix)
        .fold(nd::Array2::eye(4), |acc, gate| gate.dot(&acc));
    gates.iter_mut().for_each(|g| { g.shift(offs); });
    (gates, mat)
}

fn probs_approx_eq(l: (f64, f64), r: (f64, f64)) -> bool {
    const EPSILON: f64 = 1e-12;
    (l.0 - r.0).abs() < EPSILON && (l.1 - r.1).abs() < EPSILON
}

fn check_probs(mps: &MPS<Q, C64>, stab: &mut Stab, n: usize) -> bool {
    for j in 0..n {
        let mps_probs = mps.prob(j, 0).zip(mps.prob(j, 1)).unwrap();
        let stab_probs = stab.probs(j);
        if !probs_approx_eq(mps_probs, stab_probs.into()) {
            println!("{} {:.3?} {:.3?}", j, mps_probs, stab_probs);
            let mps_arr = mps.contract();
            for (k, ak) in mps_arr.iter().enumerate() {
                println!("{:?}: {:+.5}", bits(n, k), ak);
            }
            let mut rng = thread_rng();
            let mc_stab_probs =
                get_probs_stab(stab, j, 100_000, &mut rng);
            println!("{:.3?}", mc_stab_probs);
            let stab_kets = stab.as_kets_cloned().unwrap();
            println!("{}", stab_kets);
            return false;
        }
    }
    true
}

fn running() {
    const RUNS: usize = 1_000_000;
    const N: usize = 3;
    
    let mut rng = thread_rng();
    let z = (RUNS as f64).log10().floor() as usize + 1;

    for k in 0..RUNS {
        println!("{:w$}", k, w=z);
        let (circ0, circ0_mat) = cliff2(0, &mut rng);
        let (circ1, circ1_mat) = cliff2(1, &mut rng);
        let (circ2, circ2_mat) = cliff2(0, &mut rng);
        println!("{:?}", circ0);
        println!("{:?}", circ1);
        println!("{:?}", circ2);

        let mut mps: MPS<Q, C64> = MPS::new_qubits(N, None).unwrap();
        let mut stab = Stab::new(N);

        mps.apply_unitary2(0, &circ0_mat).unwrap();
        mps.apply_unitary2(1, &circ1_mat).unwrap();
        stab.apply_circuit(&circ0);
        stab.apply_circuit(&circ1);

        // let mps_arr = mps.contract();
        // for (k, ak) in mps_arr.iter().enumerate() {
        //     println!("{:?}: {:+.5}", bits(N, k), ak);
        // }
        // let stab_kets = stab.as_kets_cloned().unwrap();
        // println!("{}", stab_kets);

        if !check_probs(&mps, &mut stab, N) { println!("(pre)"); return; }

        let p = mps.measure(1, &mut rng).unwrap();
        if p == 0 {
            stab = stab.measure_postsel(1, Postsel::Zero).unwrap();
        } else {
            stab = stab.measure_postsel(1, Postsel::One).unwrap();
        }

        if !check_probs(&mps, &mut stab, N) { println!("(init {})", p); return; }

        mps.apply_unitary2(0, &circ2_mat).unwrap();
        stab.apply_circuit(&circ2);

        if !check_probs(&mps, &mut stab, N) { println!("(final)"); return; }

    }
}

fn main() {
    // simple()
    running();
}

