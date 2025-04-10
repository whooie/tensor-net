#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::{ f64::consts::PI, path::PathBuf };
use ndarray as nd;
use num_complex::Complex64 as C64;
// use rand::{ SeedableRng, rngs::StdRng, thread_rng };
use tensor_net::{ circuit::*, gate::*, mps::* };

const N: usize = 5;
const ANGLE: f64 = 3.0 * PI / 5.0;
const EPSILON: f64 = 1e-12;

fn main() {
    let mut state: MPS<Q, C64> =
        // MPS::new_qubits(N, None).unwrap();
        MPS::new_qubits(N, Some(BondDim::Const(2))).unwrap();

    let xrot: nd::Array2<C64> = make_xrot(ANGLE);
    let cnot: nd::Array2<C64> = make_cx();
    state.apply_unitary1(2, &xrot).unwrap();
    state.apply_unitary2(2, &cnot).unwrap();

    let p0_expected: f64 = (ANGLE / 2.0).cos().powi(2);
    let p1_expected: f64 = (ANGLE / 2.0).sin().powi(2);
    for k in 0..N { println!("{}: {:.6?}", k, state.probs(k).unwrap()); }
    println!("expected p0: {}", p0_expected);
    println!("expected p1: {}", p1_expected);

    println!();

    let mut state0 = state.clone();
    state0.measure_postsel(2, 0);
    for k in 0..N { println!("{}: {:.6?}", k, state0.probs(k).unwrap()); }
    println!();
    let mut state1 = state.clone();
    state1.measure_postsel(2, 1);
    for k in 0..N { println!("{}: {:.6?}", k, state1.probs(k).unwrap()); }
}

