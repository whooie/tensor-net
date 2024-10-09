#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ SeedableRng, rngs::StdRng, thread_rng };
use tensor_net::{ circuit::*, gate::*, mps::* };

// small-endian
fn bits(n: usize, k: usize) -> Vec<usize> {
    (0..n).map(|b| (k >> (n - 1 - b)) % 2).collect()
}

const N: usize = 5;

fn doit() -> Vec<usize> {
    // let mut rng = StdRng::seed_from_u64(10546);
    let mut rng = thread_rng();

    let mut apply_swap = |state: &mut MPS<Q, C64>, a: usize| {
        state
            .apply_gate(Gate::CX(a))
            .apply_gate(Gate::CXRev(a))
            .apply_gate(Gate::CX(a));
    };
    let mut apply_distant =
        |state: &mut MPS<Q, C64>, a: usize, b: usize, g: G2| {
            let n = a.min(b);
            let m = a.max(b);
            for k in n..m - 1 { apply_swap(state, k); }
            state.apply_gate(g.sample(m - 1, &mut rng));
            for k in (n..m - 1).rev() { apply_swap(state, k); }
        };

    let mut state: MPS<Q, C64> =
        MPS::new_qubits(N, None).unwrap();

    state.apply_gate(Gate::H(0));
    apply_distant(&mut state, 0, N - 1, G2::CX);
    state.apply_gate(Gate::H(2));
    state.apply_gate(Gate::CX(2));

    for k in 0..N - 1 { println!("{}: {:?}", k, state.entropy_vn(k)); }

    let (idx, state) = state.into_contract();
    for (k, ak) in state.iter().enumerate() {
        // println!("{:?}, {:+.5e}", bits(N, k), ak);
    }

    let mut state: MPS<Q, C64> =
        MPS::from_vector(idx, state, Some(BondDim::Const(1))).unwrap();
    for k in 0..N - 1 { println!("{}: {:?}", k, state.entropy_vn(k)); }

    let state_arr = state.contract();
    for (k, ak) in state_arr.iter().enumerate() {
        println!("{:?}: {:+.5e}", bits(N, k), ak);
    }

    let m: Vec<usize> =
        [2, 3].into_iter()
        .flat_map(|k| state.measure(k, &mut rng))
        .collect();
    println!("{:?}", m);
    state.measure(2, &mut rng);

    for k in 0..N - 1 { println!("{}: {:?}", k, state.entropy_vn(k)); }

    m
}

fn main() {
    let mut n0: usize = 0;
    let mut n1: usize = 0;
    for _ in 0..1 {
        let m = doit();
        if m == [0, 0] {
            n0 += 1;
        } else if m == [1, 1] {
            n1 += 1;
        } else {
            panic!("unexpected output {:?}", m);
        }
    }
    println!("n0: {n0}\nn1: {n1}");
}

