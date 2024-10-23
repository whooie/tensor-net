#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ SeedableRng, rngs::StdRng, thread_rng };
use tensor_net::{ circuit::*, gate::*, mps::* };

const N: usize = 8;
const DT: usize = 2;
const TARGET_X: (usize, usize) = (N / 2, N / 2);
const MC: usize = 10000;

const EPSILON: f64 = 1e-12;

fn apply_layer(state: &mut MPS<Q, C64>, unis: &[(usize, nd::Array2<C64>)]) {
    for (k, uni) in unis.iter() {
        state.apply_unitary2(*k, uni).unwrap();
    }
}

#[derive(Copy, Clone, Debug)]
struct Data {
    p00: f64,
    p01: f64,
    p10: f64,
    p11: f64,
}

impl Data {
    fn entropy(&self) -> f64 {
        (if self.p00 > 0.0 { -self.p00 * self.p00.ln() } else { 0.0 })
        + (if self.p01 > 0.0 { -self.p01 * self.p01.ln() } else { 0.0 })
        + (if self.p10 > 0.0 { -self.p10 * self.p10.ln() } else { 0.0 })
        + (if self.p11 > 0.0 { -self.p11 * self.p11.ln() } else { 0.0 })
    }
}

fn main() {
    // let mut rng = StdRng::seed_from_u64(10546);
    let mut rng = thread_rng();

    let init_layers: Vec<Vec<(usize, nd::Array2<C64>)>> =
        (0..2 * N).map(|t| {
            TileQ2::new(t % 2 == 1, N)
                .map(|k| (k, haar(2, &mut rng)))
                .collect::<Vec<(usize, nd::Array2<C64>)>>()
        })
        .collect();
    let uni_layers: Vec<Vec<(usize, nd::Array2<C64>)>> =
        (0..DT + 1).map(|t| {
            TileQ2::new(t % 2 == 1, N)
                .map(|k| (k, haar(2, &mut rng)))
                .collect::<Vec<(usize, nd::Array2<C64>)>>()
        })
        .collect();

    // init
    let mut state: MPS<Q, C64> =
        MPS::new_qubits(N, Some(BondDim::Const(6))).unwrap();
        // MPS::new_qubits(N, None).unwrap();
    init_layers.iter()
        .for_each(|layer| { apply_layer(&mut state, layer); });
    let (x0, x1) = TARGET_X;
    let frozen = state; // ensure immutability

    // exact sampling
    let data_exact =
        if let Some(layer) = uni_layers.first() {
            let mut state = frozen.clone();
            apply_layer(&mut state, layer);
            let p_x0 = state.probs(x0).unwrap();
            if (p_x0[0] + p_x0[1] - 1.0).abs() >= EPSILON {
                panic!("bad probabilities");
            }

            let (p00, p01) =
                if p_x0[0] > 0.0 {
                    let mut state0 = state.clone();
                    state0.measure_postsel(x0, 0);
                    uni_layers[1..].iter()
                        .for_each(|layer| { apply_layer(&mut state0, layer); });
                    let p_0_x1 = state0.probs(x1).unwrap();
                    if (p_0_x1[0] + p_0_x1[1] - 1.0).abs() >= EPSILON {
                        panic!("bad probabilities");
                    }
                    (p_x0[0] * p_0_x1[0], p_x0[0] * p_0_x1[1])
                } else {
                    (0.0, 0.0)
                };

            let (p10, p11) =
                if p_x0[1] > 0.0 {
                    let mut state1 = state.clone();
                    state1.measure_postsel(x0, 1);
                    uni_layers[1..].iter()
                        .for_each(|layer| { apply_layer(&mut state1, layer); });
                    let p_1_x1 = state1.probs(x1).unwrap();
                    if (p_1_x1[0] + p_1_x1[1] - 1.0).abs() >= EPSILON {
                        panic!("bad probabilities");
                    }
                    (p_x0[1] * p_1_x1[0], p_x0[1] * p_1_x1[1])
                } else {
                    (0.0, 0.0)
                };

            Data { p00, p01, p10, p11 }
        } else { unreachable!() };

    // naive MC
    let mut n00: usize = 0;
    let mut n01: usize = 0;
    let mut n10: usize = 0;
    let mut n11: usize = 0;
    for _ in 0..MC {
        let mut s = frozen.clone();
        if let Some(layer) = uni_layers.first() {
            apply_layer(&mut s, layer);
            let m0 = s.measure(x0, &mut rng).unwrap();
            uni_layers[1..].iter()
                .for_each(|layer| { apply_layer(&mut s, layer); });
            let m1 = s.measure(x1, &mut rng).unwrap();
            match (m0, m1) {
                (0, 0) => { n00 += 1; },
                (0, 1) => { n01 += 1; },
                (1, 0) => { n10 += 1; },
                (1, 1) => { n11 += 1; },
                _ => { },
            }
        } else { unreachable!() }
    }
    let data_mc = Data {
        p00: n00 as f64 / MC as f64,
        p01: n01 as f64 / MC as f64,
        p10: n10 as f64 / MC as f64,
        p11: n11 as f64 / MC as f64,
    };

    println!("exact: {:.3?} {:.3}", data_exact, data_exact.entropy());
    println!("   mc: {:.3?} {:.3}", data_mc, data_mc.entropy());
}

