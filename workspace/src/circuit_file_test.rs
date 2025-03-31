#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ SeedableRng, rngs::StdRng, thread_rng };
use lib::{ *, haar::* };

const N: usize = 6;
const T: usize = 10 * N;
const CIRCS: usize = 20;
const P_MEAS: &[f64] = &[
    0.010, 0.020, 0.030, 0.040, 0.050, 0.070, 0.090,
    0.100, 0.115, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190,
    0.210, 0.225, 0.250, 0.275,
    0.300, 0.325, 0.350, 0.375,
    0.400,
];

fn main() {
    let mut rng = thread_rng();

    let unis_out = PathBuf::from("unis.npz");
    let unis: Vec<UniLayer> =
        (0..T)
        .map(|t| UniLayer::gen(N, t % 2 == 1, &mut rng))
        .collect();
    save_unis(&unis_out, &unis);
    let unis_read = load_unis(&unis_out);
    assert_eq!(unis, unis_read);

    let meas_out = PathBuf::from("meas.npz");
    let meas_p: Vec<Vec<MeasLayer>> =
        P_MEAS.iter().copied()
        .map(|p| (0..T).map(|_| MeasLayer::gen(N, p, &mut rng)).collect())
        .collect();
    save_meas(&meas_out, P_MEAS.iter().zip(&meas_p));
    let (p_read, meas_read) = load_meas(&meas_out);
    assert!(
        p_read.len() == P_MEAS.len()
        && p_read.iter().all(|p| P_MEAS.contains(p))
    );
    assert_eq!(meas_p, meas_read);
}

