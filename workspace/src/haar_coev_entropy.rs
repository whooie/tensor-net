#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::{
    ops::Range,
    path::PathBuf,
    sync::atomic::{ AtomicUsize, Ordering },
};
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ SeedableRng, rngs::StdRng, thread_rng };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_net::{ circuit::Q, mps::* };
use whooie::write_npz;
use lib::{ *, haar::* };

const N: usize = 12;
const T: usize = 4 * N;
const CIRCS: usize = 20;
const RUNS: usize = 500;
const P_MEAS: &[f64] = &[
    0.010, 0.020, 0.030, 0.040, 0.050, 0.070, 0.090,
    0.100, 0.115, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190,
    0.210, 0.225, 0.250, 0.275,
    0.300, 0.325, 0.350, 0.375,
    0.400,
];
const BONDS: &[Option<usize>] = &[
    None,
    Some(4), Some(6), Some(8), Some(10),
    Some(12), Some(16), Some(20), Some(32),
];

type MeasRecord = Vec<Vec<Meas>>;

fn sample_entropy(circ: (&[UniLayer], &[MeasLayer])) -> Vec<f64> {
    assert_eq!(circ.0.len(), circ.1.len());
    let mut rng = thread_rng();
    let mut state_q: MPS<Q, C64> = MPS::new_qubits(N, None).unwrap();
    let traj: MeasRecord =
        circ.0.iter().zip(circ.1.iter())
        .map(|(unis, meas)| {
            apply_main_layer(&mut state_q, unis, meas, &mut rng)
        })
        .collect();
    let entropy_q = state_q.entropy_vn(N / 2).unwrap();

    let mut entropies: Vec<f64> = Vec::with_capacity(BONDS.len());
    entropies.push(entropy_q);

    for chi in BONDS.iter().copied().flatten() {
        let bond = BondDim::Const(chi);
        let mut state_c: MPS<Q, C64> = MPS::new_qubits(N, Some(bond)).unwrap();
        circ.0.iter().zip(traj.iter())
            .for_each(|(unis, meas)| {
                apply_main_layer(&mut state_c, unis, meas, &mut rng);
            });
        let entropy_c = state_c.entropy_vn(N / 2).unwrap();
        entropies.push(entropy_c);
    }

    entropies
}

fn main() {
    // RNG for circuit generation only
    const SEED: u64 = 10546;
    // let mut rng = thread_rng();
    let mut rng = StdRng::seed_from_u64(SEED);

    let n_p = P_MEAS.len();
    let n_b = BONDS.iter().flatten().count() + 1;

    // entropy_data :: { circ, p, run, chi }
    let mut entropy_data: nd::Array4<f64> =
        nd::Array::from_elem((CIRCS, n_p, RUNS, n_b), -1.0);

    // output text formatting
    let w_c: usize = (CIRCS as f64).log10().floor() as usize + 1;
    let w_p: usize = (P_MEAS.len() as f64).log10().floor() as usize + 1;
    let w_run: usize = (RUNS as f64).log10().floor() as usize + 1;

    let outdir = PathBuf::from("output").join("haar_coev_entropy");

    eprint!(" {:w_c$} / {:w_c$} ", 0, CIRCS);
    for (i, mut entropy_c) in entropy_data.outer_iter_mut().enumerate() {
        eprint!("\x1b[{}D{:w_c$} / {:w_c$} ",
            2 * w_c + 4, i + 1, CIRCS);

        let unis: Vec<UniLayer> =
            (0..T)
            .map(|t| UniLayer::gen(N, t % 2 == 1, &mut rng))
            .collect();
        let meas_p: Vec<Vec<MeasLayer>> =
            P_MEAS.iter().copied()
            .map(|p| (0..T).map(|_| MeasLayer::gen(N, p, &mut rng)).collect())
            .collect();

        eprint!(" {:w_p$} / {:w_p$} ", 0, n_p);
        let p_iter =
            entropy_c.outer_iter_mut()
            .zip(meas_p.iter())
            .enumerate();
        for (j, (mut entropy_cp, meas_p)) in p_iter {
            eprint!("\x1b[{}D{:w_p$} / {:w_p$} ",
                2 * w_p + 4, j + 1, n_p);

            let mut run = AtomicUsize::new(0);
            eprint!(" {:w_run$} / {:w_run$} ", run.get_mut(), RUNS);
            nd::Zip::from(entropy_cp.outer_iter_mut())
                .par_for_each(|mut entropy_cpr| {
                    let entropies = sample_entropy((&unis, meas_p));
                    entropies.into_iter()
                        .zip(entropy_cpr.iter_mut())
                        .for_each(|(entropy_b, entropy_cprx)| {
                            *entropy_cprx = entropy_b;
                        });
                    let prev_run = run.fetch_add(1, Ordering::SeqCst);
                    eprint!("\x1b[{}D{:w_run$} / {:w_run$} ",
                        2 * w_run + 4, prev_run + 1, RUNS);
                });
            eprint!("\x1b[{}D", 2 * w_run + 5);
        }
        eprint!("\x1b[{}D", 2 * w_p + 5);
    }
    eprintln!();

    let fname =
        format!("haar_coev_entropy_n={}_d={}_runs={}_seed={}.npz",
            N, T, RUNS, SEED);
    write_npz!(
        outdir.join(fname),
        arrays: {
            "size" => &nd::array![N as i32],
            "depth" => &nd::array![T as i32],
            "circs" => &nd::array![CIRCS as i32],
            "runs" => &nd::array![RUNS as i32],
            "p_meas" =>
                &P_MEAS.iter().copied()
                .collect::<nd::Array1<f64>>(),
            "chi" =>
                &[0_i32].into_iter()
                .chain(
                    BONDS.iter().copied()
                    .filter_map(|mb_chi| mb_chi.map(|x| x as i32))
                )
                .collect::<nd::Array1<i32>>(),
            "entropy" => &entropy_data,
            "seed" => &nd::array![SEED as i32],
        }
    );
}

