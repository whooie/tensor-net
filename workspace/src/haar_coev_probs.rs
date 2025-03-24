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

const N: usize = 14;
const T: usize = 10 * N;
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
    None, Some(4), Some(8), Some(10), Some(12), Some(16), Some(20), Some(32),
];

type MeasRecord = Vec<Vec<Meas>>;
type ProbRecord = Vec<Vec<(usize, f64)>>;

#[derive(Clone, Debug, PartialEq)]
struct Trajectory {
    traj: MeasRecord,
    probs: Vec<ProbRecord>,
}

fn sample_trajectory(circ: (&[UniLayer], &[MeasLayer])) -> Trajectory {
    assert_eq!(circ.0.len(), circ.1.len());
    let mut rng = thread_rng();
    let mut state_q: MPS<Q, C64> = MPS::new_qubits(N, None).unwrap();
    let (traj, probs_q): (MeasRecord, ProbRecord) =
        circ.0.iter().zip(circ.1.iter())
        .map(|(unis, meas)| {
            apply_main_layer_probs(&mut state_q, unis, meas, &mut rng)
        })
        .unzip();

    let mut probs: Vec<ProbRecord> = Vec::with_capacity(BONDS.len());
    probs.push(probs_q);

    for chi in BONDS.iter().copied().flatten() {
        let bond = BondDim::Const(chi);
        let mut state_c: MPS<Q, C64> = MPS::new_qubits(N, Some(bond)).unwrap();
        let (_, probs_c): (Vec<_>, ProbRecord) =
            circ.0.iter().zip(traj.iter())
            .map(|(unis, meas)| {
                apply_main_layer_probs(&mut state_c, unis, meas, &mut rng)
            })
            .unzip();
        probs.push(probs_c);
    }

    Trajectory { traj, probs }
}

fn main() {
    // RNG for circuit generation only
    const SEED: u64 = 10546;
    // let mut rng = thread_rng();
    let mut rng = StdRng::seed_from_u64(SEED);

    let n_p = P_MEAS.len();
    let n_b = BONDS.iter().flatten().count() + 1;

    // output text formatting
    let w_c: usize = (CIRCS as f64).log10().floor() as usize + 1;
    let w_p: usize = (P_MEAS.len() as f64).log10().floor() as usize + 1;
    let w_run: usize = (RUNS as f64).log10().floor() as usize + 1;

    let outdir = PathBuf::from("output").join("haar_coev_probs");

    eprint!(" {:w_c$} / {:w_c$} ", 0, CIRCS);
    for i in 0 .. CIRCS {
        eprint!("\x1b[{}D{:w_c$} / {:w_c$} ",
            2 * w_c + 4, i + 1, CIRCS);

        // traj_data :: { p, run, t, x }
        let mut traj_data: nd::Array4<i8> =
            nd::Array::zeros((n_p, RUNS, T, N));
        // prob_data :: { p, run, chi, t, x }
        let mut prob_data: nd::Array5<f64> =
            nd::Array::from_elem((n_p, RUNS, n_b, T, N), -1.0);

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
            traj_data.outer_iter_mut()
            .zip(prob_data.outer_iter_mut())
            .zip(meas_p.iter())
            .enumerate();
        for (j, ((mut traj_p, mut prob_p), meas_p)) in p_iter {
            eprint!("\x1b[{}D{:w_p$} / {:w_p$} ",
                2 * w_p + 4, j + 1, n_p);

            let mut run = AtomicUsize::new(0);
            eprint!(" {:w_run$} / {:w_run$} ", run.get_mut(), RUNS);
            nd::Zip::from(traj_p.outer_iter_mut())
                .and(prob_p.outer_iter_mut())
                .par_for_each(|mut traj_pr, mut prob_pr| {
                    let Trajectory { traj, probs } =
                        sample_trajectory((&unis, meas_p));
                    traj.into_iter()
                        .zip(traj_pr.outer_iter_mut())
                        .for_each(|(meas_layer, mut meas_rec)| {
                            meas_layer.into_iter()
                                .for_each(|m| {
                                    match m {
                                        Meas::Rand(_) => { },
                                        Meas::Postsel(k, out) => {
                                            meas_rec[k] =
                                                if out { 1 } else { -1 };
                                        },
                                    }
                                });
                        });
                    probs.into_iter()
                        .zip(prob_pr.outer_iter_mut())
                        .for_each(|(prob_data_prx, mut prob_rec_prx)| {
                            prob_data_prx.into_iter()
                                .zip(prob_rec_prx.outer_iter_mut())
                                .for_each(|(prob_layer, mut prob_rec)| {
                                    prob_layer.into_iter()
                                        .for_each(|(k, p)| { prob_rec[k] = p; });
                                });
                        });
                    let prev_run = run.fetch_add(1, Ordering::SeqCst);
                    eprint!("\x1b[{}D{:w_run$} / {:w_run$} ",
                        2 * w_run + 4, prev_run + 1, RUNS);
                });
            eprint!("\x1b[{}D", 2 * w_run + 5);
        }
        eprint!("\x1b[{}D", 2 * w_p + 5);
        let fname =
            format!("haar_coev_probs_n={}_d={}_runs={}_seed={}_circ={}.npz",
                N, T, RUNS, SEED, i);
        write_npz!(
            outdir.join(fname),
            arrays: {
                "size" => &nd::array![N as i32],
                "depth" => &nd::array![T as i32],
                "circ" => &nd::array![i as i32],
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
                "traj" => &traj_data,
                "prob" => &prob_data,
                "seed" => &nd::array![SEED as i32],
            }
        );
    }
    eprintln!();


}

