use std::{
    path::PathBuf,
    sync::atomic::{ AtomicUsize, Ordering },
};
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{
    thread_rng,
    distributions::{ Alphanumeric, DistString },
};
// use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_net::{ circuit::Q, mps::* };
use whooie::write_npz;
use lib::{ *, haar::* };

const N: usize = 6;
const T: usize = 10 * N;
const CIRCS: usize = 30;
const P_MEAS: &[f64] = &[
    0.010, 0.020, 0.030, 0.040, 0.050, 0.070, 0.090, 0.100, 0.115,
    0.130, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180,
    0.190, 0.210, 0.225, 0.250, 0.275,
    0.300, 0.325, 0.350, 0.375,
    0.400,
];
const RUNS: usize = 32;
const BONDS: &[Option<usize>] = &[
    None,
    Some(4),  Some(6),  Some(8),  Some(10), Some(12),
    Some(16), Some(20), Some(24), Some(28), Some(32),
];
const CIRCUIT_SEED: u64 = 10546;

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Instance {
    Unis,
    Meas,
    Output,
}

impl Instance {
    fn fmt(&self, circ: usize, task_id: Option<&str>) -> String {
        match (self, task_id) {
            (Self::Unis, None) =>
                format!("unis_seed={}_n={}_depth={}_circ={}.npz",
                    CIRCUIT_SEED, N, T, circ),
            (Self::Meas, None) =>
                format!("meas_seed={}_n={}_depth={}_plen={}_circ={}.npz",
                    CIRCUIT_SEED, N, T, P_MEAS.len(), circ),
            (Self::Output, None) =>
                format!("haar_coev_probs_seed={}_n={}_depth={}_circ={}.npz",
                    CIRCUIT_SEED, N, T, circ),
            (Self::Unis, Some(id)) =>
                format!("unix_seed={}_n={}_depth={}_circ={}_id={}.npz",
                    CIRCUIT_SEED, N, T, circ, id),
            (Self::Meas, Some(id)) =>
                format!("meas_seed={}_n={}_depth={}_plen={}_circ={}_id={}.npz",
                    CIRCUIT_SEED, N, T, P_MEAS.len(), circ, id),
            (Self::Output, Some(id)) =>
                format!("haar_coev_probs_seed={}_n={}_depth={}_circ={}_id={}.npz",
                    CIRCUIT_SEED, N, T, circ, id),
        }
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(RUNS.min(32))
        .build_global()
        .unwrap();

    let circuit_dir = PathBuf::from("output/haar_circuits");
    let outdir = PathBuf::from("/scratch/whuie2/haar_coev_probs");
    // let task_id =
    //     env::var("SLURM_ARRAY_TASK_ID").unwrap()
    //     .parse::<usize>().unwrap();
    let task_id = Alphanumeric.sample_string(&mut thread_rng(), 10);

    for i in 0..CIRCS {
        if !circuit_dir.join(Instance::Unis.fmt(i, None)).is_file() {
            panic!("missing unitaries file for circuit {}", i);
        }
        if !circuit_dir.join(Instance::Meas.fmt(i, None)).is_file() {
            panic!("missing measurement locations file for circuit {}", i);
        }
    }

    let n_p = P_MEAS.len();
    let n_b = BONDS.iter().flatten().count() + 1;

    // output text formatting
    let w_c: usize = (CIRCS as f64).log10().floor() as usize + 1;
    let w_p: usize = (P_MEAS.len() as f64).log10().floor() as usize + 1;
    let w_run: usize = (RUNS as f64).log10().floor() as usize + 1;

    eprint!(" {:w_c$} / {:w_c$} ", 0, CIRCS);
    for i in 0..CIRCS {
        eprint!("\x1b[{}D{:w_c$} / {:w_c$} ",
            2 * w_c + 4, i + 1, CIRCS);

        // traj_data :: { p, run, t, x }
        let mut traj_data: nd::Array4<i8> =
            nd::Array::zeros((n_p, RUNS, T, N));
        // prob_data :: { p, run, chi, t, x }
        let mut prob_data: nd::Array5<f64> =
            nd::Array::from_elem((n_p, RUNS, n_b, T, N), -1.0);

        let unis = load_unis(circuit_dir.join(Instance::Unis.fmt(i, None)));
        let (p, meas) = load_meas(circuit_dir.join(Instance::Meas.fmt(i, None)));
        assert!(p.len() == P_MEAS.len() && p.iter().all(|p| P_MEAS.contains(p)));

        eprint!(" {:w_p$} / {:w_p$} ", 0, n_p);
        let p_iter =
            traj_data.outer_iter_mut()
            .zip(prob_data.outer_iter_mut())
            .zip(meas.iter())
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
        let fname = Instance::Output.fmt(i, Some(&task_id));
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
                "seed" => &nd::array![CIRCUIT_SEED as i32],
            }
        );
    }
    eprintln!();
}

