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
use tensor_net::{
    circuit::{ Q, Meas, Outcome, UniSeq, MeasSeq, Elements, apply_bilayer },
    mps::*,
};
use whooie::write_npz;

const N: usize = 12;
const T: usize = 10 * N;
const RUNS: usize = 64;
const BONDS: &[Option<usize>] = &[
    None,
    Some(4),  Some(8),  Some(12), Some(16), Some(20),
    Some(24), Some(28), Some(32), Some(36), Some(40),
    Some(44), Some(48), Some(56), Some(64), Some(72),
];
const CIRCUIT_SEED: u64 = 10546;

type MeasRecord = Vec<Vec<Meas>>;
type ProbRecord = Vec<Vec<(usize, f64)>>;

#[derive(Clone, Debug, PartialEq)]
struct Trajectory {
    traj: MeasRecord,
    probs: Vec<ProbRecord>,
}

fn sample_trajectory(circ: (&[UniSeq], &[MeasSeq])) -> Trajectory {
    assert_eq!(circ.0.len(), circ.1.len());
    let mut rng = thread_rng();
    let mut state_q: MPS<Q, C64> = MPS::new_qubits(N, None).unwrap();
    let (traj, probs_q): (MeasRecord, ProbRecord) =
        circ.0.iter().zip(circ.1.iter())
        .map(|(unis, meas)| {
            let mut outcome_buf: Vec<Meas> = Vec::new();
            let mut prob_buf: Vec<(usize, f64)> = Vec::new();
            apply_bilayer(
                &mut state_q,
                unis,
                meas,
                Some(&mut outcome_buf),
                Some(&mut prob_buf),
                &mut rng,
            ).unwrap();
            (outcome_buf, prob_buf)
        })
        .unzip();

    let mut probs: Vec<ProbRecord> = Vec::with_capacity(BONDS.len());
    probs.push(probs_q);

    for chi in BONDS.iter().copied().flatten() {
        let bond = BondDim::Const(chi);
        let mut state_c: MPS<Q, C64> = MPS::new_qubits(N, Some(bond)).unwrap();
        let probs_c: ProbRecord =
            circ.0.iter().zip(traj.iter())
            .map(|(unis, meas)| {
                let mut prob_buf: Vec<(usize, f64)> = Vec::new();
                apply_bilayer(
                    &mut state_c,
                    unis, meas,
                    None,
                    Some(&mut prob_buf),
                    &mut rng,
                ).unwrap();
                prob_buf
            })
            .collect();
        probs.push(probs_c);
    }

    Trajectory { traj, probs }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Instance {
    Unis,
    Meas(f64),
    Output(f64),
}

impl Instance {
    fn fmt(
        &self,
        circ: usize,
        task_id: Option<&str>,
        npz: bool,
    ) -> String
    {
        let npz_ext = if npz { ".npz" } else { "" };
        match (self, task_id) {
            (Self::Unis, None) =>
                format!("unis_seed={}_n={}_depth={}_circ={}{}",
                    CIRCUIT_SEED, N, T, circ, npz_ext),
            (Self::Meas(p), None) =>
                format!("meas_seed={}_n={}_depth={}_p={:.6}_circ={}{}",
                    CIRCUIT_SEED, N, T, p, circ, npz_ext),
            (Self::Output(p), None) =>
                format!("haar_coev_probs_seed={}_n={}_depth={}_p={:.6}_circ={}{}",
                    CIRCUIT_SEED, N, T, p, circ, npz_ext),
            (Self::Unis, Some(id)) =>
                format!("unis_seed={}_n={}_depth={}_circ={}_id={}{}",
                    CIRCUIT_SEED, N, T, circ, id, npz_ext),
            (Self::Meas(p), Some(id)) =>
                format!("meas_seed={}_n={}_depth={}_p={:.6}_circ={}_id={}{}",
                    CIRCUIT_SEED, N, T, p, circ, id, npz_ext),
            (Self::Output(p), Some(id)) =>
                format!("haar_coev_probs_seed={}_n={}_depth={}_p={:.6}_circ={}_id={}{}",
                    CIRCUIT_SEED, N, T, p, circ, id, npz_ext),
        }
    }
}

fn main() {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(RUNS.min(32))
    //     .build_global()
    //     .unwrap();

    let circuit_dir = PathBuf::from("output/haar_circuits");
    let outdir = PathBuf::from("/scratch/whuie2/haar_coev_probs");
    let task_id = Alphanumeric.sample_string(&mut thread_rng(), 10);

    let mut args = std::env::args().skip(1);

    let circ: usize =
        args.next()
        .expect("missing circuit number")
        .parse::<usize>()
        .expect("invalid circuit number");
    if !circuit_dir.join(Instance::Unis.fmt(circ, None, false)).is_file() {
        panic!("missing unitaries file for circuit {}", circ);
    }

    let p: f64 =
        args.next()
        .expect("missing p value")
        .parse::<f64>()
        .expect("invalid p value");
    if !circuit_dir.join(Instance::Meas(p).fmt(circ, None, false)).is_file() {
        panic!("missing measurement locations file for circuit {}", circ);
    }
    println!("running circuit {}, p={:.6}", circ, p);

    let n_b = BONDS.iter().flatten().count() + 1;

    // output text formatting
    let w_run: usize = (RUNS as f64).log10().floor() as usize + 1;

    // traj_data :: { run, t, x }
    let mut traj_data: nd::Array3<i8> =
        nd::Array::zeros((RUNS, T, N));
    // prob_data :: { run, chi, t, x }
    let mut prob_data: nd::Array4<f64> =
        nd::Array::from_elem((RUNS, n_b, T, N), -1.0);

    let unis: Elements<UniSeq> = Elements::load(
        circuit_dir.join(Instance::Unis.fmt(circ, None, false))).unwrap();
    let meas: Elements<MeasSeq> = Elements::load(
        circuit_dir.join(Instance::Meas(p).fmt(circ, None, false))).unwrap();

    let run = AtomicUsize::new(0);
    nd::Zip::from(traj_data.outer_iter_mut())
        .and(prob_data.outer_iter_mut())
        .par_for_each(|mut traj_pr, mut prob_r| {
            let Trajectory { traj, probs } =
                sample_trajectory((unis.as_ref(), meas.as_ref()));
            traj.into_iter()
            .zip(traj_pr.outer_iter_mut())
            .for_each(|(meas_layer, mut meas_rec)| {
                meas_layer.into_iter()
                .for_each(|m| {
                    match m {
                        Meas::Rand(_) => { },
                        Meas::Proj(k, out) => {
                            meas_rec[k] =
                                match out {
                                    Outcome::Zero => -1,
                                    Outcome::One => 1,
                                };
                        },
                    }
                });
            });
            probs.into_iter()
            .zip(prob_r.outer_iter_mut())
            .for_each(|(prob_data_prx, mut prob_rec_prx)| {
                prob_data_prx.into_iter()
                .zip(prob_rec_prx.outer_iter_mut())
                .for_each(|(prob_layer, mut prob_rec)| {
                    prob_layer.into_iter()
                    .for_each(|(k, p)| { prob_rec[k] = p; });
                });
            });
            let prev_run = run.fetch_add(1, Ordering::SeqCst);
            eprintln!("  {:w_run$} / {:w_run$} ", prev_run + 1, RUNS);
        });
    let fname = Instance::Output(p).fmt(circ, Some(&task_id), true);
    write_npz!(
        outdir.join(fname),
        arrays: {
            "size" => &nd::array![N as i32],
            "depth" => &nd::array![T as i32],
            "p" => &nd::array![p],
            "circ" => &nd::array![circ as i32],
            "runs" => &nd::array![RUNS as i32],
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
    eprintln!();
}

