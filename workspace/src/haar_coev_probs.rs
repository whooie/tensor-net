use std::{
    path::PathBuf,
    sync::atomic::{ AtomicUsize, Ordering },
};
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, thread_rng };
use tensor_net::{
    circuit::{ Q, apply_bilayer, UniSeq, MeasSeq, Meas, load_cbor },
    mps::{ MPS, BondDim },
};
use whooie::write_npz;
use lib::haar::MiptManifest;

const RUNS: usize = 64;
const BONDS: &[Option<usize>] =
    &[
        Some(4),  Some(8),  Some(12), Some(16), Some(20),
        Some(24), Some(28), Some(32), Some(36), Some(40),
        Some(44), Some(48), Some(56), Some(64), Some(72),
        None,
    ];

type MeasRecord = Vec<Vec<Meas>>;
type ProbRecord = Vec<Vec<(usize, f64)>>;

#[derive(Clone, Debug, PartialEq)]
struct Trajectory {
    traj: MeasRecord,
    probs: Vec<ProbRecord>,
}

fn sample_trajectory(
    nqubits: usize,
    circ: (&[UniSeq], &[MeasSeq])
) -> Trajectory
{
    assert_eq!(circ.0.len(), circ.1.len());
    let mut rng = thread_rng();
    let mut state_q: MPS<Q, C64> =
        MPS::new_qubits(nqubits, None).unwrap();
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
    for chi in BONDS.iter().copied().flatten() {
        let bond = BondDim::Const(chi);
        let mut state_c: MPS<Q, C64> =
            MPS::new_qubits(nqubits, Some(bond)).unwrap();
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
    probs.push(probs_q);

    Trajectory { traj, probs }
}

fn main() {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(RUNS.min(32))
    //     .build_global()
    //     .unwrap();

    let outdir = PathBuf::from("/scratch/whuie2/haar_coev_probs");
    let output_id = format!("{:016x}", thread_rng().gen::<u64>());

    // parse cli args to open manifest/select circuit+p
    let mut args = std::env::args().skip(1);
    let manifest_file: String =
        args.next()
        .expect("missing manifest file");
    let manifest_file = PathBuf::from(manifest_file);
    let circuit_dir =
        manifest_file.parent()
        .map(PathBuf::from)
        .unwrap_or(PathBuf::from("/"));
    let circ: usize =
        args.next()
        .expect("missing circuit number")
        .parse::<usize>()
        .expect("invalid circuit number");
    let p: f64 =
        args.next()
        .expect("missing p value")
        .parse::<f64>()
        .expect("invalid p value");

    let manifest =
        MiptManifest::load(&manifest_file)
        .expect("failed to load circuit manifest file");
    if circ >= manifest.num_circs() {
        panic!("invalid circuit number {} for {} total circuits",
            circ, manifest.num_circs());
    }
    if !manifest.p_meas().contains(&p) {
        panic!("invalid p value {:.6} for batch values {:?}",
            p, manifest.p_meas());
    }
    println!("running circuit {}, p={:.6} of batch ID {}",
        circ, p, manifest.id());
    println!("output ID: {}", output_id);

    // output text formatting
    let w_run: usize = (RUNS as f64).log10().floor() as usize + 1;

    let unis_file = circuit_dir.join(manifest.unis_fname(circ));
    let unis: Vec<UniSeq> =
        load_cbor(&unis_file)
        .expect("failed to read unitaries");
    let meas_file = circuit_dir.join(manifest.meas_fname(p, circ));
    let meas: Vec<MeasSeq> =
        load_cbor(&meas_file)
        .expect("failed to read measurement locations");

    // meas_locs :: { measurement index, [depth, qubit index] }
    let meas_locs: nd::Array1<u8> =
        meas.iter().enumerate()
        .flat_map(|(depth, layer)| {
            layer.iter()
            .flat_map(move |meas| [depth as u8, meas.idx() as u8])
        })
        .collect();
    let num_meas = meas_locs.len() / 2;
    let meas_locs: nd::Array2<u8> =
        meas_locs.into_shape((num_meas, 2))
        .unwrap();
    // traj_data :: { run, measurement index }
    let mut traj_data: nd::Array2<u8> =
        nd::Array::zeros((RUNS, num_meas));
    // prob_data :: { run, chi, measurement index }
    let mut prob_data: nd::Array3<f64> =
        nd::Array::zeros((RUNS, BONDS.len(), num_meas));

    let run = AtomicUsize::new(0);
    nd::Zip::from(traj_data.outer_iter_mut())
        .and(prob_data.outer_iter_mut())
        .par_for_each(|mut traj_rec_r, mut prob_rec_r| {
            let Trajectory { traj, probs } =
                sample_trajectory(
                    manifest.nqubits(),
                    (unis.as_ref(), meas.as_ref())
                );

            assert_eq!(
                traj_rec_r.len(),
                traj.iter().map(|layer| layer.len()).sum(),
            );
            traj_rec_r.iter_mut()
                .zip(traj.iter().flatten())
                .for_each(|(traj_rec_rm, meas)| {
                    let meas_result =
                        match meas {
                            Meas::Rand(_) => unreachable!(),
                            Meas::Proj(_, out) => (*out) as u8,
                        };
                    *traj_rec_rm = meas_result;
                });

            assert_eq!(prob_rec_r.shape()[0], probs.len());
            prob_rec_r.outer_iter_mut()
                .zip(probs.iter())
                .for_each(|(mut prob_rec_rx, probs_x)| {
                    assert_eq!(
                        prob_rec_rx.len(),
                        probs_x.iter().map(|layer| layer.len()).sum(),
                    );
                    prob_rec_rx.iter_mut()
                    .zip(probs_x.iter().flatten())
                    .for_each(|(prob_rec_rxm, &(_, prob_xm))| {
                        *prob_rec_rxm = prob_xm;
                    });
                });

            let prev_run = run.fetch_add(1, Ordering::SeqCst);
            eprintln!("  {:w_run$} / {:w_run$} ", prev_run + 1, RUNS);
        });
    eprintln!();

    let fname =
        format!("\
            haar_coev_probs\
            _seed={}\
            _nqubits={}\
            _depth={}\
            _p={:.6}\
            _circ={}\
            _outid={}\
            .npz",
            manifest.seed(),
            manifest.nqubits(),
            manifest.depth(),
            p,
            circ,
            output_id,
        );
    write_npz!(
        outdir.join(fname),
        arrays: {
            "manifest_file" =>
                &manifest_file.to_str().unwrap().chars()
                .map(|c| c as i32)
                .collect::<nd::Array1<i32>>(),
            "seed" => &nd::array![manifest.seed()],
            "nqubits" => &nd::array![manifest.nqubits() as i32],
            "depth" => &nd::array![manifest.depth() as i32],
            "num_circs" => &nd::array![manifest.num_circs() as i32],
            "genid" =>
                &manifest.id().chars()
                .map(|c| c as i32)
                .collect::<nd::Array1<i32>>(),
            "p" => &nd::array![p],
            "circ" => &nd::array![circ as i32],
            "meas_locs" => &meas_locs,
            "traj_data" => &traj_data,
            "prob_data" => &prob_data,
        }
    );
}

