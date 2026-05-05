use std::{
    path::PathBuf,
    sync::atomic::{ AtomicUsize, Ordering },
};
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, thread_rng };
use tensor_net::{
    circuit::{ Q, apply_bilayer, UniSeq, MeasSeq, load_cbor },
    mps::{ MPS, BondDim },
};
use whooie::write_npz;
use lib::mipt::TwoPointManifest;

const RUNS: usize = 64;
const BONDS: &[usize] = &[
    8,   16,  24,  32,  40,
    48,  56,  64,  72,  80,
    88,  96,  104, 112, 120,
    128, 136, 144, 152, 160,
    168, 176, 184, 192, 200,
];

fn circuit_state<R>(
    nqubits: usize,
    circ: (&[UniSeq], &[MeasSeq]),
    bond: Option<usize>,
    rng: &mut R
) -> MPS<Q, C64>
where R: Rng + ?Sized
{
    assert_eq!(circ.0.len(), circ.1.len());
    let mut state: MPS<Q, C64> =
        MPS::new_qubits(nqubits, bond.map(BondDim::Const)).unwrap();
    circ.0.iter().zip(circ.1.iter())
        .for_each(|(unis, meas)| {
            apply_bilayer(&mut state, unis, meas, None, None, rng).unwrap()
        });
    state
}

#[derive(Copy, Clone, Debug)]
struct MeasDist {
    p00: f64,
    p01: f64,
    p10: f64,
    p11: f64,
}

fn apply_uni_layer<R>(state: &mut MPS<Q, C64>, layer: &UniSeq, rng: &mut R)
where R: Rng + ?Sized
{
    for uni in layer.iter() { state.apply_uni_rng(uni, rng).unwrap(); }
}

fn apply_uni_circ<R>(state: &mut MPS<Q, C64>, layers: &[UniSeq], rng: &mut R)
where R: Rng + ?Sized
{
    for layer in layers.iter() { apply_uni_layer(state, layer, rng); }
}

fn check_probs(probs: &[f64], label: &str) {
    const EPSILON: f64 = 1e-12;
    if (probs.iter().copied().sum::<f64>() - 1.0).abs() >= EPSILON {
        eprintln!("\n{:?}", probs);
        panic!("bad probabilities! ({})", label);
    }
}

fn do_probe<R>(
    mut state: MPS<Q, C64>,
    unis: &[UniSeq],
    x0: usize,
    x1: usize,
    rng: &mut R,
) -> MeasDist
where R: Rng + ?Sized
{
    assert!(unis.len() > 1);
    let Some(layer0) = unis.first() else { unreachable!() };
    apply_uni_layer(&mut state, layer0, rng);
    let state = state;

    let p_x0 = state.probs(x0).unwrap();
    check_probs(&p_x0, "base-x0");

    let (p00, p01) =
        if p_x0[0] > 0.0 {
            let mut state_0 = state.clone();
            state_0.measure_postsel_prob(x0, 0);
            apply_uni_circ(&mut state_0, &unis[1..], rng);
            let p_0_x1 = state_0.probs(x1).unwrap();
            check_probs(&p_0_x1, "proj-x0=0");
            (p_x0[0] * p_0_x1[0], p_x0[0] * p_0_x1[1])
        } else {
            (0.0, 0.0)
        };

    let (p10, p11) =
        if p_x0[1] > 0.0 {
            let mut state_1 = state;
            state_1.measure_postsel_prob(x0, 1);
            apply_uni_circ(&mut state_1, &unis[1..], rng);
            let p_1_x1 = state_1.probs(x1).unwrap();
            check_probs(&p_1_x1, "proj-x0=1");
            (p_x0[1] * p_1_x1[0], p_x0[1] * p_1_x1[1])
        } else {
            (0.0, 0.0)
        };

    let norm = p00 + p01 + p10 + p11;
    MeasDist {
        p00: p00 / norm,
        p01: p01 / norm,
        p10: p10 / norm,
        p11: p11 / norm,
    }
}

fn main() {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(RUNS.min(32))
    //     .build_global()
    //     .unwrap();

    let outdir = PathBuf::from("/scratch/whuie2/haar_two_point");
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
        TwoPointManifest::load(&manifest_file)
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
    let probe_file = circuit_dir.join(manifest.probe_fname(circ));
    let probe: Vec<UniSeq> =
        load_cbor(&probe_file)
        .expect("failed to read probe unitaries");
    let meas_file = circuit_dir.join(manifest.meas_fname(p, circ));
    let meas: Vec<MeasSeq> =
        load_cbor(&meas_file)
        .expect("failed to read measurement locations");

    // dist :: { run, chi, [p00, p01, p10, p11] }
    let mut dist: nd::Array3<f64> =
        nd::Array::zeros((RUNS, BONDS.len(), 4));

    // dist_inf :: { run, [p00, p01, p10, p11] }
    let mut dist_inf: nd::Array2<f64> =
        nd::Array::zeros((RUNS, 4));

    let nqubits = manifest.nqubits();
    let (x0, x1) = manifest.xx();
    let completed = AtomicUsize::new(0);
    nd::Zip::from(dist.outer_iter_mut())
        .and(dist_inf.outer_iter_mut())
        .par_for_each(|mut d_r, mut dinf_r| {
            let mut rng = thread_rng();
            let state = circuit_state(nqubits, (&unis, &meas), None, &mut rng);
            let dist_inf = do_probe(state, &probe, x0, x1, &mut rng);
            dinf_r[0] = dist_inf.p00;
            dinf_r[1] = dist_inf.p01;
            dinf_r[2] = dist_inf.p10;
            dinf_r[3] = dist_inf.p11;

            for (chi, mut d_rx) in BONDS.iter().copied().zip(d_r.outer_iter_mut()) {
                let state =
                    circuit_state(nqubits, (&unis, &meas), Some(chi), &mut rng);
                let dist = do_probe(state, &probe, x0, x1, &mut rng);
                d_rx[0] = dist.p00;
                d_rx[1] = dist.p01;
                d_rx[2] = dist.p10;
                d_rx[3] = dist.p11;
            }

            let prev_run = completed.fetch_add(1, Ordering::SeqCst);
            eprintln!("  {:w_run$} / {:w_run$} ", prev_run + 1, RUNS);
        });
    eprintln!();

    let fname =
        format!("\
            two_point_dist\
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
                &manifest_file.to_str().unwrap()
                .chars()
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
            "chi" =>
                &BONDS.iter()
                .map(|chi| *chi as i32)
                .collect::<nd::Array1<i32>>(),
            "dist_inf" => &dist_inf,
            "dist" => &dist,
        }
    );
}

