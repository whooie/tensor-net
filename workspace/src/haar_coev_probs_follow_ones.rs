use std::{
    path::{ Path, PathBuf },
    sync::atomic::{ AtomicUsize, Ordering },
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, thread_rng };
use tensor_net::{
    circuit::{ Q, apply_bilayer, UniSeq, MeasSeq, Meas, Outcome, load_cbor },
    mps::{ MPS, BondDim },
};
use whooie::{ read_npz, write_npz };
use lib::mipt::{ MiptManifest, save_mps_bonds };

type MeasRecord = Vec<Vec<Meas>>; // :: { layer, subindex }
type ProbRecord = Vec<Vec<(usize, f64)>>; // :: { layer, subindex }

#[derive(Copy, Clone, Debug, PartialEq)]
struct SaveState<'a, P> {
    outdir: P,
    seed: u64,
    nqubits: usize,
    depth: usize,
    circ: usize,
    p: f64,
    run: usize,
    output_id: &'a str,
}

impl<'a, P> SaveState<'a, P>
where P: AsRef<Path>
{
    fn target_path(&self, chi: usize) -> PathBuf {
        let fname = format!("\
            haar_coev_state_ones\
            _seed={}\
            _nqubits={}\
            _depth={}\
            _p={:.6}\
            _circ={}\
            _chi={}\
            _run={}\
            _outid={}\
            .npz",
            self.seed,
            self.nqubits,
            self.depth,
            self.p,
            self.circ,
            chi,
            self.run,
            self.output_id,
        );
        self.outdir.as_ref().join(fname)
    }
}

fn construct_meas_record<A, B>(
    depth: usize,
    meas_locs: &nd::ArrayBase<A, nd::Ix2>,
    traj_data: &nd::ArrayBase<B, nd::Ix1>,
) -> MeasRecord // :: { layer, subindex }
where
    A: nd::Data<Elem = u8>,
    B: nd::Data<Elem = u8>,
{
    assert_eq!(meas_locs.shape()[0], traj_data.shape()[0]);
    assert_eq!(meas_locs.shape()[1], 2);
    let mut record: MeasRecord = Vec::with_capacity(depth);
    let chunked =
        meas_locs.outer_iter()
        .zip(traj_data.iter())
        .chunk_by(|(loc, _outcome)| loc[0]);
    let mut last_layer: u8 = 0;
    for (layer, seq) in chunked.into_iter() {
        for _ in 0 .. layer.saturating_sub(last_layer).saturating_sub(1) {
            record.push(Vec::new());
        }
        let meas_layer: Vec<Meas> =
            seq.into_iter()
            .map(|(loc, outcome)| {
                Meas::Proj(loc[1] as usize, Outcome::from(*outcome))
            })
            .collect();
        last_layer = layer;
        record.push(meas_layer);
    }
    record
}

fn compute_probs<'a, P>(
    nqubits: usize,
    bonds: &[usize],
    circ: (&[UniSeq], &[MeasSeq]),
    save_states: Option<SaveState<'a, P>>
) -> Vec<ProbRecord> // :: { chi, layer, subindex }
where P: AsRef<Path>
{
    assert_eq!(circ.0.len(), circ.1.len());
    // `apply_bilayer` requires an rng, but this function should be entirely
    // deterministic because we're working off an existing set of measurement
    // results
    let mut rng = thread_rng();
    let mut probs: Vec<ProbRecord> = Vec::with_capacity(bonds.len());
    for chi in bonds.iter().copied() {
        let bond = BondDim::Const(chi);
        let mut state_c: MPS<Q, C64> =
            MPS::new_qnums((0 .. nqubits).map(|k| (Q(k), 1)), Some(bond))
            .unwrap();
        let probs_c: ProbRecord =
            circ.0.iter().zip(circ.1.iter())
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
        if let Some(save) = save_states.as_ref() {
            save_mps_bonds(&state_c, save.target_path(chi))
                .expect("error saving state");
        }
    }
    probs
}

fn main() {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(RUNS.min(32))
    //     .build_global()
    //     .unwrap();

    let outdir = PathBuf::from("/scratch/whuie2/haar_coev_probs");
    let output_id = format!("{:016x}", thread_rng().gen::<u64>());

    // parse cli args to open an existing output data file
    let mut args = std::env::args().skip(1);
    let traj_file: String =
        args.next()
        .expect("missing trajectory file");
    let traj_file = PathBuf::from(traj_file);
    let save = args.next().is_some_and(|arg| !arg.is_empty());

    let mut data = read_npz!(traj_file);
    let chi: nd::Array1<i32> = data.by_name("chi").unwrap();
    let circ: nd::Array1<i32> = data.by_name("circ").unwrap();
    let depth: nd::Array1<i32> = data.by_name("depth").unwrap();
    let genid: nd::Array1<i32> = data.by_name("genid").unwrap();
    let manifest_file: nd::Array1<i32> = data.by_name("manifest_file").unwrap();
    let meas_locs: nd::Array2<u8> = data.by_name("meas_locs").unwrap();
    let nqubits: nd::Array1<i32> = data.by_name("nqubits").unwrap();
    let p: nd::Array1<f64> = data.by_name("p").unwrap();
    let seed: nd::Array1<u64> = data.by_name("seed").unwrap();
    let traj_data: nd::Array2<u8> = data.by_name("traj_data").unwrap();

    let manifest_path = PathBuf::from(
        manifest_file.iter().map(|b| *b as u8 as char).collect::<String>()
    );
    let manifest =
        MiptManifest::load(&manifest_path)
        .expect("failed to load circuit manifest file");
    let circuit_dir =
        manifest_path.parent()
        .map(PathBuf::from)
        .unwrap_or(PathBuf::from("."));
    let genid_str: String = genid.iter().map(|b| *b as u8 as char).collect();
    println!("running circuit {}, p={:.6} of batch ID {}",
        circ[0], p[0], genid_str);
    println!("output ID: {}", output_id);

    let runs = traj_data.shape()[0];
    let bonds: Vec<usize> =
        chi.iter()
        .filter_map(|x| (*x > 0).then_some(*x as usize))
        .collect();

    // output text formatting
    let w_run: usize = (runs as f64).log10().floor() as usize + 1;

    let unis_file = circuit_dir.join(manifest.unis_fname(circ[0] as usize));
    let unis: Vec<UniSeq> =
        load_cbor(&unis_file)
        .expect("failed to read unitaries");

    let num_meas = traj_data.shape()[1];
    // prob_data :: { run, chi, measurement index }
    let mut prob_data: nd::Array3<f64> =
        nd::Array::zeros((runs, bonds.len(), num_meas));

    let completed = AtomicUsize::new(0);
    nd::Zip::indexed(traj_data.outer_iter())
        .and(prob_data.outer_iter_mut())
        .par_for_each(|run, traj_rec_r, mut prob_rec_r| {
            let meas_record = construct_meas_record(
                depth[0] as usize,
                &meas_locs,
                &traj_rec_r,
            );
            let probs = compute_probs(
                nqubits[0] as usize,
                &bonds,
                (unis.as_ref(), meas_record.as_ref()),
                save.then_some(
                    SaveState {
                        outdir: &outdir,
                        seed: manifest.seed(),
                        nqubits: manifest.nqubits(),
                        depth: manifest.depth(),
                        circ: circ[0] as usize,
                        p: p[0],
                        run,
                        output_id: &output_id,
                    }
                ),
            );

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

            let prev_run = completed.fetch_add(1, Ordering::SeqCst);
            eprintln!("  {:w_run$} / {:w_run$} ", prev_run + 1, runs);
        });
    eprintln!();

    let fname =
        format!("\
            haar_coev_probs_ones\
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
            "manifest_file" => &manifest_file,
            "seed" => &seed,
            "nqubits" => &nqubits,
            "depth" => &depth,
            "genid" => &genid,
            "p" => &p,
            "circ" => &circ,
            "chi" => &chi,
            "meas_locs" => &meas_locs,
            "traj_data" => &traj_data,
            "prob_data" => &prob_data,
        }
    );
}


