use std::path::PathBuf;
use ndarray as nd;
use rand::thread_rng;
use tensor_net::circuit::*;
use tensor_net::mps::BondDim;
use whooie::{ mkdir, write_npz };

/// Number of qubits
const NQUBITS: usize = 15;
/// Circuit depth
const DEPTH: usize = 3 * NQUBITS;
/// Measurement probabilities
// const PMEAS: &[f64] = &[0.10, 0.15, 0.20, 0.26, 0.27, 0.30, 0.35];
// const PMEAS: &[f64] = &[0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20];
const PMEAS: &[f64] = &[
    0.02, 0.03, 0.04, 0.05, 0.06,
    0.07, 0.08, 0.10, 0.12, 0.14,
    0.15, 0.16, 0.17, 0.18, 0.20,
    0.22, 0.25, 0.28, 0.31, 0.34,
    0.37, 0.40,
];
/// Bond dimensions
const BONDS: &[Option<usize>] = &[None, Some(16), Some(32), Some(64), /*Some(128)*/];
/// Number of runs for each circuit
const RUNS: usize = 1000;
/// Number of circuits to generate
const CIRCS: usize = 1;

/// Convert a bare measurement outcome to a Z-operator measurement:
/// ```text
/// ∣0⟩ -> z = +1
/// ∣1⟩ -> z = -1
/// ```
fn z_op(meas: Option<Outcome>) -> i8 {
    match meas {
        Some(Outcome::Zero) =>  1,
        Some(Outcome::One)  => -1,
        None                =>  0,
    }
}

/// Sample the classical measurement outcomes of a fixed circuit `avg` times,
/// returning data in a 3D array :: { avg, d, n } for circuit depth d and number
/// of qubits n.
#[allow(static_mut_refs)]
fn sample_outcomes(circ: &Circuit, bond: BondDim<f64>, avg: usize)
    -> nd::Array3<i8>
{
    // this is kinda bad
    static mut SAMPLE_COUNTER: usize = 0;

    let mut z: nd::Array3<i8>
        = nd::Array3::zeros((avg, circ.depth(), circ.nqubits()));
    unsafe {
        SAMPLE_COUNTER = 0;
        eprint!("    {} / {} ", SAMPLE_COUNTER, avg);
    }
    nd::Zip::from(z.outer_iter_mut())
        .par_for_each(|mut run| {
            let mut mps = MPSCircuit::new(
                circ.nqubits(), Some(bond), None);
            let mut meas_record = MeasRecord::new();
            mps.run_fixed(circ, Some(&mut meas_record));
            meas_record.into_iter()
                .flatten()
                .map(z_op)
                .zip(run.iter_mut())
                .for_each(|(z_jk, run_jk)| { *run_jk = z_jk; });
            unsafe {
                SAMPLE_COUNTER += 1;
                eprint!("\r    {} / {} ", SAMPLE_COUNTER, avg);
            }
        });
    eprintln!();
    z
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let mut rng = thread_rng();
    for k in 0..CIRCS {
        let config = CircuitConfig {
            depth: DepthConfig::Const(DEPTH),
            gates: GateConfig::Haar2,
            measurement: MeasureConfig {
                layer: MeasLayerConfig::Every,
                prob: MeasProbConfig::Random(*PMEAS.first().unwrap()),
                reset: false,
            },
            entropy: EntropyConfig::VonNeumann(NQUBITS / 2..NQUBITS),
        };
        let mut circ = Circuit::gen(NQUBITS, config, true, &mut rng)
            .expect("error generating fixed circuit");
        let Some(layer) = circ.get_layer_mut(DEPTH - 1) else { panic!(); };
        layer.meas.push(Measurement::Rand(NQUBITS / 2));
        layer.meas.sort_by_key(Measurement::idx);
        let Some(layer) = circ.get_layer_mut(DEPTH - 3) else { panic!(); };
        layer.meas.push(Measurement::Rand(NQUBITS / 2));
        layer.meas.sort_by_key(Measurement::idx);
        for (j, p) in PMEAS.iter().copied().enumerate() {
            circ.upsample_measurements(p, &mut rng);
            for (l, b) in BONDS.iter().copied().enumerate() {
                eprintln!("  {} / {}; {} / {}; {} / {}: ",
                    k + 1, CIRCS, j + 1, PMEAS.len(), l + 1, BONDS.len());
                let paramstring
                    = format!("n={}_d={}_p={:.3}_chi={:03}_runs={}",
                        NQUBITS, DEPTH, p, b.unwrap_or(0), RUNS);
                let bond
                    = b.map(BondDim::Const).unwrap_or(BondDim::Cutoff(1e-9));
                let out = sample_outcomes(&circ, bond, RUNS);
                let outfile = format!("{}_{:03}.npz", paramstring, k);
                println!("output/{}", outfile);
                write_npz!(
                    outdir.join(outfile),
                    arrays: {
                        "size" => &nd::array![NQUBITS as u32],
                        "depth" => &nd::array![DEPTH as u32],
                        "p_meas" => &nd::array![p],
                        "chi" => &nd::array![b.unwrap_or(0) as u32],
                        "runs" => &nd::array![RUNS as u32],
                        "outcomes" => &out,
                    }
                );
            }
        }
    }
    eprintln!();
}

