use std::path::PathBuf;
use ndarray as nd;
use rand::thread_rng;
use tensor_net::circuit::*;
use tensor_net::mps::BondDim;
use whooie::{ mkdir, write_npz };

/// Number of qubits
const NQUBITS: usize = 24;
/// Circuit depth
const DEPTH: usize = 4 * NQUBITS;
/// Measurement probability
const PMEAS: f64 = 0.35;
/// Number of runs for each circuit
const RUNS: usize = 500;
/// Number of circuits to generate
const CIRCS: usize = 10;

/// Convert a bare measurement outcome to a Z-operator measurement:
/// ∣0⟩ -> z = +1
/// ∣1⟩ -> z = -1
fn z_op(meas: Option<Outcome>) -> i8 {
    match meas {
        Some(Outcome::Zero) =>  1,
        Some(Outcome::One)  => -1,
        None                =>  0,
    }
}

static mut SAMPLE_COUNTER: usize = 0;

/// Sample the classical measurement outcomes of a fixed circuit `avg` times,
/// returning data in a 3D array :: { avg, d, n } for circuit depth d and number
/// of qubits n.
fn sample_outcomes(circ: &Circuit, avg: usize) -> nd::Array3<i8> {
    let mut z: nd::Array3<i8>
        = nd::Array3::zeros((avg, circ.depth(), circ.nqubits()));
    unsafe {
        SAMPLE_COUNTER = 0;
        eprint!("    {} / {} ", SAMPLE_COUNTER, avg);
    }
    nd::Zip::from(z.outer_iter_mut())
        .par_for_each(|mut run| {
            let mut mps = MPSCircuit::new(
                circ.nqubits(), Some(BondDim::Cutoff(1e-9)), None);
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
    let paramstring
        = format!("n={}_d={}_p={:.3}_runs={}", NQUBITS, DEPTH, PMEAS, RUNS);
    for k in 0..CIRCS {
        eprintln!("  {} / {}: ", k + 1, CIRCS);
        let config = CircuitConfig {
            depth: DepthConfig::Const(DEPTH),
            gates: GateConfig::Haar2,
            measurement: MeasureConfig {
                layer: MeasLayerConfig::Every,
                prob: MeasProbConfig::Random(PMEAS),
                reset: false,
            },
            entropy: EntropyConfig::VonNeumann(NQUBITS / 2..NQUBITS),
        };
        let circ = Circuit::gen(NQUBITS, config, true, &mut rng)
            .expect("error generating fixed circuit");
        let out = sample_outcomes(&circ, RUNS);
        let outfile = format!("{}_{:03}.npz", paramstring, k);
        println!("output/{}", outfile);
        write_npz!(
            outdir.join(outfile),
            arrays: {
                "size" => &nd::array![NQUBITS as u32],
                "depth" => &nd::array![DEPTH as u32],
                "p_meas" => &nd::array![PMEAS],
                "runs" => &nd::array![RUNS as u32],
                "outcomes" => &out,
            }
        );
    }
    eprintln!();
}

