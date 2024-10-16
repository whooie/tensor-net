use std::path::PathBuf;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, thread_rng };
use tensor_net::circuit::*;
use tensor_net::gate::*;
use tensor_net::mps::*;
use whooie::{ mkdir, write_npz };

/// Number of qubits
const NQUBITS: usize = 15;
/// Circuit depth
const DEPTH: usize = 3 * NQUBITS;
/// Measurement probabilities
const PMEAS: &[f64] = &[
    0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200
];
/// Bond dimensions
const BONDS: &[Option<usize>] = &[
    None, Some(16), Some(32), Some(64), /*Some(128),*/
];
/// Number of circuits to generate
const CIRCS: usize = 10;
/// Time spacing between target measurements
const DT: usize = 2;
/// Qubit indices of target measurements
const TARGET_X: (usize, usize) = (NQUBITS / 2, NQUBITS / 2);

#[derive(Clone, Debug)]
struct CircuitExtra {
    circuit: Circuit,
    extra: Vec<Vec<Unitary>>, // sets Δt
    target_x: (usize, usize), // sets Δx
}

fn gen_circuit_extra<R>(
    config: CircuitConfig,
    dt: usize,
    target_x: (usize, usize),
    rng: &mut R,
) -> CircuitExtra
where R: Rng + ?Sized
{
    let circuit = Circuit::gen(NQUBITS, config, true, rng)
        .expect("error generating fixed circuit");
    let offs = circuit.depth() % 2 == 1;
    let extra: Vec<Vec<Unitary>> =
        (0..dt)
        .map(|_| {
            TileQ2::new(offs, NQUBITS)
                .map(|k| {
                    let haar2 = haar(2, rng);
                    Unitary::ExactGate(ExactGate::Q2(k, haar2))
                })
                .collect::<Vec<Unitary>>()
        })
        .collect();
    CircuitExtra { circuit, extra, target_x }
}

#[derive(Copy, Clone, Debug)]
struct MeasDist {
    p_00: f64,
    p_01: f64,
    p_10: f64,
    p_11: f64,
}

fn apply_unis(state: &mut MPS<Q, C64>, unis: &[Unitary]) {
    for uni in unis.iter() {
        match uni {
            Unitary::Gate(gate) => {
                state.apply_gate(*gate);
            },
            Unitary::ExactGate(ExactGate::Q1(k, gate)) => {
                state.apply_unitary1(*k, gate)
                    .expect("invalid q1 unitary application");
            },
            Unitary::ExactGate(ExactGate::Q2(k, gate)) => {
                state.apply_unitary2(*k, gate)
                    .expect("invalid q2 unitary application");
            },
        }
    }
}

fn sample_dist(circ: &CircuitExtra, bond: BondDim<f64>)
    -> MeasDist
{
    let mut mps =
        MPSCircuit::new(circ.circuit.nqubits(), Some(bond), None);
    mps.run_fixed(&circ.circuit, None);
    let mut state = mps.into_state();
    let (x0, x1) = circ.target_x;
    if let Some(layer) = circ.extra.first() {
        apply_unis(&mut state, layer);
        let p_x0 = state.probs(x0).unwrap();
        assert!((p_x0[0] + p_x0[1] - 1.0).abs() < 1e-15);

        let mut state_0 = state.clone();
        state_0.measure_postsel(x0, 0);
        circ.extra[1..].iter()
            .for_each(|layer| { apply_unis(&mut state_0, layer); });
        let p_0_x1 = state_0.probs(x1).unwrap();
        assert!((p_0_x1[0] + p_0_x1[1] - 1.0).abs() < 1e-15);

        let mut state_1 = state;
        state_1.measure_postsel(x0, 1);
        circ.extra[1..].iter()
            .for_each(|layer| { apply_unis(&mut state_1, layer); });
        let p_1_x1 = state_1.probs(x1).unwrap();
        assert!((p_1_x1[0] + p_1_x1[1] - 1.0).abs() < 1e-15);

        MeasDist {
            p_00: p_x0[0] * p_0_x1[0],
            p_01: p_x0[0] * p_0_x1[1],
            p_10: p_x0[1] * p_1_x1[0],
            p_11: p_x0[1] * p_1_x1[1],
        }
    } else {
        let p_x0 = state.probs(x0).unwrap();
        assert!((p_x0[0] + p_x0[1] - 1.0).abs() < 1e-15);

        let mut state_0 = state.clone();
        state_0.measure_postsel(x0, 0);
        let p_0_x1 = state_0.probs(x1).unwrap();
        assert!((p_0_x1[0] + p_0_x1[1] - 1.0).abs() < 1e-15);

        let mut state_1 = state;
        state_1.measure_postsel(x0, 1);
        let p_1_x1 = state_1.probs(x1).unwrap();
        assert!((p_1_x1[0] + p_1_x1[1] - 1.0).abs() < 1e-15);

        MeasDist {
            p_00: p_x0[0] * p_0_x1[0],
            p_01: p_x0[0] * p_0_x1[1],
            p_10: p_x0[1] * p_1_x1[0],
            p_11: p_x0[1] * p_1_x1[1],
        }
    }
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    // this is kinda bad
    static mut CIRC_COUNTER: usize = 0;
    unsafe { eprint!("    {} / {} ", CIRC_COUNTER, CIRCS); }

    let mut data: nd::Array4<f64> =
        nd::Array4::zeros((CIRCS, PMEAS.len(), BONDS.len(), 4));
    nd::Zip::from(data.outer_iter_mut())
        .par_for_each(|mut data_c| {
            let mut rng = thread_rng();
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
            let mut circ = gen_circuit_extra(config, DT, TARGET_X, &mut rng);
            for (&p, mut data_cp) in PMEAS.iter().zip(data_c.outer_iter_mut()) {
                circ.circuit.upsample_measurements(p, &mut rng);
                for (&b, mut data_cpb) in BONDS.iter().zip(data_cp.outer_iter_mut()) {
                    let bond =
                        b.map(BondDim::Const).unwrap_or(BondDim::Cutoff(1e-9));
                    let dist = sample_dist(&circ, bond);
                    data_cpb[0] = dist.p_00;
                    data_cpb[1] = dist.p_01;
                    data_cpb[2] = dist.p_10;
                    data_cpb[3] = dist.p_11;
                }
            }
            unsafe {
                CIRC_COUNTER += 1;
                eprint!("\r    {} / {} ", CIRC_COUNTER, CIRCS);
            }
        });

    let outfile = format!("fixed_conserv_exact_n={}_d={}.npz", NQUBITS, DEPTH);
    write_npz!(
        outdir.join(outfile),
        arrays: {
            "size" => &nd::array![NQUBITS as i32],
            "depth" => &nd::array![DEPTH as i32],
            "p_meas" =>
                &PMEAS.iter().copied().collect::<nd::Array1<f64>>(),
            "chi" =>
                &BONDS.iter().copied().map(|b| b.unwrap_or(0) as i32)
                .collect::<nd::Array1<i32>>(),
            "dists" => &data,
        }
    );
}


