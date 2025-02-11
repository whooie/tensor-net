#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::{ ops::Range, path::PathBuf };
use ndarray::{ self as nd, linalg::kron };
use num_complex::Complex64 as C64;
use rand::{ Rng, SeedableRng, rngs::StdRng, thread_rng };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use clifford_sim::{
    gate::{ Clifford, Gate },
    stab::{ Stab, Postsel, Probs },
    stabm::StabM,
};
use tensor_net::{
    circuit::{ Q, TileQ2 },
    gate::haar,
    mps::{ BondDim, MPS },
};
use whooie::write_npz;

const N: usize = 10;
const T: usize = 2 * N;
const DT: usize = 2;
const TARGET_X: usize = N / 2;
const CIRCS: usize = 50;
const RUNS: usize = 5000;
const P_MEAS: &[f64] = &[
    0.010, 0.020, 0.030, 0.040, 0.050, 0.070, 0.090,
    0.100, 0.115, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190,
    0.210, 0.225, 0.250, 0.275,
    0.300, 0.325, 0.350, 0.375,
    0.400,
];
const BONDS: &[Option<usize>] = &[
    None, Some(4), Some(8), Some(16), Some(32),
];

const EPSILON: f64 = 1e-12;

fn target_range() -> Range<usize> {
    if (T + TARGET_X) % 2 == 0 {
        let start = TARGET_X.saturating_sub(2 * (DT / 2));
        let end = (TARGET_X + 2 * ((DT + 1) / 2)).min(N);
        start .. end
    } else {
        let start = TARGET_X.saturating_sub(2 * ((DT + 1) / 2)) + 1;
        let end = (TARGET_X + 2 * (DT / 2) + 1).min(N);
        start .. end
    }
}

fn gate_matrix(gate: &Gate) -> nd::Array2<C64> {
    use std::f64::consts::FRAC_1_SQRT_2;
    const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    const I: C64 = C64 { re: 0.0, im: 1.0 };
    const ORT2: C64 = C64 { re: FRAC_1_SQRT_2, im: 0.0 };
    const IORT2: C64 = C64 { re: 0.0, im: FRAC_1_SQRT_2 };
    let eye: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, ONE]];
    match gate {
        Gate::H(k) => {
            let mat: nd::Array2<C64> = nd::array![[ORT2, ORT2], [ORT2, -ORT2]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!()
            }
        },
        Gate::X(k) => {
            let mat: nd::Array2<C64> = nd::array![[ZERO, ONE], [ONE, ZERO]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!()
            }
        },
        Gate::Y(k) => {
            let mat: nd::Array2<C64> = nd::array![[ZERO, -I], [I, ZERO]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!()
            }
        },
        Gate::Z(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, -ONE]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!()
            }
        },
        Gate::S(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, I]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!()
            }
        },
        Gate::SInv(k) => {
            let mat: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, -I]];
            if *k == 0 {
                kron(&mat, &eye)
            } else if *k == 1 {
                kron(&eye, &mat)
            } else {
                panic!()
            }
        },
        Gate::CX(a, _) => {
            if *a == 0 {
                nd::array![
                    [ONE, ZERO, ZERO, ZERO],
                    [ZERO, ONE, ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE],
                    [ZERO, ZERO, ONE, ZERO],
                ]
            } else if *a == 1 {
                nd::array![
                    [ONE, ZERO, ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE],
                    [ZERO, ZERO, ONE, ZERO],
                    [ZERO, ONE, ZERO, ZERO],
                ]
            } else {
                panic!()
            }
        },
        Gate::CZ(..) => {
            nd::array![
                [ONE,  ZERO, ZERO,  ZERO],
                [ZERO, ONE,  ZERO,  ZERO],
                [ZERO, ZERO, ONE,   ZERO],
                [ZERO, ZERO, ZERO, -ONE ],
            ]
        },
        Gate::Swap(..) => {
            nd::array![
                [ONE, ZERO, ZERO, ZERO],
                [ZERO, ZERO, ONE, ZERO],
                [ZERO, ONE, ZERO, ZERO],
                [ZERO, ZERO, ZERO, ONE],
            ]
        },
    }
}

#[derive(Clone, Debug)]
struct Uni {
    gates: Vec<Gate>,
    mat: (usize, nd::Array2<C64>),
}

#[derive(Clone, Debug)]
struct UniLayer(Vec<Uni>);

impl UniLayer {
    fn gen<R>(offs: bool, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let data: Vec<Uni> =
            TileQ2::new(offs, N)
            .map(|k| {
                let mut gates: Vec<Gate> =
                    Clifford::gen(2, rng).unpack().0;
                let mat: nd::Array2<C64> =
                    gates.iter()
                    .map(gate_matrix)
                    .fold(nd::Array2::eye(4), |acc, gate| gate.dot(&acc));
                gates.iter_mut().for_each(|g| { g.shift(k); });
                Uni { gates, mat: (k, mat) }
            })
            .collect();
        Self(data)
    }
}

#[derive(Clone, Debug)]
struct MeasLayer(Vec<usize>);

impl MeasLayer {
    fn gen<R>(p: f64, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let data: Vec<usize> =
            (0..N).filter(|_| rng.gen::<f64>() < p)
            .collect();
        Self(data)
    }
}

fn apply_layer_mps<R>(
    state: &mut MPS<Q, C64>,
    unis: &UniLayer,
    meas: &MeasLayer,
    target: Option<usize>,
    rng: &mut R,
) -> Option<(f64, f64)>
where R: Rng + ?Sized
{
    unis.0.iter()
        .for_each(|Uni { gates: _, mat }| {
            state.apply_unitary2(mat.0, &mat.1).unwrap();
        });
    meas.0.iter()
        .filter(|k| !target.is_some_and(|j| j == **k))
        .for_each(|k| { state.measure(*k, rng); });
    target.and_then(|k| state.prob(k, 0).zip(state.prob(k, 1)))
}

fn apply_layer_stab<R>(
    state: &mut Stab,
    unis: &UniLayer,
    meas: &MeasLayer,
    target: Option<usize>,
    rng: &mut R,
) -> Option<Probs>
where R: Rng + ?Sized
{
    unis.0.iter()
        .for_each(|Uni { gates, mat: _ }| { state.apply_circuit(gates); });
    meas.0.iter()
        .filter(|k| !target.is_some_and(|j| j == **k))
        .for_each(|k| { state.measure(*k, rng); });
    target.map(|k| state.probs(k))
}

type Dens = nd::Array2<C64>;

fn sample_density_mps(
    bond: BondDim<f64>,
    circ: (&[UniLayer], &[MeasLayer]),
    part: Range<usize>,
) -> Dens
{
    assert_eq!(circ.0.len(), circ.1.len());
    let mut rng = thread_rng();
    let mut state: MPS<Q, C64> = MPS::new_qubits(N, Some(bond)).unwrap();
    circ.0.iter().zip(circ.1.iter())
        .for_each(|(unis, meas)| {
            apply_layer_mps(&mut state, unis, meas, None, &mut rng);
        });
    state.into_matrix_part(part).1
}

fn sample_density_stab(
    circ: (&[UniLayer], &[MeasLayer]),
    part: Range<usize>,
) -> Dens
{
    assert_eq!(circ.0.len(), circ.1.len());
    let mut rng = thread_rng();
    let mut state = Stab::new(N);
    circ.0.iter().zip(circ.1.iter())
        .for_each(|(unis, meas)| {
            apply_layer_stab(&mut state, unis, meas, None, &mut rng);
        });
    let sidelen = 2_usize.pow((part.end - part.start) as u32);
    StabM::from(state)
        .partial_trace_multi((0..N).filter(|k| !part.contains(k)))
        .as_matrix()
        .into_iter()
        .copied()
        .collect::<nd::Array1<C64>>()
        .into_shape((sidelen, sidelen))
        .unwrap()
}

fn compute_bond_avg(
    bond: Option<usize>,
    circ: (&[UniLayer], &[MeasLayer]),
    part: Range<usize>,
    avg: usize,
) -> f64
{
    static mut COUNTER: usize = 0;
    assert_eq!(circ.0.len(), circ.1.len());
    let z: usize = 1 + f64::log10(avg as f64).floor() as usize;
    unsafe {
        COUNTER = 0;
        eprint!(" {:w$} / {:w$} ", 0, avg, w=z);
    }
    let runs: Vec<f64> =
        // (0..avg)
        (0..avg).into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let rho =
                if let Some(chi) = bond {
                    sample_density_mps(BondDim::Const(chi), circ, part.clone())
                } else {
                    sample_density_stab(circ, part.clone())
                };
            let res = rho.dot(&rho).diag().sum().norm();
            unsafe {
                COUNTER += 1;
                eprint!("\x1b[{}D{:w$} / {:w$} ", 2 * z + 4, COUNTER, avg, w=z);
            }
            res
        })
        .collect();
    runs.into_iter()
        .sum::<f64>()
        / avg as f64
}

fn main() {
    static mut COUNTER: usize = 0;
    let counter_total: usize = CIRCS * P_MEAS.len() * BONDS.len();
    eprint!("\r  0 / {} ", counter_total);

    let mut data: nd::Array3<f64> =
        nd::Array3::zeros((CIRCS, P_MEAS.len(), BONDS.len()));
    let part = target_range();

    let mut rng = thread_rng();
    // let mut rng = StdRng::seed_from_u64(10546);

    let iter =
        data.outer_iter_mut();
    for mut data_c in iter {

        let unis_init: Vec<UniLayer> =
            (0..T)
            .map(|t| UniLayer::gen(t % 2 == 1, &mut rng))
            .collect();
        let meas_init_p: Vec<Vec<MeasLayer>> =
            P_MEAS.iter().copied()
            .map(|p| (0..T).map(|_| MeasLayer::gen(p, &mut rng)).collect())
            .collect();

        let iter =
            meas_init_p.iter()
            .zip(data_c.outer_iter_mut());
        for (meas_init, mut data_cp) in iter {

            let iter =
                BONDS.iter().copied()
                .zip(data_cp.iter_mut());
            for (mb_chi, data_cpb) in iter {

                *data_cpb = compute_bond_avg(
                    mb_chi, (&unis_init, meas_init), part.clone(), RUNS);

                unsafe {
                    COUNTER += 1;
                    eprint!("\r  {} / {} ", COUNTER, counter_total);
                }
            }
        }
    }
    eprintln!();

    let outdir = PathBuf::from("output");
    let fname = format!("clifford_mixing_n={}_d={}_circs={}_runs={}.npz", N, 2 * N, CIRCS, RUNS);
    println!("{}", fname);
    write_npz!(
        outdir.join(fname),
        arrays: {
            "size" => &nd::array![N as i32],
            "depth" => &nd::array![T as i32],
            "circs" => &nd::array![CIRCS as i32],
            "p_meas" => &P_MEAS.iter().copied().collect::<nd::Array1<f64>>(),
            "chi" => &BONDS.iter().copied().map(|b| b.unwrap_or(0) as i32).collect::<nd::Array1<i32>>(),
            "dt" => &nd::array![DT as i32],
            "target_x" => &nd::array![TARGET_X as i32],
            "target_range" => &nd::array![part.start as i32, part.end as i32],
            "tr" => &data,
        }
    );
}

