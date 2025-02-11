#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, SeedableRng, rngs::StdRng, thread_rng };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_net::{ circuit::*, gate::*, mps::* };
use whooie::write_npz;

const N: usize = 14;
const T: usize = 2 * N;
const DT: usize = 2;
const TARGET_X: (usize, usize) = (N / 2, N / 2);
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

#[derive(Clone, Debug)]
struct UniLayer(Vec<(usize, nd::Array2<C64>)>);

impl UniLayer {
    fn gen<R>(offs: bool, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let data: Vec<(usize, nd::Array2<C64>)> =
            TileQ2::new(offs, N)
            .map(|k| (k, haar(2, rng)))
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

fn apply_layer<R>(
    state: &mut MPS<Q, C64>,
    unis: &UniLayer,
    meas: &MeasLayer,
    target: Option<usize>,
    rng: &mut R,
) -> Option<(f64, f64)>
where R: Rng + ?Sized
{
    unis.0.iter()
        .for_each(|(k, uni)| { state.apply_unitary2(*k, uni).unwrap(); });
    meas.0.iter()
        .filter(|k| !target.is_some_and(|j| j == **k))
        .for_each(|k| { state.measure(*k, rng); });
    target.and_then(|k| state.prob(k, 0).zip(state.prob(k, 1)))
}

#[derive(Copy, Clone, Debug)]
struct Data {
    p00: f64,
    p01: f64,
    p10: f64,
    p11: f64,
}

fn if_gtz<F>(x: f64, map: F) -> Option<f64>
where F: FnOnce(f64) -> f64
{
    (x > 0.0).then(|| map(x))
}

fn if_gtz2<F>(x: f64, y: f64, map: F) -> Option<f64>
where F: FnOnce(f64, f64) -> f64
{
    (x > 0.0 && y > 0.0).then(|| map(x, y))
}

impl Data {
    fn entropy(&self) -> f64 {
        let h = |px: f64| -px * px.ln();
        if_gtz(self.p00, h).unwrap_or(0.0)
            + if_gtz(self.p01, h).unwrap_or(0.0)
            + if_gtz(self.p10, h).unwrap_or(0.0)
            + if_gtz(self.p11, h).unwrap_or(0.0)
    }

    // KL(self||other)
    fn kl(&self, other: &Self) -> f64 {
        let kl = |px: f64, qx: f64| px * (px.ln() - qx.ln());
        if_gtz2(self.p00, other.p00, kl).unwrap_or(0.0)
            + if_gtz2(self.p01, other.p01, kl).unwrap_or(0.0)
            + if_gtz2(self.p10, other.p10, kl).unwrap_or(0.0)
            + if_gtz2(self.p11, other.p11, kl).unwrap_or(0.0)
    }
}

fn sample<R>(
    state_init: &MPS<Q, C64>,
    uni_layers: &[UniLayer],
    meas_layers: &[MeasLayer],
    targets: (usize, usize),
    rng: &mut R,
) -> Data
where R: Rng + ?Sized
{
    let (x0, x1) = targets;
    assert_eq!(uni_layers.len(), meas_layers.len());
    assert!(!uni_layers.is_empty());
    let nlayers = uni_layers.len();
    let (fst_uni, fst_meas) =
        uni_layers.first().zip(meas_layers.first()).unwrap();
    let mut state = state_init.clone();
    let (mut p0, mut p1) =
        apply_layer(&mut state, fst_uni, fst_meas, Some(x0), rng).unwrap();
    let frozen = state;
    let n = p0 + p1;
    p0 /= n;
    p1 /= n;

    let (p00, p01) =
        if p0 > 0.0 {
            let mut state0 = frozen.clone();
            state0.measure_postsel(x0, 0);
            uni_layers.iter().zip(meas_layers)
                .skip(1).take(nlayers - 2)
                .for_each(|(unis, meas)| {
                    apply_layer(&mut state0, unis, meas, None, rng);
                });
            let (last_uni, last_meas) =
                uni_layers.last().zip(meas_layers.last()).unwrap();
            let (mut p0_0, mut p0_1) =
                apply_layer(&mut state0, last_uni, last_meas, Some(x1), rng)
                .unwrap();
            let n = p0_0 + p0_1;
            p0_0 /= n;
            p0_1 /= n;
            (p0 * p0_0, p0 * p0_1)
        } else {
            (0.0, 0.0)
        };

    let (p10, p11) =
        if p1 > 0.0 {
            let mut state1 = frozen.clone();
            state1.measure_postsel(x0, 1);
            uni_layers.iter().zip(meas_layers)
                .skip(1).take(nlayers - 2)
                .for_each(|(unis, meas)| {
                    apply_layer(&mut state1, unis, meas, None, rng);
                });
            let (last_uni, last_meas) =
                uni_layers.last().zip(meas_layers.last()).unwrap();
            let (mut p1_0, mut p1_1) =
                apply_layer(&mut state1, last_uni, last_meas, Some(x1), rng)
                .unwrap();
            let n = p1_0 + p1_1;
            p1_0 /= n;
            p1_1 /= n;
            (p1 * p1_0, p1 * p1_1)
        } else {
            (0.0, 0.0)
        };

    Data { p00, p01, p10, p11 }
}

fn compute_bond_avg(
    bond: Option<BondDim<f64>>,
    init: (&[UniLayer], &[MeasLayer]),
    targ: (&[UniLayer], &[MeasLayer]),
    targets: (usize, usize),
    avg: usize,
) -> Data
{
    static mut COUNTER: usize = 0;
    let (init_unis, init_meas) = init;
    assert_eq!(init_unis.len(), init_meas.len());
    let (targ_unis, targ_meas) = targ;
    unsafe {
        COUNTER = 0;
        eprint!(" {:4} / {:4} ", 0, avg);
    }
    let runs: Vec<Data> =
        (0..avg).into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut state: MPS<Q, C64> =
                MPS::new_qubits(N, bond).unwrap();
            init_unis.iter().zip(init_meas)
                .for_each(|(unis, meas)| {
                    apply_layer(&mut state, unis, meas, None, &mut rng);
                });
            let res = sample(&state, targ_unis, targ_meas, targets, &mut rng);
            unsafe {
                COUNTER += 1;
                eprint!("\x1b[12D{:4} / {:4} ", COUNTER, avg);
            }
            res
        })
        .collect();
    let mut totals: Data =
        runs.into_iter()
        .fold(
            Data { p00: 0.0, p01: 0.0, p10: 0.0, p11: 0.0 },
            |mut acc, run| {
                acc.p00 += run.p00;
                acc.p01 += run.p01;
                acc.p10 += run.p10;
                acc.p11 += run.p11;
                acc
            },
        );
    totals.p00 /= avg as f64;
    totals.p01 /= avg as f64;
    totals.p10 /= avg as f64;
    totals.p11 /= avg as f64;
    totals
}

fn main() {
    static mut COUNTER: usize = 0;
    let counter_total: usize = CIRCS * P_MEAS.len() * BONDS.len();
    eprint!("\r  0 / {} ", counter_total);

    let mut data: nd::Array4<f64> =
        nd::Array4::zeros((CIRCS, P_MEAS.len(), BONDS.len(), 4));

    let mut rng = thread_rng();

    let iter =
        data.outer_iter_mut();
    for mut data_c in iter {

        let unis_init: Vec<UniLayer> =
            (0..T)
            .map(|t| UniLayer::gen(t % 2 == 1, &mut rng))
            .collect();
        let unis_targ: Vec<UniLayer> =
            (0..DT)
            .map(|t| UniLayer::gen((T + t) % 2 == 1, &mut rng))
            .collect();
        let meas_init_p: Vec<Vec<MeasLayer>> =
            P_MEAS.iter().copied()
            .map(|p| (0..T).map(|_| MeasLayer::gen(p, &mut rng)).collect())
            .collect();
        let meas_targ_p: Vec<Vec<MeasLayer>> =
            P_MEAS.iter().copied()
            .map(|p| (0..DT).map(|_| MeasLayer::gen(p, &mut rng)).collect())
            .collect();

        let iter =
            meas_init_p.iter()
            .zip(&meas_targ_p)
            .zip(data_c.outer_iter_mut());
        for ((meas_init, meas_targ), mut data_cp) in iter {

            let iter =
                BONDS.iter().copied()
                .zip(data_cp.outer_iter_mut());
            for (mb_chi, mut data_cpb) in iter {

                let mb_bond = mb_chi.map(BondDim::Const);
                let avg =
                    compute_bond_avg(
                        mb_bond,
                        (&unis_init, meas_init),
                        (&unis_targ, meas_targ),
                        TARGET_X,
                        RUNS,
                    );

                data_cpb[0] = avg.p00;
                data_cpb[1] = avg.p01;
                data_cpb[2] = avg.p10;
                data_cpb[3] = avg.p11;

                unsafe {
                    COUNTER += 1;
                    eprint!("\r  {} / {} ", COUNTER, counter_total);
                }
            }
        }
    }
    eprintln!();

    let outdir = PathBuf::from("output");
    let fname = format!("haar_n={}_d={}_circs={}_runs={}.npz", N, 2 * N, CIRCS, RUNS);
    println!("{}", fname);
    write_npz!(
        outdir.join(fname),
        arrays: {
            "size" => &nd::array![N as i32],
            "depth" => &nd::array![2 * N as i32],
            "circs" => &nd::array![CIRCS as i32],
            "p_meas" => &P_MEAS.iter().copied().collect::<nd::Array1<f64>>(),
            "chi" => &BONDS.iter().copied().map(|b| b.unwrap_or(0) as i32).collect::<nd::Array1<i32>>(),
            "dt" => &nd::array![DT as i32],
            "target_x" => &nd::array![TARGET_X.0 as i32, TARGET_X.1 as i32],
            "dists" => &data,
        }
    );
}

