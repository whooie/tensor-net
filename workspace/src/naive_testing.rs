#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, SeedableRng, rngs::StdRng, thread_rng };
use tensor_net::{ circuit::*, gate::*, mps::* };
use whooie::write_npz;

const N: usize = 12;
const DT: usize = 2;
const TARGET_X: (usize, usize) = (N / 2, N / 2);
const CIRCS: usize = 50;
const AVG: usize = 100;

const EPSILON: f64 = 1e-12;

#[derive(Clone, Debug)]
struct Layer {
    unis: Vec<(usize, nd::Array2<C64>)>,
    meas: Vec<usize>,
}

impl Layer {
    fn gen<R>(offs: bool, p_meas: f64, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let unis: Vec<(usize, nd::Array2<C64>)> =
            TileQ2::new(offs, N)
            .map(|k| (k, haar(2, rng)))
            .collect();
        let meas: Vec<usize> =
            (0..N).filter(|_| rng.gen::<f64>() < p_meas).collect();
        Self { unis, meas }
    }

    fn gen_idswap<R>(offs: bool, p_meas: f64, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let unis: Vec<(usize, nd::Array2<C64>)> =
            TileQ2::new(offs, N)
            .map(|k| {
                if rng.gen::<bool>() {
                    (k, make_swap())
                } else {
                    (k, nd::Array2::eye(4))
                }
            })
            .collect();
        let meas: Vec<usize> =
            (0..N).filter(|_| rng.gen::<f64>() < p_meas).collect();
        Self { unis, meas }
    }
}

fn apply_layer<R>(
    state: &mut MPS<Q, C64>,
    layer: &Layer,
    target: Option<usize>,
    rng: &mut R,
) -> Option<Vec<f64>>
where R: Rng + ?Sized
{
    layer.unis.iter()
        .for_each(|(k, uni)| { state.apply_unitary2(*k, uni).unwrap(); });
    layer.meas.iter()
        .filter(|k| !target.is_some_and(|j| j == **k))
        .for_each(|k| { state.measure(*k, rng); });
    target.and_then(|k| state.probs(k))
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
    targ_layers: &[Layer],
    targets: (usize, usize),
    rng: &mut R,
) -> Data
where R: Rng + ?Sized
{
    let (x0, x1) = targets;
    let nlayers = targ_layers.len();
    if let Some(layer) = targ_layers.first() {
        let mut state = state_init.clone();
        let mut p_x0 = apply_layer(&mut state, layer, Some(x0), rng).unwrap();
        if (p_x0[0] + p_x0[1] - 1.0).abs() >= EPSILON {
            panic!("bad probabilities");
        }
        let n = p_x0[0] + p_x0[1];
        p_x0[0] /= n;
        p_x0[1] /= n;

        let (p00, p01) =
            if p_x0[0] > 0.0 {
                let mut state0 = state.clone();
                state0.measure_postsel(x0, 0);
                targ_layers[1..nlayers - 1].iter()
                    .for_each(|layer| {
                        apply_layer(&mut state0, layer, None, rng);
                    });
                let mut p_0_x1 =
                    apply_layer(&mut state0, &targ_layers[nlayers - 1], Some(x1), rng)
                    .unwrap();
                if (p_0_x1[0] + p_0_x1[1] - 1.0).abs() >= EPSILON {
                    panic!("bad probabilities");
                }
                let n = p_0_x1[0] + p_0_x1[1];
                p_0_x1[0] /= n;
                p_0_x1[1] /= n;
                (p_x0[0] * p_0_x1[0], p_x0[0] * p_0_x1[1])
            } else {
                (0.0, 0.0)
            };

        let (p10, p11) =
            if p_x0[1] > 0.0 {
                let mut state1 = state.clone();
                state1.measure_postsel(x0, 1);
                targ_layers[1..nlayers - 1].iter()
                    .for_each(|layer| {
                        apply_layer(&mut state1, layer, None, rng);
                    });
                let mut p_1_x1 =
                    apply_layer(&mut state1, &targ_layers[nlayers - 1], Some(x1), rng)
                    .unwrap();
                if (p_1_x1[0] + p_1_x1[1] - 1.0).abs() >= EPSILON {
                    panic!("bad probabilities");
                }
                let n = p_1_x1[0] + p_1_x1[1];
                p_1_x1[0] /= n;
                p_1_x1[1] /= n;
                (p_x0[1] * p_1_x1[0], p_x0[1] * p_1_x1[1])
            } else {
                (0.0, 0.0)
            };

        Data { p00, p01, p10, p11 }
    } else { unreachable!() }
}

fn compute_avg<R>(
    progress: Option<&str>,
    bond: Option<BondDim<f64>>,
    init_layers: &[Layer],
    targ_layers: &[Layer],
    targets: (usize, usize),
    avg: usize,
    rng: &mut R,
) -> Data
where R: Rng + ?Sized
{
    if let Some(label) = progress {
        println!("{}", label);
        println!("\r  0 / {} ", avg);
        print!("\x1b[1G\x1b[1A");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
    let mut totals: Data =
        (0..avg)
            .fold(
                Data { p00: 0.0, p01: 0.0, p10: 0.0, p11: 0.0 },
                |mut acc, k| {
                    let mut state: MPS<Q, C64> =
                        MPS::new_qubits(N, bond).unwrap();

                    // let mut state: MPS<Q, C64> =
                    //     MPS::new_qnums(
                    //         (0..N).map(|k| (Q::from(k), k % 2)),
                    //         bond
                    //     ).unwrap();

                    init_layers.iter()
                        .for_each(|layer| {
                            apply_layer(&mut state, layer, None, rng);
                        });
                    let Data { p00, p01, p10, p11 } =
                        sample(&state, targ_layers, targets, rng);
                    acc.p00 += p00;
                    acc.p01 += p01;
                    acc.p10 += p10;
                    acc.p11 += p11;
                    if progress.is_some() {
                        print!("\r  {} / {} ", k + 1, avg);
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    }
                    acc
                },
            );
    if progress.is_some() { println!(); }
    totals.p00 /= avg as f64;
    totals.p01 /= avg as f64;
    totals.p10 /= avg as f64;
    totals.p11 /= avg as f64;
    totals
}

fn do_p(p: f64, bonds: &[Option<usize>]) -> Vec<Data> {
    let mut rng = thread_rng();
    // let mut rng = StdRng::seed_from_u64(10546);

    let init: Vec<Layer> =
        (0..2 * N).map(|t| Layer::gen(t % 2 == 1, p, &mut rng)).collect();
    let targ: Vec<Layer> =
        (0..DT + 1).map(|t| Layer::gen(t % 2 == 1, p, &mut rng)).collect();

    bonds.iter().copied()
        .map(|mb_chi| {
            let mb_bond = mb_chi.map(BondDim::Const);
            compute_avg(None, mb_bond, &init, &targ, TARGET_X, AVG, &mut rng)
        })
        .collect()
}

fn main() {
    const P_MEAS: &[f64] = &[
        0.010, 0.020, 0.030, 0.040, 0.050,
        0.075, 0.100, 0.125, 0.150, 0.175,
        0.200,
    ];
    const BONDS: &[Option<usize>] = &[
        Some(2), Some(4), Some(8), Some(16), None,
    ];

    static mut COUNTER: usize = 0;
    let counter_total: usize = CIRCS * P_MEAS.len();
    eprint!("\r  0 / {} ", counter_total);

    let mut data: nd::Array4<f64> =
        nd::Array4::zeros((CIRCS, P_MEAS.len(), BONDS.len(), 4));
    nd::Zip::from(data.outer_iter_mut())
        .par_for_each(|mut data_c| {
            for (p, mut data_cp) in P_MEAS.iter().zip(data_c.outer_iter_mut()) {
                let dist_p: Vec<Data> = do_p(*p, BONDS);
                for (dist_pb, mut data_cpb) in dist_p.into_iter().zip(data_cp.outer_iter_mut()) {
                    data_cpb[0] = dist_pb.p00;
                    data_cpb[1] = dist_pb.p01;
                    data_cpb[2] = dist_pb.p10;
                    data_cpb[3] = dist_pb.p11;
                }
                unsafe {
                    COUNTER += 1;
                    eprint!("\r  {} / {} ", COUNTER, counter_total);
                }
            }
        });
    eprintln!();

    let outdir = PathBuf::from("output");
    let fname = format!("fixed_conserv_exact_n={}_d={}_mc={}.npz", N, 2 * N, CIRCS);
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

