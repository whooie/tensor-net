#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, SeedableRng, rngs::StdRng, thread_rng };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_net::{ circuit::*, gate::*, mps::* };
use whooie::write_npz;

const N: usize = 12;
const DT: usize = 1;
const TARGET_X: (usize, usize) = (N / 2, N / 2);
const AVG: usize = 5000;
const PMEAS: f64 = 1.00;

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
        .for_each(|(j, uni)| { state.apply_unitary2(*j, uni).unwrap(); });
    layer.meas.iter()
        .filter(|j| target.is_none_or(|k| k == **j))
        .for_each(|j| { state.measure(*j, rng); });
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
        let frozen = state; // ensure immutability
        // apply_layer(&mut state, layer, rng);
        // let mut p_x0 = state.probs(x0).unwrap();
        if (p_x0[0] + p_x0[1] - 1.0).abs() >= EPSILON {
            panic!("bad probabilities");
        }
        let n = p_x0[0] + p_x0[1];
        p_x0[0] /= n;
        p_x0[1] /= n;

        let (p00, p01) =
            if p_x0[0] >= EPSILON {
                let mut state0 = frozen.clone();
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
            if p_x0[1] >= EPSILON {
                let mut state1 = frozen.clone();
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

#[allow(static_mut_refs)]
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
    static mut COUNTER: usize = 0;

    if let Some(label) = progress {
        println!("{}", label);
        println!("\r  0 / {} ", avg);
        print!("\x1b[1G\x1b[1A");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }

    unsafe { COUNTER = 0; }
    let mut runs: Vec<Data> =
        (0..avg).into_par_iter()
            .map(|k| {
                let mut rng = thread_rng();
                let mut state: MPS<Q, C64> =
                    MPS::new_qubits(N, bond).unwrap();
                init_layers.iter()
                    .for_each(|layer| {
                        apply_layer(&mut state, layer, None, &mut rng);
                    });
                let run = sample(&state, targ_layers, targets, &mut rng);
                if progress.is_some() {
                    unsafe {
                        COUNTER += 1;
                        print!("\r  {} / {} ", COUNTER, avg);
                    }
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                run
            })
            .collect();
    if progress.is_some() { println!(); }
    // println!("{:?}", runs);
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

    // let mut totals: Data =
    //     (0..avg)
    //         .fold(
    //             Data { p00: 0.0, p01: 0.0, p10: 0.0, p11: 0.0 },
    //             |mut acc, k| {
    //                 let mut state: MPS<Q, C64> =
    //                     MPS::new_qubits(N, bond).unwrap();
    //                 init_layers.iter()
    //                     .for_each(|layer| {
    //                         apply_layer(&mut state, layer, None, rng);
    //                     });
    //                 let Data { p00, p01, p10, p11 } =
    //                     sample(&state, targ_layers, targets, rng);
    //                 acc.p00 += p00;
    //                 acc.p01 += p01;
    //                 acc.p10 += p10;
    //                 acc.p11 += p11;
    //                 if progress.is_some() {
    //                     print!("\r  {} / {} ", k + 1, avg);
    //                     std::io::Write::flush(&mut std::io::stdout()).unwrap();
    //                 }
    //                 acc
    //             },
    //         );
    // if progress.is_some() { println!(); }
    // totals.p00 /= avg as f64;
    // totals.p01 /= avg as f64;
    // totals.p10 /= avg as f64;
    // totals.p11 /= avg as f64;
    // totals

}

fn main() {
    let mut rng = thread_rng();
    // let mut rng = StdRng::seed_from_u64(10546);

    let cursor_reset = || { print!("\x1b[1G\x1b[2A"); };

    let init: Vec<Layer> =
        (0..2 * N).map(|t| Layer::gen(t % 2 == 1, PMEAS, &mut rng)).collect();
    let targ: Vec<Layer> =
        (0..DT + 1).map(|t| Layer::gen(t % 2 == 1, PMEAS, &mut rng)).collect();
    for (k, uni) in targ[0].unis.iter() {
        println!("{}\n{:+.3}", k, uni);
    }

    let q = compute_avg(
        Some("quantum"),
        None, &init, &targ, TARGET_X, AVG, &mut rng);
    // cursor_reset();

    let c1 = compute_avg(
        Some("χ = 1  "),
        Some(BondDim::Const(1)), &init, &targ, TARGET_X, AVG, &mut rng);
    // cursor_reset();

    let c2 = compute_avg(
        Some("χ = 2  "),
        Some(BondDim::Const(2)), &init, &targ, TARGET_X, AVG, &mut rng);
    // cursor_reset();

    let c4 = compute_avg(
        Some("χ = 4  "),
        Some(BondDim::Const(4)), &init, &targ, TARGET_X, AVG, &mut rng);
    // cursor_reset();

    let c8 = compute_avg(
        Some("χ = 8  "),
        Some(BondDim::Const(8)), &init, &targ, TARGET_X, AVG, &mut rng);
    // cursor_reset();

    let c16 = compute_avg(
        Some("χ = 16 "),
        Some(BondDim::Const(16)), &init, &targ, TARGET_X, AVG, &mut rng);
    // cursor_reset();

    println!("  q: {:.3?}", q);
    println!("     H = {:.3e}", q.entropy());
    println!(" c1: {:.3?}", c1);
    println!("     H = {:.3e}; KL = {:.3e}", c1.entropy(), c1.kl(&q));
    println!(" c2: {:.3?}", c2);
    println!("     H = {:.3e}; KL = {:.3e}", c2.entropy(), c2.kl(&q));
    println!(" c4: {:.3?}", c4);   
    println!("     H = {:.3e}; KL = {:.3e}", c4.entropy(), c4.kl(&q));
    println!(" c8: {:.3?}", c8);  
    println!("     H = {:.3e}; KL = {:.3e}", c8.entropy(), c8.kl(&q));
    println!("c16: {:.3?}", c16);  
    println!("     H = {:.3e}; KL = {:.3e}", c16.entropy(), c16.kl(&q));
}

