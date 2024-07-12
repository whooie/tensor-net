#[allow(unused_imports)]
use std::{ time, path::PathBuf };
use ndarray as nd;
#[allow(unused_imports)]
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_net::circuit::*;
use tensor_net::mps::BondDim;
use whooie::{ mkdir, print_flush, write_npz };

fn main() {
    const N: usize = 4;
    const MC: usize = 1000;
    const DEPTH: usize = 4 * N;
    const P_MEAS: f64 = 0.00;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    static mut COUNTER: usize = 0;
    unsafe { print_flush!(" {COUNTER} "); }
    let mut s_acc: nd::Array1<f64>
        = (0..MC).into_par_iter()
        .map(|_| {
            let mut circuit
                = MPSCircuit::new(N, Some(BondDim::Cutoff(f64::EPSILON)), None);
            let config = CircuitConfig {
                depth: DepthConfig::Const(DEPTH),
                // gates: GateConfig::Simple,
                gates: GateConfig::GateSet(G1Set::HS, G2Set::CZ),
                measurement: MeasureConfig {
                    layer: MeasLayerConfig::Every,
                    prob: MeasProbConfig::Random(P_MEAS),
                    // prob: MeasProbConfig::cycling_prob(P_MEAS),
                },
                entropy: EntropyConfig::VonNeumann(0..N / 2),
                // entropy: EntropyConfig::VonNeumann(N / 2..N),
            };
            let res = nd::Array1::from(circuit.run_entropy(config, None));
            unsafe { COUNTER += 1; print_flush!("\r {} ", COUNTER); }
            res
        })
        .reduce(
            || nd::Array1::<f64>::zeros(DEPTH + 1),
            |acc, term| acc + &term,
        );
    println!();

    // let mut s_acc: nd::Array1<f64> = nd::Array1::zeros(DEPTH + 1);
    // let mut dt: nd::Array1<f64> = nd::Array1::zeros(MC);
    // let mut s_final: nd::Array1<f64> = nd::Array1::zeros(MC);
    // let iter = s_final.iter_mut().zip(dt.iter_mut()).enumerate();
    // for (k, (s_final_k, dt_k)) in iter {
    //     print_flush!("\r {} ", k);
    //     let mut circuit
    //         = MPSCircuit::new(N, Some(BondDim::Cutoff(f64::EPSILON)), None);
    //     let config = CircuitConfig {
    //         depth: DepthConfig::Const(DEPTH),
    //         // gates: GateConfig::Simple,
    //         gates: GateConfig::GateSet(G1Set::H, G2Set::CZ),
    //         measurement: MeasureConfig {
    //             layer: MeasLayerConfig::Every,
    //             prob: MeasProbConfig::Random(P_MEAS),
    //         },
    //         entropy: EntropyConfig::VonNeumann(0..N / 2),
    //     };
    //     let t0 = time::Instant::now();
    //     let entropy = circuit.run_entropy(config, None);
    //     *dt_k = (time::Instant::now() - t0).as_secs_f64();
    //     *s_final_k = entropy.iter().skip(3 * DEPTH / 4).copied().sum::<f64>();
    //     s_acc += &nd::Array1::from(entropy);
    // }
    // println!();
    // let mut n: f64;
    // let dt_mean = dt.mean().unwrap();
    // n = 0.0;
    // let dt_std_p
    //     = dt.iter().copied()
    //     .filter(|dtk| *dtk > dt_mean)
    //     .map(|dtk| { n += 1.0; (dtk - dt_mean).powi(2) })
    //     .sum::<f64>()
    //     .sqrt() / n.sqrt();
    // n = 0.0;
    // let dt_std_m
    //     = dt.iter().copied()
    //     .filter(|dtk| *dtk < dt_mean)
    //     .map(|dtk| { n += 1.0; (dtk - dt_mean).powi(2) })
    //     .sum::<f64>()
    //     .sqrt() / n.sqrt();
    // let s_final_mean = s_final.mean().unwrap();
    // n = 0.0;
    // let s_final_std_p
    //     = s_final.iter().copied()
    //     .filter(|sk| *sk > s_final_mean)
    //     .map(|sk| { n += 1.0; (sk - s_final_mean).powi(2) })
    //     .sum::<f64>()
    //     .sqrt() / n.sqrt();
    // n = 0.0;
    // let s_final_std_m
    //     = s_final.iter().copied()
    //     .filter(|sk| *sk < s_final_mean)
    //     .map(|sk| { n += 1.0; (sk - s_final_mean).powi(2) })
    //     .sum::<f64>()
    //     .sqrt() / n.sqrt();
    // println!(" time per run: {:.3} + {:.3} - {:.3}",
    //     dt_mean, dt_std_p, dt_std_m);
    // println!("final entropy: {:.3} + {:.3} - {:.3}",
    //     s_final_mean, s_final_std_p, s_final_std_m);

    s_acc /= MC as f64;

    write_npz!(
        outdir.join("entropy.npz"),
        arrays: {
            "size" => &nd::array![N as u32],
            "mc" => &nd::array![MC as u32],
            "p_meas" => &nd::array![P_MEAS],
            "entropy" => &s_acc,
        }
    );
}

