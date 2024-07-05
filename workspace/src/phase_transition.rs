use std::path::PathBuf;
use ndarray as nd;
use tensor_net::circuit::*;
use whooie::{ loop_call, mkdir, write_npz };

#[derive(Copy, Clone, Debug)]
struct Entropy {
    mean: f64,
    std_p: f64,
    std_m: f64,
}

fn eval_entropy(n: usize, p: f64, depth: usize, avg: usize) -> Entropy {
    let mut entropy: nd::Array1<f64> = nd::Array::zeros(avg);
    nd::Zip::from(entropy.view_mut())
        .par_for_each(move |s| {
            let mut circuit = MPSCircuit::new(n, Some(1e-9), None);
            let config = CircuitConfig {
                depth: DepthConfig::Const(depth),
                gates: GateConfig::Simple,
                // gates: GateConfig::GateSet(G1Set::U, G2Set::CX),
                measurement: MeasureConfig {
                    layer: MeasLayerConfig::Every,
                    prob: MeasProbConfig::Random(p),
                },
                entropy: EntropyConfig::VonNeumann(n / 2..n),
            };
            *s
                = circuit.run_entropy(config, None)
                .into_iter()
                .skip(3 * depth / 4)
                .sum::<f64>() / ((depth / 4) as f64);
        });
    let mean = entropy.mean().unwrap();
    let mut n: f64;
    n = 0.0;
    let std_p
        = entropy.iter().copied()
        .filter(|sk| *sk > mean)
        .map(|sk| { n += 1.0; (sk - mean).powi(2) })
        .sum::<f64>()
        .sqrt() / n.sqrt();
    n = 0.0;
    let std_m
        = entropy.iter().copied()
        .filter(|sk| *sk < mean)
        .map(|sk| { n += 1.0; (sk - mean).powi(2) })
        .sum::<f64>()
        .sqrt() / n.sqrt();
    Entropy { mean, std_p, std_m }
}

fn main() {
    const AVG: usize = 500;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let p_meas: nd::Array1<f64> = nd::Array1::linspace(0.12, 0.30, 19);
    let size: nd::Array1<u32> = (6..=20).step_by(2).collect();
    let caller = |q: &[usize]| -> (f64, f64, f64) {
        let n = size[q[1]] as usize;
        let p = p_meas[q[0]];
        let Entropy { mean, std_p, std_m } = eval_entropy(n, p, 3 * n, AVG);
        (mean, std_p, std_m)
    };
    let (s_mean, s_std_p, s_std_m)
        : (nd::ArrayD<f64>, nd::ArrayD<f64>, nd::ArrayD<f64>)
        = loop_call!(
            caller => (s_mean: f64, s_std_p: f64, s_std_m: f64),
            vars: { p_meas, size }
        );

    write_npz!(
        outdir.join("phase_transition.npz"),
        arrays: {
            "p_meas" => &p_meas,
            "size" => &size,
            "s_mean" => &s_mean,
            "s_std_p" => &s_std_p,
            "s_std_m" => &s_std_m,
        }
    );
}
