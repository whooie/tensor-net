use std::path::PathBuf;
use rand::{ SeedableRng, rngs::StdRng };
use tensor_net::circuit::*;

const N: usize = 6;
const T: usize = 10 * N;
const CIRCS: usize = 30;
const P_MEAS: &[f64] = &[
    0.010, 0.020, 0.030, 0.040, 0.050, 0.070, 0.090, 0.100, 0.115,
    0.130, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180,
    0.190, 0.210, 0.225, 0.250, 0.275,
    0.300, 0.325, 0.350, 0.375,
    0.400,
];
const SEED: u64 = 10546;

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let outdir = PathBuf::from("output").join("haar_circuits_na");
    for i in 0 .. CIRCS {
        let unis: Elements<UniSeq> =
            (0..T)
            .map(|t| brickwork_haar(N, t % 2 == 1, &mut rng))
            .collect();
        let meas_p: Elements<Elements<MeasSeq>> =
            P_MEAS.iter().copied()
            .map(|p| (0..T).map(|_| uniform_meas(N, p, &mut rng)).collect::<Elements<_>>())
            .collect();

        let unis_out = outdir.join(
            format!("unis_seed={}_n={}_depth={}_circ={}", SEED, N, T, i));
        unis.save(&unis_out).unwrap();
        let meas_out = outdir.join(
            format!("meas_seed={}_n={}_depth={}_plen={}_circ={}",
                SEED, N, T, P_MEAS.len(), i)
        );
        meas_p.save(&meas_out).unwrap();
    }
}
