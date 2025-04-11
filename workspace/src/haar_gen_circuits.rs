use std::path::PathBuf;
use rand::{ SeedableRng, rngs::StdRng };
use lib::{ *, haar::* };

const N: usize = 20;
const T: usize = 10 * N;
const CIRCS: usize = 30;
const P_MEAS: &[f64] = &[
    0.025, 0.050, 0.075, 0.100, 0.115, 0.130,
    0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180,
    0.190, 0.210, 0.225, 0.250, 0.275,
    0.300, 0.325, 0.350, 0.375,
    0.400,
];
const SEED: u64 = 10546;

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let outdir = PathBuf::from("output").join("haar_circuits");
    for i in 0 .. CIRCS {
        let unis: Vec<UniLayer> =
            (0..T)
            .map(|t| UniLayer::gen(N, t % 2 == 1, &mut rng))
            .collect();
        let meas_p: Vec<Vec<MeasLayer>> =
            P_MEAS.iter().copied()
            .map(|p| (0..T).map(|_| MeasLayer::gen(N, p, &mut rng)).collect())
            .collect();

        let unis_out = outdir.join(
            format!("unis_seed={}_n={}_depth={}_circ={}.npz", SEED, N, T, i));
        save_unis(&unis_out, &unis);
        let meas_out = outdir.join(
            format!("meas_seed={}_n={}_depth={}_plen={}_circ={}.npz",
                SEED, N, T, P_MEAS.len(), i)
        );
        save_meas(&meas_out, P_MEAS.iter().zip(&meas_p));
    }
}
