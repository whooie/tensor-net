use std::path::PathBuf;
use lib::mipt::TwoPointManifest;

fn main() {
    let seed: u64 = 10546;
    let nqubits: Vec<usize> = [6].into_iter().chain(14 ..= 20).collect();
    let depth = |nqubits: usize| -> usize { 10 * nqubits };
    let num_circs: usize = 100;
    let dt: usize = 2;
    let x0 = |nqubits: usize| -> usize { nqubits / 2 };
    let x1 = |nqubits: usize| -> usize { nqubits / 2 };
    let p_meas: Vec<f64> =
        vec![
            0.025, 0.050, 0.075, 0.100, 0.115, 0.130,
            0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180,
            0.190, 0.210, 0.225, 0.250, 0.275,
            0.300, 0.325, 0.350, 0.375,
            0.400,
        ];
    let outdir = PathBuf::from("output").join("haar_two_point_circuits");

    eprint!("  0 / {}  ", nqubits.len());
    for (k, &n) in nqubits.iter().enumerate() {
        eprint!("\r  {} / {}  (n = {:2}) ", k, nqubits.len(), n);
        let d = depth(n);
        let xx = (x0(n), x1(n));
        let manifest =
            TwoPointManifest::new(
                seed, n, d, num_circs, p_meas.iter().copied(), dt, xx);
        manifest.gen_save(&outdir)
            .expect("failed to generate and save circuit");
        eprint!("\r  {} / {}  (n = {:2}) ", k + 1, nqubits.len(), n);
    }
    eprintln!();
}

