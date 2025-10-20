use std::path::PathBuf;
use lib::haar::MiptManifest;

fn main() {
    let seed: u64 = 10546;
    let nqubits: Vec<usize> = [6].into_iter().chain(14 ..= 20).collect();
    let depth = |nqubits: usize| -> usize { 10 * nqubits };
    let num_circs: usize = 30;
    let p_meas: Vec<f64> =
        vec![
            0.025, 0.050, 0.075, 0.100, 0.115, 0.130,
            0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180,
            0.190, 0.210, 0.225, 0.250, 0.275,
            0.300, 0.325, 0.350, 0.375,
            0.400,
        ];
    let outdir = PathBuf::from("output").join("haar_circuits");

    eprint!("  0 / {}  ", nqubits.len());
    for (k, &n) in nqubits.iter().enumerate() {
        eprint!("\r  {} / {}  (n = {:2}) ", k, nqubits.len(), n);
        let manifest = MiptManifest::new(
            seed, n, depth(n), num_circs, p_meas.iter().copied());
        manifest.save_all(&outdir)
            .expect("failed to generate and save circuit");
        eprint!("\r  {} / {}  (n = {:2}) ", k + 1, nqubits.len(), n);
    }
    eprintln!();
}

