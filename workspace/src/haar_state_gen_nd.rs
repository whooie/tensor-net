#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::{ ops:: Range, path::PathBuf };
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ SeedableRng, rngs::StdRng, thread_rng };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_net::{ circuit::Q, mps::* };
use whooie::write_npz;
use lib::{ *, haar::* };

const N: usize = 12;
const T: usize = 2 * N;
const TARGET_X: usize = N / 2;
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

fn sample_matrix_record(
    bond: Option<BondDim<f64>>,
    circ: (&[UniLayer], &[MeasLayer]),
    part: Range<usize>,
) -> (nd::Array2<C64>, Vec<Vec<Meas>>)
{
    assert_eq!(circ.0.len(), circ.1.len());
    let mut rng = thread_rng();
    let mut state: MPS<Q, C64> = MPS::new_qubits(N, bond).unwrap();
    let meas_record: Vec<Vec<Meas>> =
        circ.0.iter().zip(circ.1.iter())
            .map(|(unis, meas)| {
                apply_main_layer(&mut state, unis, meas, &mut rng)
            })
            .collect();
    (state.into_matrix_part(part).1, meas_record)
}

#[allow(static_mut_refs)]
fn main() {
    const SEED: u64 = 10546;
    // circuit generation RNG
    // let mut rng = thread_rng();
    let mut rng = StdRng::seed_from_u64(SEED);

    // target qubits
    let part = rev_cone_range(N, TARGET_X, T, 1);
    let subsize = part.end - part.start;
    let rho_size = 2_usize.pow(subsize as u32);

    // rho_data  :: { p, chi, run, i, j }
    let mut rho_data: nd::Array5<C64> =
        nd::Array::zeros((P_MEAS.len(), BONDS.len(), RUNS, rho_size, rho_size));
    // meas_data :: { p, chi, run, t, x }
    let mut meas_data: nd::Array5<i8> =
        nd::Array::zeros((P_MEAS.len(), BONDS.len(), RUNS, T, N));

    let w_p: usize = (P_MEAS.len() as f64).log10().floor() as usize + 1;
    let w_chi: usize = (BONDS.len() as f64).log10().floor() as usize + 1;
    let w_run: usize = (RUNS as f64).log10().floor() as usize + 1;

    let unis: Vec<UniLayer> =
        (0..T)
        .map(|t| UniLayer::gen(N, t % 2 == 1, &mut rng))
        .collect();
    let meas_p: Vec<Vec<MeasLayer>> =
        P_MEAS.iter().copied()
        .map(|p| (0..T).map(|_| MeasLayer::gen(N, p, &mut rng)).collect())
        .collect();

    eprint!(" {:w_p$} / {:w_p$} ", 0, P_MEAS.len());
    let p_iter =
        rho_data.outer_iter_mut()
        .zip(meas_data.outer_iter_mut())
        .zip(meas_p.iter())
        .enumerate();
    for (i, ((mut rho_p, mut meas_p), m_p)) in p_iter {
        eprint!("\x1b[{}D{:w_p$} / {:w_p$} ",
            2 * w_p + 4, i + 1, P_MEAS.len()
        );

        eprint!(" {:w_chi$} / {:w_chi$} ", 0, BONDS.len());
        let chi_iter =
            rho_p.outer_iter_mut()
            .zip(meas_p.outer_iter_mut())
            .zip(BONDS.iter().copied())
            .enumerate();
        for (j, ((mut rho_px, mut meas_px), mb_chi)) in chi_iter {
            eprint!("\x1b[{}D{:w_chi$} / {:w_chi$} ",
                2 * w_chi + 4, j + 1, BONDS.len()
            );

            let mb_bond = mb_chi.map(BondDim::Const);
            static mut RUN: usize = 0;
            unsafe {
                RUN = 0;
                eprint!(" {:w_run$} / {:w_run$} ", RUN, RUNS);
            }
            nd::Zip::from(rho_px.outer_iter_mut())
                .and(meas_px.outer_iter_mut())
                .par_for_each(|mut rho_pxr, mut meas_pxr| {
                    let (rho, meas_rec) = sample_matrix_record(
                        mb_bond, (&unis, m_p), part.clone());
                    rho.move_into(&mut rho_pxr);
                    meas_rec.into_iter()
                        .zip(meas_pxr.outer_iter_mut())
                        .for_each(|(m_layer, mut layer_rec)| {
                            m_layer.into_iter()
                                .for_each(|m| {
                                    match m {
                                        Meas::Rand(_) => { },
                                        Meas::Postsel(k, out) => {
                                            layer_rec[k] =
                                                if out { 1 } else { -1 };
                                        },
                                    }
                                });
                        });
                    unsafe {
                        RUN += 1;
                        eprint!("\x1b[{}D{:w_run$} / {:w_run$} ",
                            2 * w_run + 4, RUN, RUNS
                        );
                    }
                });
            eprint!("\x1b[{}D", 2 * w_run + 5);
        }
        eprint!("\x1b[{}D", 2 * w_chi + 5);
    }
    eprintln!();

    let outdir = PathBuf::from("output").join("haar_state_gen");
    let fname =
        format!(
            "haar_state_gen_n={}_d={}_runs={}_seed={}.npz",
            N, T, RUNS, SEED
        );
    println!("{}", fname);
    write_npz!(
        outdir.join(fname),
        arrays: {
            "size" => &nd::array![N as i32],
            "depth" => &nd::array![T as i32],
            "p_meas" => &P_MEAS.iter().copied().collect::<nd::Array1<f64>>(),
            "chi" =>
                &BONDS.iter().copied()
                .map(|b| b.unwrap_or(0) as i32)
                .collect::<nd::Array1<i32>>(),
            "target_x" => &nd::array![TARGET_X as i32],
            "target_range" => &nd::array![part.start as i32, part.end as i32],
            "rho" => &rho_data,
            "meas" => &meas_data,
            "seed" => &nd::array![SEED as i32],
        }
    );
}





