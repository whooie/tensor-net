#![allow(unused_imports, unused_variables, unused_mut)]

use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, thread_rng };
use tensor_net::mps::*;
use tensor_net::tensor3::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Q(usize);

impl Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String { format!("{:?}", self) }
}

fn main() {
    let id: nd::Array2<C64>
        = nd::array![
            [C64::from(1.0), C64::from(0.0)],
            [C64::from(0.0), C64::from(1.0)],
        ];
    let h: nd::Array2<C64>
        = nd::array![
            [C64::from(0.5).sqrt(),  C64::from(0.5).sqrt()],
            [C64::from(0.5).sqrt(), -C64::from(0.5).sqrt()],
        ];
    let cx: nd::Array2<C64>
        = nd::array![
            [C64::from(1.0), C64::from(0.0), C64::from(0.0), C64::from(0.0)],
            [C64::from(0.0), C64::from(1.0), C64::from(0.0), C64::from(0.0)],
            [C64::from(0.0), C64::from(0.0), C64::from(0.0), C64::from(1.0)],
            [C64::from(0.0), C64::from(0.0), C64::from(1.0), C64::from(0.0)],
        ];

    let mut rng = thread_rng();
    let n: usize = 5;
    let indices: Vec<Q> = (0..n).map(Q).collect();

    let mut mps: MPS<Q, C64>
        = MPS::new(indices, Some(BondDim::Cutoff(1e-9))).unwrap();

    // let na: usize = indices.iter().map(|qk| qk.dim()).product();
    // let mut psi: nd::Array1<C64> = nd::Array1::zeros(na);
    // psi[0] = C64::from(1.0);
    // // psi[0] = C64::from(0.5).sqrt();
    // // psi[8] = C64::from(0.5).sqrt();
    // println!("{:.2}", psi);
    // let mut mps: MPS<Q, C64> = MPS::from_vector(indices, psi, None).unwrap();

    // let na: usize = indices.iter().map(|qk| qk.dim()).product();
    // let mut psi: nd::Array1<C64>
    //     = (0..na)
    //     .map(|_| C64 { re: rng.gen(), im: rng.gen() })
    //     .collect();
    // let norm = psi.iter().copied().map(|a| a * a.conj()).sum::<C64>().sqrt();
    // psi.iter_mut().for_each(|a| { *a /= norm; });
    // println!("{:.2}", psi);
    // let mut mps: MPS<Q, C64> = MPS::from_vector(indices, psi, None).unwrap();

    for k in 0..n - 1 {
        mps.apply_unitary1(k, &h).unwrap();
        mps.apply_unitary2(k, &cx).unwrap();
    }
    // println!("{:.2}", mps.contract());

    // mps.apply_unitary1(0, &h).unwrap();
    // mps.apply_unitary2(0, &cx).unwrap();
    // println!("{mps:?}");
    // println!("{:.2}", mps.contract());
    // mps.apply_unitary1(1, &h).unwrap();
    // mps.apply_unitary2(1, &cx).unwrap();
    // println!("{mps:?}");
    // println!("{:.2}", mps.contract());

    // let norm = mps.expectation_value(4, &id).unwrap();
    // println!("{}", norm);

    // let out = mps.measure(0, &mut rng).unwrap();
    // println!("{}", out);
    // let out = mps.measure(1, &mut rng).unwrap();
    // println!("{}", out);
    // println!("{mps:.2}");

    // for k in 0..n {
    //     println!("{:.2}", mps.local_norm(k));
    // }
    let outs: Vec<usize>
        = (0..n / 2)
        .map(|k| mps.measure(k, &mut rng).unwrap())
        .collect();
    println!("{:?}", outs);
    println!("{mps:?}");
    // let outs = mps.measure_multi([0, 1, 2, 3, 5, 6, 7, 8, 9, 10], &mut rng);

    // println!("{:.2}", mps);

    let (indices, state) = mps.into_contract();
    println!("{:?}", indices);
    println!("{:.2}", state);
}

