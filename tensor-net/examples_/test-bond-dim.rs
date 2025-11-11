#![allow(unused_imports, unused_variables, unused_mut)]

use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, thread_rng };
use tensor_net::mps::*;
use tensor_net::tensor::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Q(usize);

impl Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String { format!("{:?}", self) }
}

fn main() {
    let mut rng = thread_rng();
    let n: usize = 5;
    let indices: Vec<Q> = (0..n).map(Q).collect();

    let na: usize = indices.iter().map(|qk| qk.dim()).product();
    let mut psi: nd::Array1<C64>
        = (0..na)
        .map(|_| C64 { re: rng.gen(), im: rng.gen() })
        .collect();
    let norm = psi.iter().copied().map(|a| a * a.conj()).sum::<C64>().sqrt();
    psi.iter_mut().for_each(|a| { *a /= norm; });
    println!("{:.2}", psi);
    let bond: Option<BondDim<f64>>
        = Some(BondDim::Const(4));
    let mut mps: MPS<Q, C64> = MPS::from_vector(indices, psi, bond).unwrap();

    // // for k in 0..n {
    // //     println!("{:.2}", mps.local_norm(k));
    // // }
    // let outs: Vec<usize>
    //     = (0..n / 2)
    //     .map(|k| mps.measure(k, &mut rng).unwrap())
    //     .collect();
    // println!("{:?}", outs);
    // println!("{mps:?}");
    // // let outs = mps.measure_multi([0, 1, 2, 3, 5, 6, 7, 8, 9, 10], &mut rng);

    // println!("{:.2}", mps);

    let (indices, state) = mps.into_contract();
    println!("{:?}", indices);
    println!("{:.2}", state);
}

