#![allow(unused_imports)]

use tensor_net::network::*;
use tensor_net::pool::*;
use tensor_net::tensor::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Index {
    A0p,
    A1p,
    B1p,
    A0,
    A1,
    B1,
}
use Index::*;

impl Idx for Index {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String { format!("{:?}", self) }
}

fn main() {
    let rho = Tensor::new( // 00 + 11 Bell state
        [A0p, A1p, A0, A1],
        |idx| {
            match idx {
                &[0, 0, 0, 0] | &[0, 0, 1, 1] | &[1, 1, 0, 0] | &[1, 1, 1, 1]
                    => 0.5_f32,
                _ => 0.0_f32,
            }
        }
    ).unwrap();
    let tr = Tensor::new( // trace out qubit 1
        [A1p, A1],
        |idx| {
            match idx {
                &[0, 0] | &[1, 1] => 1.0_f32,
                _ => 0.0_f32,
            }
        }
    ).unwrap();
    let proj = Tensor::new( // projective measurement on qubit 1
        [A1p, A1, B1p, B1],
        |idx| {
            match idx {
                &[0, 0, 0, 0] | &[1, 1, 1, 1] => 1.0_f32,
                _ => 0.0_f32,
            }
        }
    ).unwrap();
    println!("ρ =\n{rho}\n");
    println!("T =\n{tr}\n");
    println!("P =\n{proj}\n");

    let mut rho0 = rho.clone().multiply(tr);
    rho0.sort_indices();
    let mut rho = rho.multiply(proj);
    rho.sort_indices();
    println!("ρ0 =\n{rho0}\n");
    println!("ρ' =\n{rho}\n");
}

