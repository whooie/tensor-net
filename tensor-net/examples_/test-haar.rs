use std::path::PathBuf;
use num_complex::Complex64 as C64;
use ndarray as nd;
use rand::{ thread_rng, Rng };
use whooie::write_npz;
use tensor_net::gate::haar;

const MC: usize = 100000;

fn main() {
    let mut rng = thread_rng();

    let mut psi0: nd::Array1<C64> = nd::array![
        C64::new(rng.gen(), rng.gen()),
        C64::new(rng.gen(), rng.gen()),
    ];
    let norm = psi0.iter().map(|a| *a * a.conj()).sum::<C64>().sqrt();
    psi0.iter_mut().for_each(|a| { *a /= norm; });

    let mut q: nd::Array2<C64> = nd::Array2::zeros((MC, 2));
    q.axis_iter_mut(nd::Axis(0))
        .for_each(|qi| {
            let u: nd::Array2<C64> = haar(1, &mut rng);
            u.dot(&psi0).move_into(qi);
        });

    write_npz!(
        PathBuf::from("test-haar.npz"),
        arrays: { "q" => &q }
    );
}


