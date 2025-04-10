#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use ndarray::{ self as nd, linalg::kron };
use num_complex::Complex64 as C64;
use rand::{ Rng, SeedableRng, rngs::StdRng, thread_rng };
use tensor_net::{ circuit::*, gate::*, mps::* };
use whooie::write_npz;

#[derive(Clone, Debug)]
struct Ops {
    id: nd::Array2<C64>,
    proj0: nd::Array2<C64>,
    proj1: nd::Array2<C64>,
    tr0: nd::Array2<C64>,
    tr0_dag: nd::Array2<C64>,
    tr1: nd::Array2<C64>,
    tr1_dag: nd::Array2<C64>,
    u0: nd::Array2<C64>,
    u0_dag: nd::Array2<C64>,
    u1: nd::Array2<C64>,
    u1_dag: nd::Array2<C64>,
}

impl Ops {
    fn new<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let id: nd::Array2<C64> =
            nd::array![
                [1.0.into(), 0.0.into()],
                [0.0.into(), 1.0.into()],
            ];
        let proj0: nd::Array2<C64> =
            nd::array![
                [1.0.into(), 0.0.into()],
                [0.0.into(), 0.0.into()],
            ];
        let proj1: nd::Array2<C64> =
            nd::array![
                [0.0.into(), 0.0.into()],
                [0.0.into(), 1.0.into()],
            ];

        let tr0: nd::Array2<C64> =
            nd::array![
                [1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()],
                [0.0.into(), 0.0.into(), 1.0.into(), 0.0.into()],
            ];
        let tr0_dag = tr0.t().into_owned();
        let tr1: nd::Array2<C64> =
            nd::array![
                [0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()],
                [0.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
            ];
        let tr1_dag = tr1.t().into_owned();

        let u0: nd::Array2<C64> = haar(2, rng);
        let u0_dag = u0.t().mapv(|u| u.conj());
        let u1: nd::Array2<C64> = haar(2, rng);
        let u1_dag = u1.t().mapv(|u| u.conj());

        Self {
            id,
            proj0,
            proj1,
            tr0,
            tr0_dag,
            tr1,
            tr1_dag,
            u0,
            u0_dag,
            u1,
            u1_dag,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Data {
    p00: f64,
    p01: f64,
    p10: f64,
    p11: f64,
}

const EPSILON: f64 = 1e-12;

fn mps(ops: &Ops) -> Data {
    let mut state: MPS<Q, C64> = MPS::new_qubits(2, None).unwrap();
    state.apply_unitary2(0, &ops.u0).unwrap();
    let frozen = state; // ensure immutability
    // let psi: nd::Array1<C64> = frozen.contract();
    // let rho: nd::Array2<C64> =
    //     psi.iter().copied()
    //         .flat_map(|a| psi.iter().copied().map(move |b| a * b.conj()))
    //         .collect::<nd::Array1<C64>>()
    //         .into_shape((4, 4))
    //         .unwrap();
    // println!("{:+.3}", rho);

    let mut p0 = frozen.prob(0, 0).unwrap();
    let mut p1 = frozen.prob(0, 1).unwrap();
    let n = p0 + p1;
    if (n - 1.0).abs() >= EPSILON { panic!("bad probs"); }
    p0 /= n;
    p1 /= n;

    let (p00, p01) =
        if p0 >= EPSILON {
            let mut state0 = frozen.clone();
            state0.measure_postsel(0, 0);
            state0.apply_unitary2(0, &ops.u1).unwrap();
            let mut p0_0 = state0.prob(0, 0).unwrap();
            let mut p0_1 = state0.prob(0, 1).unwrap();
            let n = p0_0 + p0_1;
            if (n - 1.0).abs() >= EPSILON { panic!("bad probs"); }
            p0_0 /= n;
            p0_1 /= n;
            (p0 * p0_0, p0 * p0_1)
        } else { (0.0, 0.0) };

    let (p10, p11) =
        if p1 >= EPSILON {
            let mut state1 = frozen.clone();
            state1.measure_postsel(0, 1);
            state1.apply_unitary2(0, &ops.u1).unwrap();
            let mut p1_0 = state1.prob(0, 0).unwrap();
            let mut p1_1 = state1.prob(0, 1).unwrap();
            let n = p1_0 + p1_1;
            if (n - 1.0).abs() >= EPSILON { panic!("bad probs"); }
            p1_0 /= n;
            p1_1 /= n;
            (p1 * p1_0, p1 * p1_1)
        } else { (0.0, 0.0) };

    Data { p00, p01, p10, p11 }
}

fn mat(ops: &Ops) -> Data {
    let mut state: nd::Array2<C64> =
        nd::array![
            [1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()],
            [0.0.into(), 0.0.into(), 0.0.into(), 0.0.into()],
            [0.0.into(), 0.0.into(), 0.0.into(), 0.0.into()],
            [0.0.into(), 0.0.into(), 0.0.into(), 0.0.into()],
        ]; // ∣00⟩⟨00∣
    state = ops.u0.dot(&state).dot(&ops.u0_dag);
    let frozen = state; // ensure immutability
    // println!("{:+.3}", frozen);

    let reduced =
        &ops.tr0.dot(&frozen).dot(&ops.tr0_dag)
        + &ops.tr1.dot(&frozen).dot(&ops.tr1_dag);
    let mut p0 = reduced.dot(&ops.proj0).diag().sum().re;
    let mut p1 = reduced.dot(&ops.proj1).diag().sum().re;
    drop(reduced);
    let n = p0 + p1;
    if (n - 1.0).abs() >= EPSILON { panic!("bad probs"); }
    p0 /= n;
    p1 /= n;

    let (p00, p01) =
        if p0 >= EPSILON {
            let mut state0 =
                kron(&ops.proj0, &ops.id)
                .dot(&frozen)
                .dot(&kron(&ops.proj0, &ops.id))
                / p0;
            state0 = ops.u1.dot(&state0).dot(&ops.u1_dag);
            let reduced =
                &ops.tr0.dot(&state0).dot(&ops.tr0_dag)
                + &ops.tr1.dot(&state0).dot(&ops.tr1_dag);
            let mut p0_0 = reduced.dot(&ops.proj0).diag().sum().re;
            let mut p0_1 = reduced.dot(&ops.proj1).diag().sum().re;
            drop(reduced);
            let n = p0_0 + p0_1;
            if (n - 1.0).abs() >= EPSILON { panic!("bad probs"); }
            p0_0 /= n;
            p0_1 /= n;
            (p0 * p0_0, p0 * p0_1)
        } else { (0.0, 0.0) };

    let (p10, p11) =
        if p1 >= EPSILON {
            let mut state1 =
                kron(&ops.proj1, &ops.id)
                .dot(&frozen)
                .dot(&kron(&ops.proj1, &ops.id))
                / p1;
            state1 = ops.u1.dot(&state1).dot(&ops.u1_dag);
            let reduced =
                &ops.tr0.dot(&state1).dot(&ops.tr0_dag)
                + &ops.tr1.dot(&state1).dot(&ops.tr1_dag);
            let mut p1_0 = reduced.dot(&ops.proj0).diag().sum().re;
            let mut p1_1 = reduced.dot(&ops.proj1).diag().sum().re;
            drop(reduced);
            let n = p1_0 + p1_1;
            if (n - 1.0).abs() >= EPSILON { panic!("bad probs"); }
            p1_0 /= n;
            p1_1 /= n;
            (p1 * p1_0, p1 * p1_1)
        } else { (0.0, 0.0) };

    Data { p00, p01, p10, p11 }
}

fn main() {
    let mut rng = thread_rng();
    // let mut rng = StdRng::seed_from_u64(10546);

    let ops = Ops::new(&mut rng);
    println!("{:+.3}", ops.u1);
    let data_mps = mps(&ops);
    let data_mat = mat(&ops);

    println!("{:.3?}", data_mps);
    println!("{:.3?}", data_mat);

}

