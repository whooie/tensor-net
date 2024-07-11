use tensor_net::gate::*;
use tensor_net::mps::*;

fn main() {
    let mut mps = MPS::new_qubits(4, None).unwrap();
    println!("t=0");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(0)).apply_gate(Gate::CZ(2));
    println!("t=1");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(1));
    println!("t=2");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(0)).apply_gate(Gate::CZ(2));
    println!("t=3");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(1));
    println!("t=4");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(0)).apply_gate(Gate::CZ(2));
    println!("t=5");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(1));
    println!("t=6");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(0)).apply_gate(Gate::CZ(2));
    println!("t=7");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());

    (0..4).for_each(|k| { mps.apply_gate(Gate::H(k)); });
    mps.apply_gate(Gate::CZ(1));
    println!("t=8");
    println!("{:+.2}", mps.contract());
    println!("{}", mps.entropy_vn(1).unwrap());
}

