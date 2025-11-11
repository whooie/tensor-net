use tensor_net::network::*;
use tensor_net::pool::*;
use tensor_net::tensor::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Index {
    A(usize),
    B,
    C,
    D,
}
use Index::*;

impl Idx for Index {
    fn dim(&self) -> usize {
        match self {
            Self::A(_) => 3,
            Self::B => 4,
            Self::C => 5,
            Self::D => 2,
        }
    }

    fn label(&self) -> String { format!("{:?}", self) }
}

fn main() {
    let a = Tensor::new([A(0), B], |_| 1_u32).unwrap();
    let b = Tensor::new([C, B], |_| 2_u32).unwrap();
    let c = Tensor::new([A(0), C], |_| 3_u32).unwrap();
    let d = Tensor::new([D], |_| 4_u32).unwrap();

    let net = Network::from_nodes([a, b, c, d]).unwrap();
    let pool = ContractorPool::new_cpus();
    let res = net.contract_par(&pool).unwrap();
    println!("{}", res);
}

