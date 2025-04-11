//! A collection of tensors arranged in a graph, where nodes are tensors and
//! edges are determined by common indices.
//!
//! [`Network`]s track which indices belong to which tensors and additionally
//! whether a particular index is shared between two tensors. Network edges are
//! identified by common indices, so each particular index can only exist as
//! either an unbonded degree of freedom or a bond between exactly two tensors
//! in the network.
//!
//! ```ignore
//! use tensor_net::{
//!     network::{ Network, Pool },
//!     tensor::{ Idx, Tensor },
//! };
//!
//! // our index type (see the Idx trait for static versus dynamic index types)
//! #[derive(Copy, Clone, Debug, PartialEq, Eq)]
//! enum Index { A, B, C, D }
//!
//! // every index has a dimension and a label
//! impl Idx for Index {
//!     fn dim(&self) -> usize {
//!         match self {
//!             Self::A => 3,
//!             Self::B => 4,
//!             Self::C => 5,
//!             Self::D => 2,
//!         }
//!     }
//!
//!     fn label(&self) -> String { format!("{:?}", self) }
//! }
//!
//! impl std::fmt::Display for Index {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         write!(f, "{self:?}")
//!     }
//! }
//!
//! let a = Tensor::new([Index::A, Index::B], |_| 1.0).unwrap();
//! println!("{}", a);
//! // { <A>, <B> }
//! // ┌         ┐
//! // │ 1 1 1 1 │
//! // │ 1 1 1 1 │
//! // │ 1 1 1 1 │
//! // └         ┘
//!
//! let b = Tensor::new([Index::B, Index::C], |_| 2.0).unwrap();
//! println!("{}", b);
//! // { <B>, <C> }
//! // ┌           ┐
//! // │ 2 2 2 2 2 │
//! // │ 2 2 2 2 2 │
//! // │ 2 2 2 2 2 │
//! // │ 2 2 2 2 2 │
//! // └           ┘
//!
//! let c = Tensor::new([Index::A, Index::C], |_| 3.0).unwrap();
//! println!("{}", c);
//! // { <A>, <C> }
//! // ┌           ┐
//! // │ 8 8 8 8 8 │
//! // │ 8 8 8 8 8 │
//! // │ 8 8 8 8 8 │
//! // └           ┘
//!
//! let d = Tensor::new([Index::D], |_| 4.0).unwrap();
//! println!("{}", d);
//! // { <D>, <> }
//! // ┌   ┐
//! // │ 4 │
//! // │ 4 │
//! // └   ┘
//!
//! // construct a network to compute the contraction (in Einstein sum notation)
//! // A_{a,b} B_{b,c} C_{a,c} D_{d}
//! let net = Network::from_nodes([a, b, c, d]).unwrap();
//!
//! // contraction can be performed in parallel using a thread pool
//! let pool = Pool::new_cpus(); // #threads == #cpu cores
//!
//! let res = net.contract_par(&pool).unwrap();
//! println!("{}", res);
//! // { <D>, <> }
//! // ┌      ┐
//! // │ 1440 │
//! // │ 1440 │
//! // └      ┘
//! ```

#![allow(unused_imports, dead_code)]

use std::{
    ops::{ Deref, DerefMut },
    thread,
};
use crossbeam::channel;
use thiserror::Error;
use crate::tensor::{ self, Tensor, Idx };

pub type Id = usize;

/// An edge in a [`Network`].
///
/// A wire is either an unbonded degree of freedom belonging to a single tensor
/// or a bond between exactly two tensors in the network.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Wire {
    Unpaired(Id),
    Paired(Id, Id),
}

/// A graph of [`Tensor`]s.
///
/// Every tensor in the network must have the same data type `A` and index type
/// `T`. See [`Idx`] for info on dynamic versus static index types.
#[derive(Clone, Debug)]
pub struct Network<T, A> {
    nodes: Vec<Option<Tensor<T, A>>>,
    node_count: usize,
    neighbors: Vec<Option<Vec<Id>>>,
    indices: Vec<(T, Wire)>,
    free: Vec<Id>,
}

impl<T, A> Default for Network<T, A> {
    fn default() -> Self { Self::new() }
}

impl<T, A> Network<T, A> {
    /// Create a new, empty network.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_count: 0,
            neighbors: Vec::new(),
            indices: Vec::new(),
            free: Vec::new(),
        }
    }
}

