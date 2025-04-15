//! A collection of tensors arranged in a graph, where nodes are tensors and
//! edges are determined by common indices.
//!
//! [`Network`]s track which indices belong to which tensors and additionally
//! whether a particular index is shared between two tensors. Network edges are
//! identified by common indices, so each particular index can only exist as
//! either an unbonded degree of freedom or a bond between exactly two tensors
//! in the network.
//!
//! ```
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
//! // │ 3 3 3 3 3 │
//! // │ 3 3 3 3 3 │
//! // │ 3 3 3 3 3 │
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

use std::thread;
use crossbeam::channel;
use nalgebra::ComplexField;
use thiserror::Error;
use crate::{
    id_set::{ self, IdSet },
    tensor::{ Tensor, Idx },
};

#[derive(Error, Debug)]
pub enum NetworkError {
    /// Returned when attempting to add a tensor to a network in which an
    /// instance of one or more of the tensor's indices already exists and is
    /// paired with another tensor in the network.
    #[error("error in Network::add_tensor: pre-existing index {0}")]
    PreExistingIndex(String),

    /// Returned when attempting to contract with a tensor ID, but it doesn't
    /// exist in the network.
    #[error("error in Network::contract_single: missing ID {0}")]
    ContractMissingId(usize),

    /// Returned when attempting to contract on an unpaired index.
    #[error("error in Network::contract_single_index: index {0} is unpaired")]
    ContractUnpaired(String),

    /// Returned when attempting to contract on an index that does not exist in
    /// a network.
    #[error("error in Network::contract_single_index: missing index {0}")]
    ContractMissing(String),

    /// Returned by anything involving a [`Pool`].
    #[error("contractor pool error: {0}")]
    PoolError(#[from] PoolError),
}
use NetworkError::*;
pub type NetworkResult<T> = Result<T, NetworkError>;

/// An identifying value for a particular tensor in a [`Network`].
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

impl Wire {
    /// Return `true` if `self` is `Unpaired`.
    pub fn is_unpaired(&self) -> bool { matches!(self, Self::Unpaired(_)) }

    /// Return `true` if `self` is `Unpaired` and the ID satisfies a predicate.
    pub fn is_unpaired_and<F>(&self, pred: F) -> bool
    where F: FnOnce(Id) -> bool
    {
        match self {
            Self::Unpaired(k) => pred(*k),
            Self::Paired(..) => false,
        }
    }

    /// Return `true` if `self` is `Unpaired`, or if `self` is `Paired` and the
    /// two IDs satisfy a predicate.
    pub fn is_unpaired_or<F>(&self, pred: F) -> bool
    where F: FnOnce(Id, Id) -> bool
    {
        match self {
            Self::Unpaired(_) => true,
            Self::Paired(l, r) => pred(*l, *r),
        }
    }

    /// Return `true` if `self` is `Paired`.
    pub fn is_paired(&self) -> bool { matches!(self, Self::Paired(..)) }

    /// Return `true` if `self` is `Paired` and the two IDs satisfy a predicate.
    pub fn is_paired_and<F>(&self, pred: F) -> bool
    where F: FnOnce(Id, Id) -> bool
    {
        match self {
            Self::Unpaired(_) => false,
            Self::Paired(l, r) => pred(*l, *r),
        }
    }

    /// Return `true` if `self` is `Paired`, or if `self` is `Unpaired` and the
    /// ID satisfies a predicate.
    pub fn is_paired_or<F>(&self, pred: F) -> bool
    where F: FnOnce(Id) -> bool
    {
        match self {
            Self::Unpaired(k) => pred(*k),
            Self::Paired(..) => true,
        }
    }

    /// Return `true` if any ID held by `self` matches `id`.
    pub fn contains(&self, id: Id) -> bool {
        match self {
            Self::Unpaired(k) => *k == id,
            Self::Paired(l, r) => *l == id || *r == id,
        }
    }

    /// Return `true` if `self` is `Paired` with one of the IDs equal to `id`.
    pub fn is_paired_with(&self, id: Id) -> bool {
        match self {
            Self::Unpaired(_) => false,
            Self::Paired(l, r) => *l == id || *r == id,
        }
    }

    /// Return `true` if `self` is `Paired` and contains both `a` and `b`.
    pub fn connects(&self, a: Id, b: Id) -> bool {
        match self {
            Self::Unpaired(_) => false,
            Self::Paired(l, r) => (*l == a && *r == b) || (*l == b && *r == a),
        }
    }

    /// If `self` is `Unpaired`, convert *in place* to `Paired` with the
    /// pre-existing id on the left.
    ///
    /// This is a no-op if `self` is already `Paired`.
    pub fn pair_with(&mut self, id: Id) {
        match self {
            Self::Unpaired(k) => { *self = Self::Paired(*k, id); },
            Self::Paired(..) => { },
        }
    }

    /// Replace the left index of a `Pair` with `id` and return the previous
    /// value. If `self` is `Unpaired`, convert `self` in place to `Paired` with
    /// `id` on the left.
    pub fn make_pair_left(&mut self, id: Id) -> Option<Id> {
        match self {
            Self::Unpaired(k) => {
                *self = Self::Paired(id, *k);
                None
            },
            Self::Paired(l, _) => {
                Some(std::mem::replace(l, id))
            },
        }
    }

    /// Replace the right index of a `Pair` with `id` and return the previous
    /// value. If `self` is `Unpaired`, convert `self` in place to `Paired` with
    /// `id` on the right.
    pub fn make_pair_right(&mut self, id: Id) -> Option<Id> {
        match self {
            Self::Unpaired(k) => {
                *self = Self::Paired(*k, id);
                None
            },
            Self::Paired(_, r) => {
                Some(std::mem::replace(r, id))
            },
        }
    }

    /// If `self` is `Paired` and one of the IDs is equal to `id`, return the
    /// other ID.
    pub fn other_id(&self, id: Id) -> Option<Id> {
        match self {
            Self::Unpaired(_) => None,
            Self::Paired(l, r) => {
                if *l == id {
                    Some(*r)
                } else if *r == id {
                    Some(*l)
                } else {
                    None
                }
            },
        }
    }
}

/// A graph of [`Tensor`]s.
///
/// Every tensor in the network must have the same data type `A` and index type
/// `T`. See [`Idx`] for info on dynamic versus static index types.
#[derive(Clone, Debug)]
pub struct Network<T, A> {
    nodes: IdSet<Tensor<T, A>>,
    wires: IdSet<(T, Wire)>,
}

impl<T, A> Default for Network<T, A> {
    fn default() -> Self { Self::new() }
}

impl<T, A> Network<T, A> {
    /// Create a new, empty network.
    pub fn new() -> Self {
        Self { nodes: IdSet::new(), wires: IdSet::new() }
    }

    /// Return the number of tensors in the network.
    pub fn count_nodes(&self) -> usize { self.nodes.len() }

    /// Return the number of indices in the network.
    pub fn count_wires(&self) -> usize { self.wires.len() }

    /// Return the number of paired indices in the network.
    pub fn count_internal_wires(&self) -> usize {
        self.wires.iter().filter(|(_, w)| w.is_paired()).count()
    }

    /// Return the number of unpaired indices in the network.
    pub fn count_free_wires(&self) -> usize {
        self.wires.iter().filter(|(_, w)| w.is_unpaired()).count()
    }

    /// Return `true` if `self` contains a tensor at `id`.
    pub fn has_id(&self, id: Id) -> bool { self.nodes.contains_id(id) }

    /// Return a reference to a specific tensor in the network, if it exists.
    pub fn get(&self, id: Id) -> Option<&Tensor<T, A>> {
        self.nodes.get(id)
    }

    /// Return a mutable reference to a specific tensor in the network, if it
    /// exists.
    pub fn get_mut(&mut self, id: Id) -> Option<&mut Tensor<T, A>> {
        self.nodes.get_mut(id)
    }

    /// Return the total number of wires from a given tensor that go to another
    /// tensor in the network, if it exists.
    ///
    /// This quantity is at most the rank of the tensor.
    pub fn internal_degree(&self, id: Id) -> Option<usize> {
        self.nodes.contains_id(id)
            .then(|| {
                self.wires.iter()
                .filter(|(_, w)| w.is_paired_with(id))
                .count()
            })
    }

    /// Return an iterator over all tensors in the network, visited in ID order.
    ///
    /// The iterator item type is `(`[`Id`]`, &`[`Tensor<T, A>`]`)`.
    pub fn nodes(&self) -> Nodes<'_, T, A> {
        Nodes { iter: self.nodes.iter_id() }
    }

    /// Return an iterator over all indices in the network, visited in an
    /// arbitrary order.
    ///
    /// The iterator item type is `(&T, `[`Wire`]`)`.
    pub fn indices(&self) -> Indices<'_, T> {
        Indices { iter: self.wires.iter() }
    }

    /// Return an iterator over all internal indices (connected to tensors on
    /// both ends) in the network, visited in an arbitrary order.
    ///
    /// The iterator item type is `(&T, `[`Id`]`, `[`Id`]`)`.
    pub fn internal_indices(&self) -> InternalIndices<'_, T> {
        InternalIndices { iter: self.wires.iter() }
    }

    /// Return an iterator over all indices connecting two tensors, visited in
    /// an arbitrary order, if they both exist.
    ///
    /// The iterator item type is `&T`.
    pub fn indices_between(&self, a: Id, b: Id)
        -> Option<IndicesBetween<'_, T>>
    {
        (self.has_id(a) && self.has_id(b))
            .then(|| IndicesBetween { a, b, iter: self.wires.iter() })
    }

    /// Return an iterator over all neighbors of a given node and their
    /// indices, visited in an arbitrary order, if the tensor exists.
    ///
    /// The iterator item type is `(&T, `[`Id`]`, &`[`Tensor<T, A>`]`)`.
    pub fn neighbors(&self, id: Id) -> Option<Neighbors<'_, T, A>>
    {
        self.has_id(id)
            .then(|| {
                Neighbors {
                    network: self,
                    target: id,
                    iter: self.wires.iter(),
                }
            })
    }
}

impl<T, A> Network<T, A>
where T: Idx
{
    /// Create a new network from an iterator by repeatedly
    /// [pushing][Self::push] new tensors onto an initially empty network.
    ///
    /// Note that [node IDs][Id] count from zero and no nodes will be removed
    /// during this process, so the IDs `0, ..., n - 1` will correspond uniquely
    /// to the first `n` nodes in the iterator, in order.
    ///
    /// Fails if the iteration sees a tensor containing an index that exists in
    /// a pair between two tensors already in the network.
    pub fn from_nodes<I>(nodes: I) -> NetworkResult<Self>
    where I: IntoIterator<Item = Tensor<T, A>>
    {
        let mut new = Self::new();
        for node in nodes.into_iter() {
            new.push(node)?;
        }
        Ok(new)
    }

    /// Return the tensor ID(s) associated with a particular index, if the index
    /// exists.
    pub fn get_wire(&self, idx: &T) -> Option<Wire> {
        self.wires.iter()
            .find_map(|(idx2, wire)| (idx == idx2).then_some(*wire))
    }

    /// Return the total number of wires attached to a given tensor, equal to
    /// its rank, if it exists.
    pub fn degree(&self, id: Id) -> Option<usize> {
        self.get(id).map(|n| n.rank())
    }

    /// Add a new tensor to the network and return its ID.
    ///
    /// Fails if any of the new tensor's indices already exist in the network as
    /// a bond between two other tensors.
    pub fn push(&mut self, tensor: Tensor<T, A>) -> NetworkResult<Id> {
        let idx_check: Option<&T> =
            tensor.indices()
            .find(|idx| {
                self.wires.iter()
                .any(|(idx2, wire)| *idx == idx2 && wire.is_paired())
            });
        if let Some(idx) = idx_check {
            return Err(PreExistingIndex(idx.label()));
        }
        let id = self.nodes.next_id();
        for idx in tensor.indices() {
            let in_self: Option<&mut Wire> =
                self.wires.iter_mut()
                .find_map(|(idx2, wire)| (idx == idx2).then_some(wire));
            match in_self {
                Some(Wire::Paired(..)) => unreachable!(),
                Some(unpaired) => { unpaired.pair_with(id); }
                None => { self.wires.insert((idx.clone(), Wire::Unpaired(id))); },
            }
        }
        Ok(self.nodes.insert(tensor))
    }

    /// Remove a tensor from the network.
    pub fn remove(&mut self, id: Id) -> Option<Tensor<T, A>> {
        if let Some(tensor) = self.nodes.remove(id) {
            for idx in tensor.indices() {
                let mb_wire: Option<(Id, &mut Wire)> =
                    self.wires.iter_mut_id()
                    .find_map(|(wid, (idx2, wire))| {
                        (idx == idx2).then_some((wid, wire))
                    });
                if let Some((wid, wire)) = mb_wire {
                    if wire.is_paired() {
                        *wire = Wire::Unpaired(wire.other_id(id).unwrap());
                    } else {
                        self.wires.remove(wid);
                    }
                }
            }
            Some(tensor)
        } else {
            None
        }
    }
}

impl<T, A> Network<T, A>
where
    T: Idx,
    A: ComplexField,
{
    /// Contract a single pair of tensors named by the IDs `a` and `b`.
    ///
    /// Returns the ID of the resulting tensor.
    pub fn contract_single(&mut self, a: Id, b: Id) -> NetworkResult<Id> {
        let a = self.remove(a).ok_or(ContractMissingId(a))?;
        let b = self.remove(b).ok_or(ContractMissingId(b))?;
        let c = a * b;
        self.push(c)
    }

    /// Like [`contract_single`][Self::contract_single], but with the
    /// contraction named by an index to contract on instead of the pair of
    /// tensors it connects.
    ///
    /// Note that contractions between tensors are automatically performed over
    /// all shared indices, so there's no need to call this method for each of
    /// the indices shared by a pair of tensors.
    pub fn contract_single_index(&mut self, idx: &T) -> NetworkResult<Id> {
        let mb_wire =
            self.wires.iter()
            .find_map(|(idx2, wire)| (idx == idx2).then_some(wire));
        match mb_wire {
            Some(Wire::Paired(a, b)) => self.contract_single(*a, *b),
            Some(Wire::Unpaired(_)) => Err(ContractUnpaired(idx.label())),
            None => Err(ContractMissing(idx.label())),
        }
    }

    // find the next contraction step as the pair of tensors whose contraction
    // results in a tensor of lowest total dimension
    fn find_contraction(&self) -> Option<(Id, Id)> {
        fn costf<U, B>(net: &Network<U, B>, id_a: Id, id_b: Id) -> usize
        where U: Idx
        {
            let a = net.get(id_a).unwrap();
            let b = net.get(id_b).unwrap();
            a.indices()
                .map(|idx| idx.dim())
                .chain(
                    b.indices()
                    .filter(|idx| !a.has_index(idx))
                    .map(|idx| idx.dim())
                )
                .product()
        }

        self.internal_indices()
            .min_by(|(_, id_al, id_bl), (_, id_ar, id_br)| {
                costf(self, *id_al, *id_bl).cmp(&costf(self, *id_ar, *id_br))
            })
            .map(|(_, id_a, id_b)| (id_a, id_b))
    }

    // greedily contract all pairs of tensors that share at least one index
    // following `find_contraction` until no paired tensors remain
    //
    // this may leave more than one tensor left in the network
    fn do_contract(&mut self) -> &mut Self {
        while let Some((a, b)) = self.find_contraction() {
            self.contract_single(a, b).unwrap();
        }
        self
    }

    /// Contract the entire network into a single output tensor.
    pub fn contract(mut self) -> Tensor<T, A> {
        self.do_contract();
        if self.count_nodes() > 0 {
            let mut remaining_nodes = self.into_iter();
            let acc = remaining_nodes.next().unwrap();
            remaining_nodes.fold(acc, |acc, t| acc * t)
        } else {
            Tensor::new_scalar(A::one())
        }
    }

    /// Contract the network completely byt don't consume `self`, instead
    /// removing and returning the final result.
    ///
    /// The network will be left empty but with memory still allocated after
    /// calling this method.
    pub fn contract_remove(&mut self) -> Tensor<T, A> {
        self.do_contract();
        if self.count_nodes() > 0 {
            self.wires.clear();
            let mut remaining_nodes = self.nodes.drain(..);
            let acc = remaining_nodes.next().unwrap();
            remaining_nodes.fold(acc, |acc, t| acc * t)
        } else {
            Tensor::new_scalar(A::one())
        }
    }

    /// Contract all paired indices, leaving the result as a network.
    ///
    /// The network may be left with multiple tensors remaining if they have no
    /// common indices.
    pub fn contract_network(&mut self) -> &mut Self { self.do_contract() }
}

impl<T, A> Network<T, A>
where
    T: Idx + Send + 'static,
    A: ComplexField + Send + Sync + 'static,
{
    fn do_contract_par(&mut self, pool: &Pool<T, A>)
        -> NetworkResult<&mut Self>
    {
        let mut contractions: Vec<(Tensor<T, A>, Tensor<T, A>)> =
            Vec::with_capacity(self.count_nodes() / 2);
        let mut results: Vec<Tensor<T, A>>;
        while self.internal_indices().count() > 0 {
            while let Some((a, b)) = self.find_contraction() {
                let t_a = self.remove(a).unwrap();
                let t_b = self.remove(b).unwrap();
                contractions.push((t_a, t_b));
            }
            results = pool.do_contractions(contractions.drain(..))?;
            results.into_iter()
                .try_for_each(|t| self.push(t).map(|_| ()))?;
        }
        Ok(self)
    }

    /// Like [`contract`][Self::contract], but using a thread pool.
    pub fn contract_par(mut self, pool: &Pool<T, A>)
        -> NetworkResult<Tensor<T, A>>
    {
        self.do_contract_par(pool)?;
        if self.count_nodes() > 0 {
            let mut remaining_nodes = self.into_iter();
            let acc = remaining_nodes.next().unwrap();
            Ok(remaining_nodes.fold(acc, |acc, t| acc * t))
        } else {
            Ok(Tensor::new_scalar(A::one()))
        }
    }

    /// Like [`contract_remove`][Self::contract_remove], but using a thread
    /// pool.
    pub fn contract_remove_par(&mut self, pool: &Pool<T, A>)
        -> NetworkResult<Tensor<T, A>>
    {
        self.do_contract_par(pool)?;
        if self.count_nodes() > 0 {
            self.wires.clear();
            let mut remaining_nodes = self.nodes.drain(..);
            let acc = remaining_nodes.next().unwrap();
            Ok(remaining_nodes.fold(acc, |acc, t| acc * t))
        } else {
            Ok(Tensor::new_scalar(A::one()))
        }
    }

    /// Like [`contract_network`][Self::contract_network], but using a thread
    /// pool.
    pub fn contract_network_par(&mut self, pool: &Pool<T, A>)
        -> NetworkResult<&mut Self>
    {
        self.do_contract_par(pool)?;
        Ok(self)
    }
}

impl<T, A> IntoIterator for Network<T, A> {
    type Item = Tensor<T, A>;
    type IntoIter = IntoNodes<T, A>;

    fn into_iter(self) -> Self::IntoIter {
        IntoNodes { iter: self.nodes.into_iter() }
    }
}

/// Consuming iterator over network nodes.
///
/// The iterator item type is [`Tensor<T, A>`].
#[derive(Debug)]
pub struct IntoNodes<T, A> {
    iter: id_set::IntoIter<Tensor<T, A>>,
}

impl<T, A> Iterator for IntoNodes<T, A> {
    type Item = Tensor<T, A>;

    fn next(&mut self) -> Option<Self::Item> { self.iter.next() }
}

impl<T, A> DoubleEndedIterator for IntoNodes<T, A> {
    fn next_back(&mut self) -> Option<Self::Item> { self.iter.next_back() }
}

impl<T, A> ExactSizeIterator for IntoNodes<T, A> {
    fn len(&self) -> usize { self.iter.len() }
}

impl<T, A> std::iter::FusedIterator for IntoNodes<T, A> { }

/// Borrowing iterator over network nodes.
///
/// The iterator item type is `(`[`Id`]`, &'a `[`Tensor<T, A>`]`)`.
#[derive(Clone, Debug)]
pub struct Nodes<'a, T, A> {
    iter: id_set::IterId<'a, Tensor<T, A>>
}

impl<'a, T, A> Iterator for Nodes<'a, T, A> {
    type Item = (Id, &'a Tensor<T, A>);

    fn next(&mut self) -> Option<Self::Item> { self.iter.next() }
}

impl<T, A> DoubleEndedIterator for Nodes<'_, T, A> {
    fn next_back(&mut self) -> Option<Self::Item> { self.iter.next_back() }
}

impl<T, A> ExactSizeIterator for Nodes<'_, T, A> {
    fn len(&self) -> usize { self.iter.len() }
}

impl<T, A> std::iter::FusedIterator for Nodes<'_, T, A> { }

/// Borrowing iterator over indices in a network.
///
/// Indices are guaranteed to be unique.
///
/// The iterator item type is `(&'a T, `[`Wire`]`)`.
#[derive(Clone, Debug)]
pub struct Indices<'a, T> {
    iter: id_set::Iter<'a, (T, Wire)>
}

impl<'a, T> Iterator for Indices<'a, T> {
    type Item = (&'a T, Wire);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(idx, wire)| (idx, *wire))
    }
}

impl<T> DoubleEndedIterator for Indices<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(idx, wire)| (idx, *wire))
    }
}

impl<T> ExactSizeIterator for Indices<'_, T> {
    fn len(&self) -> usize { self.iter.len() }
}

impl<T> std::iter::FusedIterator for Indices<'_, T> { }

impl<'a, T> Indices<'a, T> {
    /// Convert `self` into a [`InternalIndices`], yielding only the remaining
    /// indices that pair two tensors.
    pub fn internal(self) -> InternalIndices<'a, T> {
        InternalIndices { iter: self.iter }
    }

    /// Convert `self` into a [`IndicesBetween`], yielding only the remaining
    /// indices that pair two specific tensors.
    ///
    /// Unlike [`Network::indices_between`], this conversion does not check that
    /// `a` and `b` are valid indices. If not, the remainder of the resulting
    /// iterator will be empty.
    pub fn between(self, a: Id, b: Id) -> IndicesBetween<'a, T> {
        IndicesBetween { a, b, iter: self.iter }
    }
}

/// Borrowing iterator over only the indices in a network that pair two tensors.
///
/// Indices are guaranteed to be unique.
///
/// The iterator item type is `(&'a T, `[`Id`]`, `[`Id`]`)`.
#[derive(Clone, Debug)]
pub struct InternalIndices<'a, T> {
    iter: id_set::Iter<'a, (T, Wire)>
}

impl<'a, T> Iterator for InternalIndices<'a, T> {
    type Item = (&'a T, Id, Id);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(idx, wire)| {
            if let Wire::Paired(a, b) = wire {
                Some((idx, *a, *b))
            } else {
                None
            }
        })
    }
}

impl<T> DoubleEndedIterator for InternalIndices<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .rfind(|(_, wire)| wire.is_paired())
            .map(|(idx, wire)| {
                if let Wire::Paired(a, b) = wire {
                    (idx, *a, *b)
                } else {
                    unreachable!()
                }
            })
    }
}

impl<T> std::iter::FusedIterator for InternalIndices<'_, T> { }

impl<'a, T> InternalIndices<'a, T> {
    /// Convert `self` into a [`Indices`], yielding all remaining indices in the
    /// network.
    pub fn rest(self) -> Indices<'a, T> {
        Indices { iter: self.iter }
    }

    /// Convert `self` into a [`IndicesBetween`], yielding only the remaining
    /// indices that pair two specific tensors.
    ///
    /// Unlike [`Network::indices_between`], this conversion does not check that
    /// `a` and `b` are valid indices. If not, the remainder of the resulting
    /// iterator will be empty.
    pub fn between(self, a: Id, b: Id) -> IndicesBetween<'a, T> {
        IndicesBetween { a, b, iter: self.iter }
    }
}

/// Borrowing iterator over only the indices between two tensors in a network.
///
/// The iterator item type is `&'a T`.
#[derive(Clone, Debug)]
pub struct IndicesBetween<'a, T> {
    a: Id,
    b: Id,
    iter: id_set::Iter<'a, (T, Wire)>,
}

impl<'a, T> Iterator for IndicesBetween<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(idx, wire)| {
            wire.connects(self.a, self.b).then_some(idx)
        })
    }
}

impl<T> DoubleEndedIterator for IndicesBetween<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .rfind(|(_, wire)| wire.connects(self.a, self.b))
            .map(|(idx, _)| idx)
    }
}

impl<T> std::iter::FusedIterator for IndicesBetween<'_, T> { }

impl<'a, T> IndicesBetween<'a, T> {
    /// Convert `self` into a [`Indices`], yielding all remaining indices in the
    /// network.
    pub fn rest(self) -> Indices<'a, T> {
        Indices { iter: self.iter }
    }

    /// Convert `self` into a [`InternalIndices`], yielding all remaining
    /// indices in the network that pair any two tensors.
    pub fn rest_internal(self) -> InternalIndices<'a, T> {
        InternalIndices { iter: self.iter }
    }
}

/// Borrowing iterator over only the tensors in a network that share an index
/// with a given tensor.
///
/// The iterator item type is `(&'a T, `[`Id`]`, &'a `[`Tensor<T, A>`]`)`.
#[derive(Clone, Debug)]
pub struct Neighbors<'a, T, A> {
    network: &'a Network<T, A>,
    target: Id,
    iter: id_set::Iter<'a, (T, Wire)>,
}

impl<'a, T, A> Iterator for Neighbors<'a, T, A> {
    type Item = (&'a T, Id, &'a Tensor<T, A>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(idx, wire)| {
            wire.other_id(self.target)
            .and_then(|id| self.network.get(id).map(|t| (idx, id, t)))
        })
    }
}

impl<T, A> DoubleEndedIterator for Neighbors<'_, T, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .rfind(|(_, wire)| wire.is_paired_with(self.target))
            .and_then(|(idx, wire)| {
                wire.other_id(self.target).map(|id| (idx, id))
            })
            .and_then(|(idx, id)| self.network.get(id).map(|t| (idx, id, t)))
    }
}

impl<T, A> std::iter::FusedIterator for Neighbors<'_, T, A> { }

#[derive(Debug, Error)]
pub enum PoolError {
    /// Returned when attempting to add a tensor contraction to a [`Pool`] where
    /// one of the threads has died unexpectedly.
    #[error("failed to enqueue contractions: dead thread")]
    DeadThread,

    /// Returned when attempting to add a tensor contraction to a [`Pool`] and
    /// the channle input has closed unexpectedly.
    #[error("failed to enqueue contractions: closed sender channel")]
    ClosedSenderChannel,

    /// Returned when attempting to pull a tensor contraction result from a
    /// [`Pool`] and the channel output encounteres some error.
    #[error("failed to receive contraction result: receiver error: {0}")]
    ClosedReceiverChannel(channel::RecvError),

    /// Returned when a thread encounters some error trying to receive an input
    /// tensor contraction.
    #[error("encountered receiver error from within a thread; receiver error: {0}")]
    WorkerReceiverError(channel::RecvError),
}
pub type PoolResult<T> = Result<T, PoolError>;

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum ToWorker<T, A> {
    Stop,
    Work(Tensor<T, A>, Tensor<T, A>),
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum FromWorker<T, A> {
    RecvError(channel::RecvError),
    Output(Tensor<T, A>),
}

/// A simple thread pool to contract pairs of tensors in parallel.
///
/// Workload between threads is roughly balanced by means of a single-producer,
/// multiple-consumer channel. Contracted pairs are returns in the order in
/// which the contraction operations finished. The pool as a whole is meant to
/// be reused ebtween batches of contractions, and is **not** thread-safe.
#[derive(Debug)]
pub struct Pool<T, A> {
    threads: Vec<thread::JoinHandle<()>>,
    workers_in: channel::Sender<ToWorker<T, A>>,
    workers_out: channel::Receiver<FromWorker<T, A>>,
}

impl<T, A> Pool<T, A>
where
    T: Idx + Send + 'static,
    A: ComplexField + Send + Sync + 'static,
{
    /// Create a new thread pool of `nthreads` threads.
    pub fn new(nthreads: usize) -> Self {
        let (tx_in, rx_in) = channel::unbounded();
        let (tx_out, rx_out) = channel::unbounded();
        let mut threads = Vec::with_capacity(nthreads);
        for _ in 0..nthreads {
            let worker_receiver = rx_in.clone();
            let worker_sender = tx_out.clone();
            let th = thread::spawn(move || loop {
                match worker_receiver.recv() {
                    Ok(ToWorker::Stop) => { break; },
                    Ok(ToWorker::Work(a, b)) => {
                        let c = a * b;
                        match worker_sender.send(FromWorker::Output(c)) {
                            Ok(()) => { continue; },
                            Err(err) => { panic!("sender error: {err}"); },
                        }
                    },
                    Err(err) => {
                        match worker_sender.send(FromWorker::RecvError(err)) {
                            Ok(()) => { panic!("receiver error"); },
                            Err(_) => { panic!("sender error: {err}"); },
                        }
                    },
                }
            });
            threads.push(th);
        }
        Self { threads, workers_in: tx_in, workers_out: rx_out }
    }

    /// Create a new thread pool with the number of threads equal to the number
    /// of logical CPU cores available in the current system.
    pub fn new_cpus() -> Self { Self::new(num_cpus::get()) }

    /// Create a new thread pool with the number of threads equal to the number
    /// of physical CPU cores available in the current system.
    pub fn new_physical() -> Self { Self::new(num_cpus::get_physical()) }

    /// Enqueue a batch of contractions to be distributed across all threads.
    ///
    /// This method will block until all enqueued contractions have been
    /// completed.
    pub fn do_contractions<I>(&self, pairs: I) -> PoolResult<Vec<Tensor<T, A>>>
    where I: IntoIterator<Item = (Tensor<T, A>, Tensor<T, A>)>
    {
        if self.threads.iter().any(|th| th.is_finished()) {
            return Err(PoolError::DeadThread);
        }
        let mut count: usize = 0;
        for (t_a, t_b) in pairs.into_iter() {
            match self.workers_in.send(ToWorker::Work(t_a, t_b)) {
                Ok(()) => { count += 1; },
                Err(_) => { return Err(PoolError::ClosedSenderChannel); },
            }
        }
        let mut output = Vec::with_capacity(count);
        for _ in 0..count {
            match self.workers_out.recv() {
                Ok(FromWorker::Output(t_c)) => { output.push(t_c); },
                Ok(FromWorker::RecvError(err)) => {
                    return Err(PoolError::WorkerReceiverError(err));
                },
                Err(err) => {
                    return Err(PoolError::ClosedReceiverChannel(err));
                },
            }
        }
        Ok(output)
    }

}

impl<T, A> Drop for Pool<T, A> {
    fn drop(&mut self) {
        (0..self.threads.len())
            .for_each(|_| { self.workers_in.send(ToWorker::Stop).ok(); });
        self.threads.drain(..)
            .for_each(|th| { th.join().ok(); });
    }
}

