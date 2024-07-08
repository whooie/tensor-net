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
//!     network3::{ Network, Pool },
//!     tensor3::{ Idx, Tensor },
//! };
//!
//! // our index type (see the Idx trait for static versus dynamic index types)
//! #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
//! // [[1, 1, 1, 1],
//! //  [1, 1, 1, 1],
//! //  [1, 1, 1, 1]] { A, B }
//!
//! let b = Tensor::new([Index::B, Index::C], |_| 2.0).unwrap();
//! println!("{}", b);
//! // [[2, 2, 2, 2, 2],
//! //  [2, 2, 2, 2, 2],
//! //  [2, 2, 2, 2, 2],
//! //  [2, 2, 2, 2, 2]] { B, C }
//!
//! let c = Tensor::new([Index::A, Index::C], |_| 3.0).unwrap();
//! // [[3, 3, 3, 3, 3],
//! //  [3, 3, 3, 3, 3],
//! //  [3, 3, 3, 3, 3]] { A, C }
//!
//! let d = Tensor::new([Index::D], |_| 4.0).unwrap();
//! // [4, 4] { D }
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
//! // [1440, 1440] { D }
//! ```

use std::{
    hash::Hash,
    ops::{ Deref, DerefMut },
    thread,
};
use crossbeam::channel;
use rustc_hash::{ FxHashMap as HashMap };
use thiserror::Error;
use crate::tensor3::{ self as tensor, Tensor, Idx, Elem, };

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

    /// Returned by anything involving an operation on the level of individual
    /// tensors.
    #[error("tensor error: {0}")]
    TensorError(#[from] tensor::TensorError),

    /// Returned by anything involving a [`Pool`].
    #[error("contractor pool error: {0}")]
    PoolError(#[from] PoolError),
}
use NetworkError::*;
pub type NetworkResult<T> = Result<T, NetworkError>;

fn fst<T, U>(pair: (T, U)) -> T { pair.0 }

fn snd<T, U>(pair: (T, U)) -> U { pair.1 }

macro_rules! isomorphism {
    (
        $docstring:literal,
        $name:ident ($iso_to:ident),
        derive: { $($derive:ident),* $(,)? },
        from: { $($from:ident),* $(,)? } $(,)?
    ) => {
        #[doc = $docstring]
        #[derive($($derive),*)]
        pub struct $name(pub $iso_to);

        impl From<$iso_to> for $name {
            fn from(x: $iso_to) -> Self { Self(x) }
        }

        impl From<$name> for $iso_to {
            fn from(x: $name) -> Self { x.0 }
        }

        impl AsRef<$iso_to> for $name {
            fn as_ref(&self) -> &$iso_to { &self.0 }
        }

        impl AsMut<$iso_to> for $name {
            fn as_mut(&mut self) -> &mut $iso_to { &mut self.0 }
        }

        impl Deref for $name {
            type Target = $iso_to;

            fn deref(&self) -> &Self::Target { &self.0 }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
        }

        $(
            impl From<$from> for $name {
                fn from(x: $from) -> Self { Self(x.into()) }
            }
        )*
    }
}

macro_rules! copy_isomorphism {
    (
        $name:ident ($iso_to:ident),
        from: { $($from:ident),* $(,)? } $(,)?
    ) => {
        impl From<&$iso_to> for $name {
            fn from(x: &$iso_to) -> Self { Self(*x) }
        }

        impl From<&$name> for $name {
            fn from(x: &$name) -> Self { *x }
        }

        impl From<&$name> for $iso_to {
            fn from(x: &$name) -> Self { x.0 }
        }

        $(
            impl From<&$from> for $name {
                fn from(x: &$from) -> Self { Self((*x).into()) }
            }
        )*
    }
}

isomorphism!(
    "Sugared `usize` representing the ID of a single tensor.",
    Id (usize),
    derive: { Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash },
    from: { u8, u16 },
);
copy_isomorphism!(
    Id (usize),
    from: { u8, u16 },
);

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
/// `T`. See [`Idx`] for info on dynamic versus static networks.
#[derive(Clone, Debug, Default)]
pub struct Network<T, A> {
    nodes: HashMap<Id, Tensor<T, A>>,
    neighbors: HashMap<Id, HashMap<Id, usize>>,
    indices: HashMap<T, Wire>,
    node_id: usize,
}

impl<T, A> Network<T, A>
where
    T: Idx,
    A: Elem,
{
    /// Create a new, empty network.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::default(),
            neighbors: HashMap::default(),
            indices: HashMap::default(),
            node_id: 0,
        }
    }

    /// Create a new network from an iterator by repeatedly
    /// [pushing][Self::push] new tensors onto an initially empty network.
    ///
    /// Note that [node IDs][Id] count from zero and no nodes will be removed
    /// during this process, so the IDs 0, ..., n - 1 will correspond uniquely
    /// to the first n nodes in the iterator, in order.
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

    /// Return the number of tensors in the network.
    pub fn count_nodes(&self) -> usize { self.nodes.len() }

    /// Return a reference to a specific tensor in the network, if it exists.
    pub fn get<I>(&self, id: I) -> Option<&Tensor<T, A>>
    where I: Into<Id>
    {
        self.nodes.get(&id.into())
    }

    /// Return a mutable reference to a specific tensor in the network, if it
    /// exists.
    pub fn get_mut<I>(&mut self, id: I) -> Option<&mut Tensor<T, A>>
    where I: Into<Id>
    {
        self.nodes.get_mut(&id.into())
    }

    /// Return the total number of wires attached to a given tensor, equal to
    /// its rank, if it exists.
    pub fn degree<I>(&self, id: I) -> Option<usize>
    where I: Into<Id>
    {
        self.nodes.get(&id.into()).map(|tensor| tensor.rank())
    }

    /// Return the total number of wires from a given tensor that go to another
    /// tensor in the network, if it exists.
    ///
    /// This quantity is at most the degree of the tensor.
    pub fn internal_degree<I>(&self, id: I) -> Option<usize>
    where I: Into<Id>
    {
        self.neighbors.get(&id.into())
            .map(|neighbors| neighbors.values().copied().sum())
    }

    /// Return an iterator over all tensors in the network, visited in an
    /// arbitrary order.
    ///
    /// The iterator item type is `(`[`Id`]`, &`[`Tensor`]`)`.
    pub fn nodes(&self) -> Nodes<'_, T, A> {
        Nodes { iter: self.nodes.iter() }
    }

    /// Return an iterator over all indices in the network, visited in an
    /// arbitrary order.
    ///
    /// The iterator item type is `(&T, `[`Wire`]`)`.
    pub fn indices(&self) -> Indices<'_, T> {
        Indices { iter: self.indices.iter() }
    }

    /// Apply an updating function to all indices in the network.
    ///
    /// # Safety
    /// Modifying the network's indices in such a way that any bond dimension
    /// changes or an index is no longer identified with a particular wire
    /// between tensors will cause unrecoverable errors in network contraction.
    /// A call to this function is safe iff each bond dimension is invariant in
    /// the update and every pair of tensors sharing an index before the update
    /// still share an index after.
    pub unsafe fn update_indices<F>(&mut self, update: F) -> &mut Self
    where F: Fn(&mut T)
    {
        self.nodes.values_mut()
            .for_each(|tensor| {
                tensor.indices_mut()
                    .for_each(|idx| { update(idx); })
            });
        // ownership rules mean we have to allocate
        let indices_tmp: Vec<(T, Wire)> = self.indices.drain().collect();
        indices_tmp.into_iter()
            .for_each(|(mut idx, wire)| { 
                update(&mut idx);
                self.indices.insert(idx, wire);
            });
        self
    }

    /// Apply a mapping function to all indices in the network.
    ///
    /// # Safety
    /// Modifying the network's indices in such a way that any bond dimension
    /// changes or an index is no longer identified with a particular wire
    /// between tensors will cause unrecoverable errors in network contraction.
    /// A call to this function is safe iff each bond dimension is invariant in
    /// the update and every pair of tensors sharing an index before the update
    /// still share an index after.
    pub unsafe fn map_indices<F, U>(self, map: F)
        -> NetworkResult<Network<U, A>>
    where
        F: Fn(T) -> U,
        U: Idx,
    {
        Network::from_nodes(
            self.into_iter()
                .map(|node| node.map_indices(&map))
        )
    }

    /// Return an iterator over all internal indices (connected to tensors on
    /// both ends) in the network, visited in an arbitrary order.
    ///
    /// The iterator item type is `(&T, (`[`Id`]`, `[`Id`]`))`.
    pub fn internal_indices(&self) -> InternalIndices<'_, T> {
        InternalIndices { iter: self.indices() }
    }

    /// Return an iterator over all indices connected to a given tensor, if it
    /// exists.
    ///
    /// The iterator item type is `&T`.
    pub fn indices_of<I>(&self, id: I) -> Option<tensor::Indices<'_, T>>
    where I: Into<Id>
    {
        self.nodes.get(&id.into()).map(|tensor| tensor.indices())
    }

    /// Return an iterator over all indices connecting two tensors, visited an
    /// an arbitrary order, if they both exist.
    ///
    /// The iterator item type is `&T`.
    pub fn indices_between<I, J>(&self, a: I, b: J)
        -> Option<IndicesBetween<'_, T>>
    where
        I: Into<Id>,
        J: Into<Id>,
    {
        let a = a.into();
        let b = b.into();
        self.nodes.get(&a)
            .zip(self.nodes.get(&b))
            .map(|_| IndicesBetween { a, b, iter: self.indices.iter() })
    }

    /// Return an iterator over all neighbors of a given tensor, visited in an
    /// arbitrary order, if it exists.
    ///
    /// The iterator item type `(`[`Id`]`, &`[`Tensor`]`)`.
    pub fn neighbors_of<I>(&self, id: I) -> Option<Neighbors<'_, T, A>>
    where I: Into<Id>
    {
        self.neighbors.get(&id.into())
            .map(|neighbors| {
                Neighbors { network: self, iter: neighbors.keys() }
            })
    }

    /// Add a new tensor to the network and return its ID.
    ///
    /// Fails if any of the new tensor's indices already exist in the network as
    /// a bond/ between two other tensors.
    pub fn push(&mut self, tensor: Tensor<T, A>) -> NetworkResult<Id> {
        let id: Id = self.node_id.into();
        let mut neighbors: HashMap<Id, usize> = HashMap::default();
        for idx in tensor.indices() {
            match self.indices.get_mut(idx) {
                Some(Wire::Paired(..)) => {
                    return Err(PreExistingIndex(idx.label()));
                },
                Some(status) => {
                    let Wire::Unpaired(neighbor)
                        = *status else { unreachable!() };
                    *status = Wire::Paired(id, neighbor);
                    self.neighbors.entry(neighbor)
                        .and_modify(|of_neighbor| {
                            of_neighbor.entry(id)
                                .and_modify(|n_bonds| { *n_bonds += 1; })
                                .or_insert(1);
                        })
                        .or_default();
                    neighbors.entry(neighbor)
                        .and_modify(|n_bonds| { *n_bonds += 1; })
                        .or_insert(1);
                },
                None => {
                    self.indices.insert(idx.clone(), Wire::Unpaired(id));
                },
            }
        }
        self.nodes.insert(id, tensor);
        self.neighbors.insert(id, neighbors);
        self.node_id += 1;
        Ok(id)
    }

    /// Remove a tensor from the network, returning the tensor itself and a set
    /// of its former neighbors, if it exists.
    pub fn remove<I>(&mut self, id: I)
        -> Option<(Tensor<T, A>, HashMap<Id, usize>)>
    where I: Into<Id>
    {
        let id = id.into();
        if self.nodes.contains_key(&id) && self.neighbors.contains_key(&id) {
            let tensor = self.nodes.remove(&id).unwrap();
            for idx in tensor.indices() {
                match self.indices.get_mut(idx) {
                    Some(Wire::Unpaired(_)) => { self.indices.remove(idx); },
                    Some(status) => {
                        let Wire::Paired(a, b)
                            = *status else { unreachable!() };
                        let remaining = if a == id { b } else { a };
                        *status = Wire::Unpaired(remaining);
                    },
                    None => { unreachable!() },
                }
            }
            let neighbors = self.neighbors.remove(&id).unwrap();
            for (neighbor, _) in neighbors.iter() {
                if let Some(of_neighbor) = self.neighbors.get_mut(neighbor) {
                    of_neighbor.remove(&id);
                }
            }
            Some((tensor, neighbors))
        } else {
            None
        }
    }
}

impl<T, A> Network<T, A>
where
    T: Idx,
    A: Elem,
{
    /// Simple greedy algorithm to find the next contraction step, optimized
    /// over only a single contraction.
    fn find_contraction(&self) -> Option<(Id, Id)> {
        fn costf<T, A>(net: &Network<T, A>, id_a: &Id, id_b: &Id) -> usize
        where
            T: Idx,
            A: Elem,
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
            .min_by(|(_, (id_al, id_bl)), (_, (id_ar, id_br))| {
                costf(self, id_al, id_bl)
                    .cmp(&costf(self, id_ar, id_br))
            })
            .map(snd)
    }

    /// Contract a single pair of tensors named by the index pair `(a, b)`.
    pub fn contract_single<I, J>(&mut self, a: I, b: J) -> NetworkResult<Id>
    where
        I: Into<Id>,
        J: Into<Id>,
    {
        let a = a.into();
        let b = b.into();
        let (t_a, _) = self.remove(a).ok_or(ContractMissingId(a.0))?;
        let (t_b, _) = self.remove(b).ok_or(ContractMissingId(b.0))?;
        let t_c = t_a.contract(t_b)?;
        self.push(t_c)
    }

    /// Like [`Self::contract_single`], but with the contraction named by the index
    /// instead of the tensors.
    ///
    /// Note that contractions between tensors are automatically performed over
    /// all shared indices, so there's no need to call this method for each of
    /// the indices shared by a pair of tensors.
    pub fn contract_single_index(&mut self, idx: &T) -> NetworkResult<Id> {
        match self.indices.get(idx) {
            Some(Wire::Paired(a, b)) => self.contract_single(*a, *b),
            Some(Wire::Unpaired(_)) => Err(ContractUnpaired(idx.label())),
            None => Err(ContractMissing(idx.label())),
        }
    }

    fn do_contract(&mut self) -> NetworkResult<&mut Self> {
        while let Some((a, b)) = self.find_contraction() {
            self.contract_single(a, b)?;
        }
        Ok(self)
    }

    /// Contract the entire network into a single output tensor.
    pub fn contract(mut self) -> NetworkResult<Tensor<T, A>> {
        self.do_contract()?;
        if self.count_nodes() > 1 {
            let mut remaining_nodes = self.into_iter();
            let acc = remaining_nodes.next().unwrap();
            remaining_nodes.try_fold(acc, |a, t| a.tensor_prod(t))
                .map_err(NetworkError::from)
        } else {
            Ok(self.into_iter().next().unwrap())
        }
    }

    /// Contract the network completely but don't consume `self`, instead
    /// removing and returning the final result.
    ///
    /// The network will be left empty but with memory still allocated after
    /// calling this method.
    pub fn contract_remove(&mut self) -> NetworkResult<Tensor<T, A>> {
        self.do_contract()?;
        if self.count_nodes() > 1 {
            self.neighbors.clear();
            self.indices.clear();
            let mut remaining_nodes = self.nodes.drain();
            let acc = remaining_nodes.next().unwrap().1;
            remaining_nodes.try_fold(acc, |a, (_, t)| a.tensor_prod(t))
                .map_err(NetworkError::from)
        } else {
            Ok(self.nodes.drain().next().unwrap().1)
        }
    }

    /// Contract all paired indices, leaving the result as a network.
    ///
    /// The network may be left with multiple tensors remaining if they have no
    /// common indices.
    pub fn contract_network(&mut self) -> NetworkResult<&mut Self> {
        self.do_contract()?;
        Ok(self)
    }
}

impl<T, A> Network<T, A>
where
    T: Idx + Send + 'static,
    A: Elem + Send + Sync,
{
    fn do_contract_par(&mut self, pool: &Pool<T, A>)
        -> NetworkResult<&mut Self>
    {
        let mut contractions: Vec<(Tensor<T, A>, Tensor<T, A>)>
            = Vec::with_capacity(self.count_nodes() / 2);
        let mut results: Vec<Tensor<T, A>>;
        while self.internal_indices().count() > 0 {
            while let Some((a, b)) = self.find_contraction() {
                let (t_a, _) = self.remove(a).unwrap();
                let (t_b, _) = self.remove(b).unwrap();
                contractions.push((t_a, t_b));
            }
            results = pool.do_contractions(contractions.drain(..))?;
            results.into_iter()
                .try_for_each(|t| self.push(t).map(|_| ()))?;
        }
        Ok(self)
    }

    /// Like [`Self::contract`], but using a thread pool.
    pub fn contract_par(mut self, pool: &Pool<T, A>)
        -> NetworkResult<Tensor<T, A>>
    {
        self.do_contract_par(pool)?;
        if self.count_nodes() > 1 {
            let mut remaining_nodes = self.into_iter();
            let acc = remaining_nodes.next().unwrap();
            remaining_nodes.try_fold(acc, |a, t| a.tensor_prod(t))
                .map_err(NetworkError::from)
        } else {
            Ok(self.into_iter().next().unwrap())
        }
    }

    /// Like [`Self::contract_remove`], but using a thread pool.
    pub fn contract_remove_par(&mut self, pool: &Pool<T, A>)
        -> NetworkResult<Tensor<T, A>>
    {
        self.do_contract_par(pool)?;
        if self.count_nodes() > 1 {
            self.neighbors.clear();
            self.indices.clear();
            let mut remaining_nodes = self.nodes.drain();
            let acc = remaining_nodes.next().unwrap().1;
            remaining_nodes.try_fold(acc, |a, (_, t)| a.tensor_prod(t))
                .map_err(NetworkError::from)
        } else {
            self.neighbors.clear();
            self.indices.clear();
            Ok(self.nodes.drain().next().unwrap().1)
        }
    }

    /// Like [`Self::contract_network`], but using a thread pool.
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
        IntoNodes { iter: self.nodes.into_values() }
    }
}

/// Iterator over network nodes, created by [`Network::into_iter`].
///
/// The iterator item type is [`Tensor<T, A>`].
pub struct IntoNodes<T, A> {
    iter: std::collections::hash_map::IntoValues<Id, Tensor<T, A>>
}

impl<T, A> Iterator for IntoNodes<T, A> {
    type Item = Tensor<T, A>;

    fn next(&mut self) -> Option<Self::Item> { self.iter.next() }
}

/// Iterator over network nodes, created by [`Network::nodes`].
///
/// The iterator item type is `(`[`Id`]`, &'a `[`Tensor<T, A>`]`)`.
pub struct Nodes<'a, T, A> {
    iter: std::collections::hash_map::Iter<'a, Id, Tensor<T, A>>
}

impl<'a, T, A> Iterator for Nodes<'a, T, A> {
    type Item = (Id, &'a Tensor<T, A>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(id, tensor)| (*id, tensor))
    }
}

/// Iterator over indices in a network, created by [`Network::indices`].
///
/// The iterator item type is `(&'a T, `[`Wire`]`)`.
pub struct Indices<'a, T> {
    iter: std::collections::hash_map::Iter<'a, T, Wire>
}

impl<'a, T> Iterator for Indices<'a, T> {
    type Item = (&'a T, Wire);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(idx, wire)| (idx, *wire))
    }
}

/// Iterator over only the indices in a network that pair two tensors, created
/// by [`Network::internal_indices`].
///
/// The iterator item type is `(&'a T, (`[`Id`]`, `[`Id`]`))`.
pub struct InternalIndices<'a, T> {
    iter: Indices<'a, T>
}

impl<'a, T> Iterator for InternalIndices<'a, T> {
    type Item = (&'a T, (Id, Id));

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(idx, wire)| {
            if let Wire::Paired(a, b) = wire {
                Some((idx, (a, b)))
            } else {
                None
            }
        })
    }
}

/// Iterator over only the indices between two tensors in a network, created by
/// [`Network::indices_between`].
///
/// The iterator item type is `&'a T`.
pub struct IndicesBetween<'a, T> {
    a: Id,
    b: Id,
    iter: std::collections::hash_map::Iter<'a, T, Wire>,
}

impl<'a, T> Iterator for IndicesBetween<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(idx, wire)| {
            if matches!(
                wire, Wire::Paired(a, b) if a == &self.a && b == &self.b)
            {
                Some(idx)
            } else {
                None
            }
        })
    }
}

/// Iterator over only the tensors in a network that share an index with a given
/// tensor, created by [`Network::neighbors_of`].
///
/// The iterator item type is `(`[`Id`]`, &'a `[`Tensor<T, A>`]`)`.
pub struct Neighbors<'a, T, A> {
    network: &'a Network<T, A>,
    iter: std::collections::hash_map::Keys<'a, Id, usize>,
}

impl<'a, T, A> Iterator for Neighbors<'a, T, A> {
    type Item = (Id, &'a Tensor<T, A>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|id| (*id, &self.network.nodes[id]))
    }
}

#[derive(Debug, Error)]
pub enum PoolError {
    #[error("failed to enqueue contractions: dead thread")]
    DeadThread,

    #[error("failed to enqueue contractions: closed sender channel")]
    ClosedSenderChannel,

    #[error("failed to receive contraction result: receiver error: {0}")]
    ClosedReceiverChannel(channel::RecvError),

    #[error("encountered receiver error from within a thread: receiver error: {0}")]
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
/// Workload between threads is automatically balanced by means of a
/// single-producer, multiple-consumer channel. Contracted pairs are returned in
/// the order in which the contraction operations finished. The pool as a whole
/// is meant to be reused between batches of contractions, and is **not**
/// thread-safe.
#[derive(Debug)]
pub struct Pool<T, A> {
    threads: Vec<thread::JoinHandle<()>>,
    workers_in: channel::Sender<ToWorker<T, A>>,
    workers_out: channel::Receiver<FromWorker<T, A>>,
}

impl<T, A> Pool<T, A>
where
    T: Idx + Send + 'static,
    A: Elem + Send + Sync,
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
                        let c = a.multiply(b);
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
    pub fn do_contractions<I>(&self, pairs: I)
        -> PoolResult<Vec<Tensor<T, A>>>
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
                }
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

