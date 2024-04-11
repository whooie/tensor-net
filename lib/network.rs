use std::{
    hash::Hash,
    ops::{ Deref, DerefMut },
};
use rustc_hash::{ FxHashMap as HashMap };
use thiserror::Error;
use crate::tensor::{ self, Tensor, Idx };

#[derive(Error, Debug)]
pub enum NetworkError {
    /// Returned when attempting to add a tensor to a network in which an
    /// instance of one or more of the tensor's indices already exists and is
    /// paired with another tensor in the network.
    #[error("error in Network::add_tensor: pre-existing index {0}")]
    PreExistingIndex(String),
}
pub type NetworkResult<T> = Result<T, NetworkError>;

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Wire {
    Unpaired(Id),
    Paired(Id, Id),
}

#[derive(Clone, Debug, Default)]
pub struct Network<T, A> {
    nodes: HashMap<Id, Tensor<T, A>>,
    neighbors: HashMap<Id, HashMap<Id, usize>>,
    indices: HashMap<T, Wire>,
    node_id: usize,
}

impl<T, A> Network<T, A>
where T: Idx + Hash
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

    /// Return the number of tensors in the network.
    pub fn count_nodes(&self) -> usize { self.nodes.len() }

    /// Return a reference to a specific tensor in the network, if it exists.
    pub fn get<I>(&self, id: I) -> Option<&Tensor<T, A>>
    where I: Into<Id>
    {
        self.nodes.get(&id.into())
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
    pub fn push(&mut self, tensor: Tensor<T, A>) -> NetworkResult<Id> {
        let id: Id = self.node_id.into();
        let mut neighbors: HashMap<Id, usize> = HashMap::default();
        for idx in tensor.indices() {
            match self.indices.get_mut(idx) {
                Some(Wire::Paired(..)) => {
                    return Err(NetworkError::PreExistingIndex(idx.label()));
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

pub struct Nodes<'a, T, A> {
    iter: std::collections::hash_map::Iter<'a, Id, Tensor<T, A>>
}

impl<'a, T, A> Iterator for Nodes<'a, T, A> {
    type Item = (Id, &'a Tensor<T, A>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(id, tensor)| (*id, tensor))
    }
}

pub struct Indices<'a, T> {
    iter: std::collections::hash_map::Iter<'a, T, Wire>
}

impl<'a, T> Iterator for Indices<'a, T> {
    type Item = (&'a T, Wire);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(idx, wire)| (idx, *wire))
    }
}

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

