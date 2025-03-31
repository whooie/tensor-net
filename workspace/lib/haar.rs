use std::path::Path;
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::Rng;
use tensor_net::{
    circuit::{ Q, TileQ2 },
    gate::haar,
    mps::MPS,
};
use whooie::{ read_npz, write_npz };
use crate::Meas;

/// A single layer of unitaries.
#[derive(Clone, Debug, PartialEq)]
pub struct UniLayer(Vec<(usize, nd::Array2<C64>)>);

impl UniLayer {
    /// Get a reference to each unitary in the layer with its left operand.
    pub fn get(&self) -> &Vec<(usize, nd::Array2<C64>)> { &self.0 }

    /// Generate a brickwork layer of random (uniform Haar) random unitaries.
    pub fn gen<R>(n: usize, offs: bool, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let data: Vec<(usize, nd::Array2<C64>)> =
            TileQ2::new(offs, n)
            .map(|k| (k, haar(2, rng)))
            .collect();
        Self(data)
    }
}

impl AsRef<Vec<(usize, nd::Array2<C64>)>> for UniLayer {
    fn as_ref(&self) -> &Vec<(usize, nd::Array2<C64>)> { self.get() }
}

/// Write a series of [`UniLayer`]s to a (`.npz`) file.
#[allow(clippy::type_complexity)]
pub fn save_unis<'a, P, I>(path: P, unis: I)
where
    P: AsRef<Path>,
    I: IntoIterator<Item = &'a UniLayer>,
{
    let ((depth, targs), mat_views):
        ((Vec<u32>, Vec<u32>), Vec<nd::ArrayView2<C64>>) =
        unis.into_iter().enumerate()
        .flat_map(|(depth, layer)| {
            layer.0.iter()
            .map(move |(targ, mat)| {
                ((depth as u32, *targ as u32), mat.view())
            })
        })
        .unzip();
    let depth = nd::Array1::from_vec(depth);
    let targs = nd::Array1::from_vec(targs);
    let mats: nd::Array3<C64> = nd::stack(nd::Axis(0), &mat_views).unwrap();
    let path = path.as_ref();
    write_npz!(
        path,
        arrays: { "depth" => &depth, "targs" => &targs, "mats" => &mats }
    );
}

/// Load a series of [`UniLayer`]s from a (`.npz`) file.
pub fn load_unis<P>(path: P) -> Vec<UniLayer>
where P: AsRef<Path>
{
    let path = path.as_ref();
    let (depth, targs, mats):
        (nd::Array1<u32>, nd::Array1<u32>, nd::Array3<C64>) =
        read_npz!(path, arrays: { "depth", "targs", "mats" });
    depth.iter().zip(&targs).zip(mats.outer_iter())
        .chunk_by(|((depth, _), _)| *depth)
        .into_iter()
        .map(|(_, group)| {
            let unis =
                group
                .map(|((_, targ), mat_view)| {
                    (*targ as usize, mat_view.to_owned())
                })
                .collect();
            UniLayer(unis)
        })
        .collect()
}

/// Apply a single pair of unitary and measurement layers to a [`MPS`], with the
/// unitary layer applied first.
///
/// Measurement outcomes are returned as a list of all [`Meas::Postsel`].
pub fn apply_main_layer<U, M, R>(
    state: &mut MPS<Q, C64>,
    unis: &U,
    meas: &M,
    rng: &mut R,
) -> Vec<Meas>
where
    U: AsRef<Vec<(usize, nd::Array2<C64>)>>,
    M: AsRef<Vec<Meas>>,
    R: Rng + ?Sized,
{
    unis.as_ref().iter()
        .for_each(|(k, mat)| { state.apply_unitary2(*k, mat).unwrap(); });
    meas.as_ref().iter()
        .map(|m| {
            match m {
                Meas::Rand(k) => {
                    let out: bool = state.measure(*k, rng).unwrap() == 1;
                    Meas::Postsel(*k, out)
                },
                Meas::Postsel(k, out) => {
                    state.measure_postsel(*k, *out as usize);
                    Meas::Postsel(*k, *out)
                },
            }
        })
        .collect()
}

/// Like [`apply_main_layer`], but returning qubit state probabilities for each
/// measurement alongside outcomes.
pub fn apply_main_layer_probs<U, M, R>(
    state: &mut MPS<Q, C64>,
    unis: &U,
    meas: &M,
    rng: &mut R,
) -> (Vec<Meas>, Vec<(usize, f64)>)
where
    U: AsRef<Vec<(usize, nd::Array2<C64>)>>,
    M: AsRef<Vec<Meas>>,
    R: Rng + ?Sized
{
    unis.as_ref().iter()
        .for_each(|(k, mat)| { state.apply_unitary2(*k, mat).unwrap(); });
    meas.as_ref().iter()
        .map(|m| {
            match m {
                Meas::Rand(k) => {
                    let probs = state.probs(*k).unwrap();
                    let out = state.measure(*k, rng).unwrap();
                    (Meas::Postsel(*k, out == 1), (*k, probs[out]))
                },
                Meas::Postsel(k, out) => {
                    let prob = state.prob(*k, *out as usize).unwrap();
                    state.measure_postsel(*k, *out as usize);
                    (Meas::Postsel(*k, *out), (*k, prob))
                },
            }
        })
        .unzip()
}

/// Apply a single pair of unitary and measurement layers to a [`MPS`], with the
/// unitary layer applied first.
///
/// If `target` is `Some`, then no measurements are applied to the inner qubit
/// index; the single-qubit state probabilities at that site are instead
/// returned alongside measurement outcomes.
pub fn apply_probe_layer<U, M, R>(
    state: &mut MPS<Q, C64>,
    unis: &U,
    meas: &M,
    target: Option<usize>,
    rng: &mut R,
) -> (Vec<Meas>, Option<(f64, f64)>)
where
    U: AsRef<Vec<(usize, nd::Array2<C64>)>>,
    M: AsRef<Vec<Meas>>,
    R: Rng + ?Sized
{
    unis.as_ref().iter()
        .for_each(|(k, mat)| { state.apply_unitary2(*k, mat).unwrap(); });
    let outs: Vec<Meas> =
        meas.as_ref().iter()
        .filter(|m| !target.is_some_and(|j| j == m.idx()))
        .map(|m| {
            match m {
                Meas::Rand(k) => {
                    let out: bool = state.measure(*k, rng).unwrap() == 1;
                    Meas::Postsel(*k, out)
                },
                Meas::Postsel(k, out) => {
                    state.measure_postsel(*k, *out as usize);
                    Meas::Postsel(*k, *out)
                },
            }
        })
        .collect();
    let probs: Option<(f64, f64)> =
        target.and_then(|k| state.prob(k, 0).zip(state.prob(k, 1)));
    (outs, probs)
}

