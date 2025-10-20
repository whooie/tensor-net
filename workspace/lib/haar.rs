use std::{
    hash::{ Hash, Hasher },
    path::{ Path, PathBuf },
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use rand::{ Rng, SeedableRng, rngs::StdRng };
use serde::{ Serialize, Deserialize };
use tensor_net::{
    circuit::{
        Q,
        TileQ2,
        Uni,
        Meas,
        haar_layer,
        uniform_meas,
        save_cbor,
        load_cbor,
        CircuitResult,
    },
    gate::haar,
    mps::MPS,
};
use whooie::{ read_npz, write_npz };
use crate::Meas as MeasOld;

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
            TileQ2::new(n, offs)
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
/// Measurement outcomes are returned as a list of all [`MeasOld::Postsel`].
pub fn apply_main_layer<U, M, R>(
    state: &mut MPS<Q, C64>,
    unis: &U,
    meas: &M,
    rng: &mut R,
) -> Vec<MeasOld>
where
    U: AsRef<Vec<(usize, nd::Array2<C64>)>>,
    M: AsRef<Vec<MeasOld>>,
    R: Rng + ?Sized,
{
    unis.as_ref().iter()
        .for_each(|(k, mat)| { state.apply_unitary2(*k, mat).unwrap(); });
    meas.as_ref().iter()
        .map(|m| {
            match m {
                MeasOld::Rand(k) => {
                    let out: bool = state.measure(*k, rng).unwrap() == 1;
                    MeasOld::Postsel(*k, out)
                },
                MeasOld::Postsel(k, out) => {
                    state.measure_postsel(*k, *out as usize);
                    MeasOld::Postsel(*k, *out)
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
) -> (Vec<MeasOld>, Vec<(usize, f64)>)
where
    U: AsRef<Vec<(usize, nd::Array2<C64>)>>,
    M: AsRef<Vec<MeasOld>>,
    R: Rng + ?Sized
{
    unis.as_ref().iter()
        .for_each(|(k, mat)| { state.apply_unitary2(*k, mat).unwrap(); });
    meas.as_ref().iter()
        .map(|m| {
            match m {
                MeasOld::Rand(k) => {
                    let probs = state.probs(*k).unwrap();
                    let out = state.measure(*k, rng).unwrap();
                    (MeasOld::Postsel(*k, out == 1), (*k, probs[out]))
                },
                MeasOld::Postsel(k, out) => {
                    let prob = state.prob(*k, *out as usize).unwrap();
                    state.measure_postsel(*k, *out as usize);
                    (MeasOld::Postsel(*k, *out), (*k, prob))
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
) -> (Vec<MeasOld>, Option<(f64, f64)>)
where
    U: AsRef<Vec<(usize, nd::Array2<C64>)>>,
    M: AsRef<Vec<MeasOld>>,
    R: Rng + ?Sized
{
    unis.as_ref().iter()
        .for_each(|(k, mat)| { state.apply_unitary2(*k, mat).unwrap(); });
    let outs: Vec<MeasOld> =
        meas.as_ref().iter()
        .filter(|m| target.is_none_or(|j| j == m.idx()))
        .map(|m| {
            match m {
                MeasOld::Rand(k) => {
                    let out: bool = state.measure(*k, rng).unwrap() == 1;
                    MeasOld::Postsel(*k, out)
                },
                MeasOld::Postsel(k, out) => {
                    state.measure_postsel(*k, *out as usize);
                    MeasOld::Postsel(*k, *out)
                },
            }
        })
        .collect();
    let probs: Option<(f64, f64)> =
        target.and_then(|k| state.prob(k, 0).zip(state.prob(k, 1)));
    (outs, probs)
}

/******************************************************************************/

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MiptManifest {
    seed: u64,
    nqubits: usize,
    depth: usize,
    num_circs: usize,
    p_meas: Vec<f64>,
    id: String,
}

impl MiptManifest {
    pub fn unis_fname(&self, circ_id: usize) -> String {
        format!("unis_seed={}_nqubits={}_depth={}_circ={}.cbor",
            self.seed, self.nqubits, self.depth, circ_id)
    }

    pub fn meas_fname(&self, p: f64, circ_id: usize) -> String {
        format!("meas_seed={}_nqubits={}_depth={}_p={:.6}_circ={}.cbor",
            self.seed, self.nqubits, self.depth, p, circ_id)
    }

    fn manifest_fmt(seed: u64, nqubits: usize, depth: usize, id: &str)
        -> String
    {
        format!("manifest_seed={}_nqubits={}_depth={}_id={}.cbor",
            seed, nqubits, depth, id)
    }

    pub fn manifest_fname(&self) -> String {
        Self::manifest_fmt(self.seed, self.nqubits, self.depth, &self.id)
    }

    pub fn new<I>(
        seed: u64,
        nqubits: usize,
        depth: usize,
        num_circs: usize,
        p_meas: I,
    ) -> Self
    where I: IntoIterator<Item = f64>
    {
        let p_meas: Vec<f64> = p_meas.into_iter().collect();
        let mut hasher = std::hash::DefaultHasher::new();
        seed.hash(&mut hasher);
        nqubits.hash(&mut hasher);
        depth.hash(&mut hasher);
        num_circs.hash(&mut hasher);
        p_meas.iter().for_each(|p| { p.to_bits().hash(&mut hasher); });
        let id_seed = hasher.finish();
        let mut rng = StdRng::seed_from_u64(id_seed);
        let id: String = format!("{:016x}", rng.gen::<u64>());
        Self { seed, nqubits, depth, num_circs, p_meas, id }
    }

    pub fn seed(&self) -> u64 { self.seed }

    pub fn nqubits(&self) -> usize { self.nqubits }

    pub fn depth(&self) -> usize { self.depth }

    pub fn num_circs(&self) -> usize { self.num_circs }

    pub fn p_meas(&self) -> &Vec<f64> { &self.p_meas }

    pub fn id(&self) -> &String { &self.id }

    pub fn save_all<P>(&self, outdir: P) -> CircuitResult<()>
    where P: AsRef<Path>
    {
        let outdir = PathBuf::from(outdir.as_ref());
        let mut rng = StdRng::seed_from_u64(self.seed);
        for circ in 0 .. self.num_circs {
            let unis: Vec<Vec<Uni>> =
                (0 .. self.depth)
                .map(|t| haar_layer(self.nqubits, t % 2 == 1, &mut rng))
                .collect();
            save_cbor(&unis, outdir.join(self.unis_fname(circ)))?;

            for p in self.p_meas.iter().copied() {
                let meas: Vec<Vec<Meas>> =
                    (0 .. self.depth)
                    .map(|_| uniform_meas(self.nqubits, p, &mut rng))
                    .collect();
                save_cbor(&meas, outdir.join(self.meas_fname(p, circ)))?;
            }
        }
        save_cbor(self, outdir.join(self.manifest_fname()))?;
        Ok(())
    }

    pub fn load<P>(infile: P) -> CircuitResult<Self>
    where P: AsRef<Path>
    {
        load_cbor(infile)
    }

    pub fn load_in<P>(
        seed: u64,
        nqubits: usize,
        depth: usize,
        id: &str,
        indir: P,
    ) -> CircuitResult<Self>
    where P: AsRef<Path>
    {
        let fname = Self::manifest_fmt(seed, nqubits, depth, id);
        let infile = indir.as_ref().join(fname);
        Self::load(infile)
    }
}

