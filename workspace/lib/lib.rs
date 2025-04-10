use std::{ ops::Range, path::Path };
use itertools::Itertools;
use ndarray as nd;
use rand::Rng;
use whooie::{ read_npz, write_npz };

/// Return the range of qubits in the reverse light cone of a target measurement
/// on qubit index `x` (out of `n` total qubits) occurring `dt` layers from
/// layer `t`.
pub fn rev_cone_range(n: usize, x: usize, t: usize, dt: usize) -> Range<usize> {
    if (t + x) % 2 == 0 {
        let start = x.saturating_sub(2 * (dt / 2));
        let end = (x + 2 * ((dt + 1) / 2)).min(n);
        start .. end
    } else {
        let start = x.saturating_sub(2 * ((dt + 1) / 2)) + 1;
        let end = (x + 2 * (dt / 2) + 1).min(n);
        start .. end
    }
}

/// A single qubit measurement operation, either naive or post-selected.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Meas {
    Rand(usize),
    Postsel(usize, bool),
}

impl Meas {
    /// Return the target qubit index.
    pub fn idx(&self) -> usize {
        match self {
            Self::Rand(k) => *k,
            Self::Postsel(k, _) => *k,
        }
    }
}

/// A single layer of measurement locations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasLayer(Vec<Meas>);

impl MeasLayer {
    /// Get a reference to all measurement operations in the layer.
    pub fn get(&self) -> &Vec<Meas> { &self.0 }

    /// Generate a layer of (`Rand`) measurements whose locations are chosen
    /// uniformly with probability `o`.
    pub fn gen<R>(n: usize, p: f64, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let data: Vec<Meas> =
            (0..n)
            .filter(|_| rng.gen::<f64>() < p)
            .map(Meas::Rand)
            .collect();
        Self(data)
    }
}

/// Write a series of generated measurement locations to a (`.npz`) file.
pub fn save_meas<'a, P, I, J>(path: P, meas: I)
where
    P: AsRef<Path>,
    I: IntoIterator<Item = (&'a f64, J)>,
    J: IntoIterator<Item = &'a MeasLayer>,
{
    let mut pvals: Vec<f64> = Vec::new();
    let mut layer_lens: Vec<u32> = Vec::new();
    let mut targs: Vec<u32> = Vec::new();
    let mut ops: Vec<i32> = Vec::new();
    for (pval, layers) in meas.into_iter() {
        for layer in layers.into_iter() {
            layer.0.iter()
                .for_each(|measop| {
                    match measop {
                        Meas::Rand(targ) => {
                            targs.push(*targ as u32);
                            ops.push(-1);
                        },
                        Meas::Postsel(targ, res) => {
                            targs.push(*targ as u32);
                            ops.push(*res as i32);
                        },
                    }
                });
            pvals.push(*pval);
            layer_lens.push(layer.0.len() as u32);
        }
    }
    let pvals = nd::Array1::from_vec(pvals);
    let layer_lens = nd::Array1::from_vec(layer_lens);
    let targs = nd::Array1::from_vec(targs);
    let ops = nd::Array1::from_vec(ops);
    let path = path.as_ref();
    write_npz!(
        path,
        arrays: {
            "pvals" => &pvals,
            "layer_lens" => &layer_lens,
            "targs" => &targs,
            "ops" => &ops,
        }
    );
}

/// Load a series of generated measurement locations from a (`.npz`) file.
pub fn load_meas<P>(path: P) -> (nd::Array1<f64>, Vec<Vec<MeasLayer>>)
where P: AsRef<Path>
{
    let path = path.as_ref();
    let (pvals, layer_lens, targs, ops):
        (nd::Array1<f64>, nd::Array1<u32>, nd::Array1<u32>, nd::Array1<i32>) =
        read_npz!(path, arrays: { "pvals", "layer_lens", "targs", "ops" });
    let mut p: Vec<f64> = Vec::new();
    let mut meas: Vec<Vec<MeasLayer>> = Vec::new();
    let mut layers: Vec<MeasLayer> = Vec::new();
    let mut layer: Vec<Meas> = Vec::new();
    let p_iter = pvals.iter().zip(&layer_lens).chunk_by(|(pval, _)| *pval);
    let mut op_iter = targs.into_iter().zip(ops);
    for (pval, pgroup) in p_iter.into_iter() {
        p.push(*pval);
        for (_, layer_len) in pgroup {
            for _ in 0..*layer_len {
                if let Some((targ, op)) = op_iter.next() {
                    match op {
                        0 => {
                            layer.push(Meas::Postsel(targ as usize, false));
                        },
                        1 => {
                            layer.push(Meas::Postsel(targ as usize, true));
                        },
                        _ => {
                            layer.push(Meas::Rand(targ as usize));
                        },
                    }
                } else {
                    panic!("operations list ended unexpectedly");
                }
            }
            layers.push(MeasLayer(std::mem::take(&mut layer)));
        }
        meas.push(std::mem::take(&mut layers));
    }
    let p = nd::Array1::from_vec(p);
    (p, meas)
}

impl AsRef<Vec<Meas>> for MeasLayer {
    fn as_ref(&self) -> &Vec<Meas> { self.get() }
}

// pub mod clifford;
// pub mod haar;

// pub mod systems;

