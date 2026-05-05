use std::{
    hash::{ Hash, Hasher },
    path::{ Path, PathBuf },
};
use nalgebra as na;
use ndarray::{ self as nd, ShapeBuilder };
use ndarray_npy::NpzWriter;
use num_complex::Complex64 as C64;
use rand::{ Rng, SeedableRng, rngs::StdRng };
use serde::{ Serialize, Deserialize };
use tensor_net::{
    mps::{ MPS, Gamma },
    circuit::{
        Q,
        Uni,
        Meas,
        haar_layer,
        uniform_meas,
        save_cbor,
        load_cbor,
        CircuitError,
        CircuitResult,
    },
};

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
        format!("unis_seed={}_nqubits={}_depth={}_circ{}.cbor",
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoPointManifest {
    seed: u64,
    nqubits: usize,
    depth: usize,
    num_circs: usize,
    p_meas: Vec<f64>,
    dt: usize,
    xx: (usize, usize),
    id: String,
}

impl TwoPointManifest {
    pub fn unis_fname(&self, circ_id: usize) -> String {
        format!("unis2p_seed={}_nqubits={}_depth={}_circ={}.cbor",
            self.seed, self.nqubits, self.depth, circ_id)
    }

    pub fn probe_fname(&self, circ_id: usize) -> String {
        format!("probe2p_seed={}_nqubits={}_depth={}_circ={}.cbor",
            self.seed, self.nqubits, self.depth, circ_id)
    }

    pub fn meas_fname(&self, p: f64, circ_id: usize) -> String {
        format!("meas2p_seed={}_nqubits={}_depth={}_p={:.6}_circ={}.cbor",
            self.seed, self.nqubits, self.depth, p, circ_id)
    }

    fn manifest_fmt(seed: u64, nqubits: usize, depth: usize, id: &str)
        -> String
    {
        format!("manifest2p_seed={}_nqubits={}_depth={}_id={}.cbor",
            seed, nqubits, depth, id)
    }

    fn manifest_fname(&self) -> String {
        Self::manifest_fmt(self.seed, self.nqubits, self.depth, &self.id)
    }

    pub fn new<I>(
        seed: u64,
        nqubits: usize,
        depth: usize,
        num_circs: usize,
        p_meas: I,
        dt: usize,
        xx: (usize, usize),
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
        dt.hash(&mut hasher);
        xx.hash(&mut hasher);
        let id = format!("{:016x}", hasher.finish());
        Self { seed, nqubits, depth, num_circs, p_meas, dt, xx, id }
    }

    pub fn seed(&self) -> u64 { self.seed }

    pub fn nqubits(&self) -> usize { self.nqubits }

    pub fn depth(&self) -> usize { self.depth }

    pub fn num_circs(&self) -> usize { self.num_circs }

    pub fn p_meas(&self) -> &Vec<f64> { &self.p_meas }

    pub fn dt(&self) -> usize { self.dt }

    pub fn xx(&self) -> (usize, usize) { self.xx }

    pub fn id(&self) -> &String { &self.id }

    pub fn gen_save<P>(&self, outdir: P) -> CircuitResult<()>
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

            let probe_unis: Vec<Vec<Uni>> =
                (self.depth .. self.depth + self.dt)
                .map(|t| haar_layer(self.nqubits, t % 2 == 1, &mut rng))
                .collect();
            save_cbor(&probe_unis, outdir.join(self.probe_fname(circ)))?;

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
}


fn gamma_to_ndarray(gamma: &Gamma<C64>) -> nd::Array3<C64> {
    let tens_shape = gamma.dims();
    let mat = gamma.mat();
    let mat_shape = mat.shape();
    // nalgebra matrices are column-major, so we need to ensure `tens` is
    // column-major as well for efficiency                      v
    let mut tens: nd::Array2<C64> = nd::Array2::zeros(mat_shape.f());
    tens.iter_mut().zip(mat.iter())
        .for_each(|(to, from)| { *to = *from; });
    tens.into_shape(tens_shape).unwrap()
}

fn lambda_to_ndarray(lambda: &na::DVector<f64>) -> nd::Array1<f64> {
    lambda.iter().copied().collect()
}

pub fn save_mps<P>(mps: &MPS<Q, C64>, path: P) -> CircuitResult<()>
where P: AsRef<Path>
{
    let outfile =
        std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .append(false)
        .open(path)
        .map_err(CircuitError::IOError)?;
    let mut writer = NpzWriter::new(outfile);
    for (k, gammak) in mps.gamma().iter().enumerate() {
        let gammak_tens = gamma_to_ndarray(gammak);
        writer.add_array(format!("g{k}"), &gammak_tens)
            .expect("npz writer error");
    }
    for (k, lambdak) in mps.svals().iter().enumerate() {
        let lambdak_vec = lambda_to_ndarray(lambdak);
        writer.add_array(format!("l{k}"), &lambdak_vec)
            .expect("npz writer error");
    }
    writer.finish()
        .expect("npz writer error");
    Ok(())
}

pub fn save_mps_bonds<P>(mps: &MPS<Q, C64>, path: P) -> CircuitResult<()>
where P: AsRef<Path>
{
    let outfile =
        std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .append(false)
        .open(path)
        .map_err(CircuitError::IOError)?;
    let mut writer = NpzWriter::new(outfile);
    for (k, lambdak) in mps.svals().iter().enumerate() {
        let lambdak_vec = lambda_to_ndarray(lambdak);
        writer.add_array(format!("l{k}"), &lambdak_vec)
            .expect("npz writer error");
    }
    writer.finish()
        .expect("npz writer error");
    Ok(())
}

