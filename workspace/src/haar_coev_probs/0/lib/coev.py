from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import *
import cbor2
import numpy as np

def fname_adjust(
    path: Path,
    adj: Optional[str | Callable[[str], str]],
    suffix: Optional[str],
) -> Path:
    suff = (
        "" if suffix is None
        else ("." + suffix) if not suffix.startswith(".")
        else suffix
    )
    if isinstance(adj, type(None)):
        return path.with_suffix(suff)
    elif isinstance(adj, str):
        return path.with_stem(path.stem + adj).with_suffix(suff)
    elif isinstance(adj, Callable):
        return path.with_stem(adj(path.stem)).with_suffix(suff)
    else:
        raise TypeError("invalid string adjustment")

def map_outer[T, U](
    f: Callable[[np.ndarray[T]], U],
    a: np.ndarray[T],
    num_axes: int = 1,
) -> np.ndarray[U]:
    if num_axes == 0:
        raise ValueError("cannot map on zero axes")
    if num_axes == 1:
        return np.array([f(a_k) for a_k in a])
    else:
        shape = a.shape
        to_map = a.reshape((np.prod(shape[:num_axes]), *shape[num_axes:]))
        mapped = np.array([f(a_k) for a_k in to_map])
        reshape = (*shape[:num_axes], *mapped.shape[1:])
        return mapped.reshape(reshape)

# meas_locs :: { meas index, [d, x] }
# probs :: { meas index }
# -> :: { t }
def surprise_t(
    after_depth: int,
    meas_locs: np.ndarray[(..., 2), np.uint8],
    probs: np.ndarray[(...,), float],
) -> np.ndarray[(...,), float]:
    assert meas_locs.shape[0] == probs.shape[0]
    sel = np.where((meas_locs[:, 0] > after_depth) & (probs > 0.0))
    return np.cumsum(-np.log(probs[sel]))

# -> :: { [y-intercept, slope] }
def linfit(y: np.ndarray[(...,), float]) -> np.ndarray[(2,), float]:
    if y.shape[0] < 2:
        raise ValueError("too few measurements")
    return np.polyfit(np.arange(y.shape[0]), y, 1)[::-1]

@dataclass
class Processed:
    seed: int
    size: int
    depth: int
    d0: int
    p_meas: np.ndarray[(...,), float]
    chi: np.ndarray[(...,), int]
    fits: np.ndarray[(..., ..., ..., ...), float] # :: { circ, p, chi, fit }

    def save(self, outfile: Path):
        np.savez(
            str(outfile),
            seed=np.array([self.seed]),
            size=np.array([self.size]),
            depth=np.array([self.depth]),
            d0=np.array([self.d0]),
            p_meas=self.p_meas,
            chi=self.chi,
            fits=self.fits,
        )

    @staticmethod
    def load(infile: Path) -> Self:
        data = np.load(str(infile))
        seed = int(data["seed"][0])
        size = int(data["size"][0])
        depth = int(data["depth"][0])
        d0 = int(data["d0"][0])
        p_meas = data["p_meas"]
        chi = data["chi"]
        fits = data["fits"]
        return Processed(seed, size, depth, d0, p_meas, chi, fits)

    @staticmethod
    def process_set(manifest_path: Path, outdir: Path) -> Processed:
        with manifest_path.open("rb") as infile:
            manifest = cbor2.load(infile)
        seed = manifest["seed"]
        size = manifest["nqubits"]
        depth = manifest["depth"]
        num_circs = manifest["num_circs"]
        p_meas = np.array(manifest["p_meas"])
        genid = manifest["id"]

        outfile = f"avg_fits_seed={seed}_n={size}_depth={depth}.npz"
        cache_file = outdir.joinpath(outfile)
        if cache_file.exists():
            print(f"found cache file '{cache_file}'")
            return Processed.load(cache_file)

        indir = manifest_path.parent
        d0 = 4 * size
        chi_check: Optional[np.ndarray[(...,), int]] = None

        datafile = lambda circ, p: indir.joinpath(f"c={circ}/p={p:.6f}.npz")
        fits = list()
        for (circ, p) in product(range(num_circs), p_meas):
            maybe_fits = process_single(datafile(circ, p), d0)
            if maybe_fits is not None:
                (chi, fits_by_chi) = maybe_fits
                if chi_check is None:
                    chi_check = chi
                else:
                    assert np.all(chi == chi_check)
                fits.append(fits_by_chi)
        fits = (
            np.array(fits)
            .reshape(num_circs, p_meas.shape[0], chi_check.shape[0], 2)
        )
        processed = Processed(seed, size, depth, d0, p_meas, chi_check, fits)
        processed.save(cache_file)
        return processed

# -> :: ({ chi }, { chi, linfit param })
def process_single(
    infile: Path,
    d0: int,
) -> Optional[tuple[np.ndarray[('x',), int], np.ndarray[('x', 2), float]]]:
    print(infile)
    data = np.load(str(infile))
    chi = data["chi"]
    meas_locs = data["meas_locs"]
    prob = data["prob_data"] # :: { run, chi, meas index }

    # rotate arrays so that chi == 0 (inf) is last
    # assume that chi == 0 exists uniquely
    assert np.sum(chi == 0) == 1
    rot = len(chi) - np.argmin(chi) - 1
    chi = np.roll(chi, rot)
    prob = np.roll(prob, rot, axis=1)

    if np.sum(meas_locs[:, 0] > d0) < 2:
        print("discarding due to too few measurements")
        return None

    k_q = np.where(chi == 0)[0][0]
    k_c = np.where(chi != 0)[0]

    fits = np.array([
        [linfit(surprise_t(d0, meas_locs, prob_rx)) for prob_rx in prob_r]
        for prob_r in prob
    ]).mean(axis=0) # :: { chi, linfit param }
    return (chi, fits)

