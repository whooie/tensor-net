from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import *

import cbor2 as cbor
import numpy as np
import numpy.linalg as la
try:
    import whooie.pyplotdefs as pd
except ModuleNotFoundError:
    import lablib.plotting.pyplotdefs as pd

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
        raise Exception()

def map_outer[T, U](
    f: Callable[[np.ndarray[T]], U],
    a: np.ndarray[T],
    num_axes: int = 1,
) -> np.ndarray[U]:
    assert num_axes != 0, "cannot map on zero axes"
    if num_axes == 1:
        return np.array([f(a_k) for a_k in a])
    else:
        shape = a.shape
        to_map = a.reshape((np.prod(shape[:num_axes]), *shape[num_axes:]))
        mapped = np.array([f(a_k) for a_k in to_map])
        reshape = (*shape[:num_axes], *mapped.shape[1:])
        return mapped.reshape(reshape)

# tx_coords :: { meas index, [t, x] }
# probs :: { meas index }
# -> :: { t' }
def surprise_t(
    after: int,
    tx_coords: np.ndarray[np.uint8, 2],
    probs: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    assert tx_coords.shape[0] == probs.shape[0]
    sel = np.where((tx_coords[:, 0] > after) & (probs > 0.0))
    return np.cumsum(-np.log(probs[sel]))

# -> :: { [y-int, slope] }
def linfit(y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    x = np.arange(y.shape[0])
    A = np.array([np.ones(y.shape), x]).T
    try:
        return la.solve(A.T @ A, A.T @ y)
    except:
        print(y.shape)
        print(A.T @ A)
        raise Exception("singular matrix, probably due to small data set")

@dataclass
class Processed:
    seed: int
    size: int
    depth: int
    d0: int
    p_meas: np.ndarray[float, 1]
    chi: np.ndarray[int, 1]
    fits: np.ndarray[float, 4] # :: { circ, p_meas, chi, linfit param }

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
            manifest = cbor.load(infile)
        seed = manifest["seed"]
        size = manifest["nqubits"]
        depth = manifest["depth"]
        num_circs = manifest["num_circs"]
        p_meas = np.array(manifest["p_meas"])
        genid = manifest["id"]

        cache_file = outdir.joinpath(outfile_fmt(seed, size, depth))
        if cache_file.exists():
            print(f"found cache file '{cache_file}'")
            processed = Processed.load(cache_file)
            return processed

        indir = manifest_path.parent
        infiles = os.listdir(indir)
        d0 = 4 * size
        chi0 = None

        fits = list()
        for circ in range(num_circs):
            for p in p_meas:
                pat = infile_pat(seed, size, depth, p, circ)
                files_of = [
                    indir.joinpath(fname) for fname in infiles
                    if pat.match(fname)
                ]
                maybe_fits = process_single(files_of, d0)
                if maybe_fits is not None:
                    (chik, fitsk) = maybe_fits
                    if chi0 is None:
                        chi0 = chik
                    else:
                        assert np.all(chi0 == chik)
                    fits.append(fitsk)
                print()
        fits = (
            np.array(fits)
            .reshape(num_circs, p_meas.shape[0], chi0.shape[0], 2)
        )
        processed = Processed(seed, size, depth, d0, p_meas, chi0, fits)
        processed.save(cache_file)
        return processed

def infile_pat(
    seed: int,
    size: int,
    depth: int,
    p: float,
    circ: int,
) -> re.Pattern:
    return re.compile(
        f"haar_coev_probs"
        f"_seed={seed}"
        f"_nqubits={size}"
        f"_depth={depth}"
        f"_p={p:.6f}"
        f"_circ={circ}"
        f"_outid=([0-9a-f]+)"
        f".npz"
    )

def outfile_fmt(
    seed: int,
    size: int,
    depth: int,
) -> str:
    return f"haar_coev_probs_nd_seed={seed}_n={size}_depth={depth}_fitavg.npz"

# -> :: { chi, linfit param }
def process_single(
    infiles: list[Path],
    d0: int,
) -> Optional[tuple[np.ndarray[int, 1], np.ndarray[float, 2]]]:
    infiles = [infile for infile in infiles if infile.exists()]
    if len(infiles) == 0:
        return None
    for infile in infiles:
        print(infile)

    data = [np.load(str(infile)) for infile in infiles]
    size = int(data[0]["nqubits"][0])
    assert all(d["nqubits"][0] == size for d in data)
    chi = data[0]["chi"]
    assert all(np.all(d["chi"] == chi) for d in data)
    meas_locs = data[0]["meas_locs"] # :: { meas index, [t, x] }
    assert all(np.all(d["meas_locs"] == meas_locs) for d in data)
    prob = np.concatenate([d["prob_data"] for d in data]) # :: { run, chi, meas index }

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

def main() -> None:
    indir = (
        Path("output")
        .joinpath("haar_coev_probs")
        .joinpath("cluster")
        .joinpath("nalgebra")
    )
    outdir = Path("output").joinpath("haar_coev_probs")
    manifest_file = (
        Path("n=18")
        .joinpath("manifest_seed=10546_nqubits=18_depth=180_id=a41c1b5246b7b946.cbor")
    )

    processed = Processed.process_set(indir.joinpath(manifest_file), outdir)
    avg_slopes = processed.fits[:, :, :, 1].mean(axis=0) # :: { p, chi }
    seed = processed.seed
    size = processed.size
    depth = processed.depth
    p_meas = processed.p_meas
    chi = processed.chi

    avg_slopes_diff = map_outer(
        lambda slopes_p: slopes_p - slopes_p[-1],
        avg_slopes,
    ) # :: { p, chi }

    P = pd.Plotter()
    chi_c = chi[np.where(chi != 0)]
    chi_min = chi_c.min()
    chi_max = chi_c.max()
    dx = chi_max - chi_min
    colors = [pd.colormaps["vibrant"]((x - chi_min) / dx) for x in chi_c]
    it = enumerate(zip(chi, avg_slopes_diff.T, colors))
    for (k, (x, slopes_diff_x, c)) in it:
        if x == 0:
            continue
        P.semilogy(
            p_meas, slopes_diff_x,
            marker=".", linestyle="-", color=c,
            label=f"$\\chi = {x if x > 0 else '\\infty'}$",
        )
    (
        P
        .ggrid()
        .legend(
            fontsize="xx-small",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=1.0,
        )
        .set_xlabel("$p$")
        .set_ylabel("Slope diff. from $\\chi = \\infty$")
        .set_ylim(bottom=1e-9)
        .set_title(f"{size = }; {depth = }")
        .savefig(
            outdir.joinpath(f"haar_coev_probs_nd_seed={seed}_n={size}_depth={depth}_diff_log.png")
        )
        .close()
    )

if __name__ == "__main__":
    main()

