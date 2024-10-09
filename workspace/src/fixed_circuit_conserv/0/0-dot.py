from dataclasses import dataclass
from itertools import product
from math import ceil
import os
from pathlib import Path
import re
from typing import Self
import numpy as np
from whooie.analysis import ExpVal
import whooie.pyplotdefs as pd

indir = Path("output")
infile_pat = re.compile(r'n=(\d+)_d=(\d+)_p=([0-9.]+)_chi=000_runs=(\d+)_(\d+)\.npz')
outdir = Path("output")

@dataclass
class Data:
    size: int
    depth: int
    p_meas: float
    runs: int
    outcomes: np.ndarray[np.int8, 3]
    circ_id: int

    @staticmethod
    def from_file(infile: Path) -> Self:
        if m := infile_pat.match(str(infile.name)):
            data = np.load(str(infile))
            size = int(data["size"][0])
            depth = int(data["depth"][0]) - 2 * size
            p_meas = float(data["p_meas"][0])
            runs = int(data["runs"][0])
            outcomes = data["outcomes"][:, 2 * size:, :]
            circ_id = int(m.group(5))
            return Data(size, depth, p_meas, runs, outcomes, circ_id)
        else:
            raise Exception(f"Data.from_file: invalid file name {str(infile)}")

    def selfdot(self) -> ExpVal:
        nroll = ceil((self.runs - 1) / 2)
        if self.runs % 2 == 0:
            [accp, accm] = sum(
                (
                    dot_runs(self.outcomes, np.roll(self.outcomes, k))
                    .as_arr()[:, :(self.runs // 2 if k == nroll else self.runs)]
                    .sum(axis=1)
                )
                for k in range(1, nroll + 1)
            )
        else:
            [accp, accm] = sum(
                (
                    dot_runs(self.outcomes, np.roll(self.outcomes, k))
                    .as_arr()
                    .sum(axis=1)
                )
                for k in range(1, nroll + 1)
            )
        tot = (self.runs ** 2 - self.runs) // 2
        zz = (accp - accm) / tot
        zz_err = np.sqrt(accp + accm) / tot
        return ExpVal(zz, zz_err)

    def dot(self, other: Self) -> ExpVal:
        assert self.outcomes.shape == other.outcomes.shape
        assert self.circ_id == other.circ_id
        [accp, accm] = sum(
            (
                dot_runs(self.outcomes, np.roll(other.outcomes, k))
                .as_arr()
                .sum(axis=1)
            )
            for k in range(self.runs)
        )
        tot = self.runs ** 2
        zz = (accp - accm) / tot
        zz_err = np.sqrt(accp + accm) / tot
        return ExpVal(zz, zz_err)

@dataclass
class Dot:
    Np: np.ndarray[np.int8, 1]
    Nm: np.ndarray[np.int8, 1]

    def as_arr(self) -> np.ndarray[np.int32, 2]:
        return np.array([self.Np, self.Nm])

def dot_runs(a: np.ndarray[np.int8, 3], b: np.ndarray[np.int8, 3]) -> Dot:
    ab = a * b
    Np = (ab > 0).sum(axis=(1, 2))
    Nm = (ab < 0).sum(axis=(1, 2))
    return Dot(Np, Nm)

def pcomp(
    data: [Data]
) -> (np.ndarray[float, 1], np.ndarray[float, 2], np.ndarray[float, 2]):
    p = sorted({d.p_meas for d in data})
    comp = np.zeros((len(p), len(p)), dtype=float)
    comp_err = np.zeros(comp.shape, dtype=float)
    sel = pd.S[:]
    for ((i, pi), (j, pj)) in product(enumerate(p), enumerate(p)):
        if pi > pj:
            continue
        if i == j:
            dots = [a.selfdot() for a in data if a.p_meas == pi][sel]
            avg = sum(dots) / len(dots)
            comp[i, i] = avg.val
            comp_err[i, i] = avg.err
        else:
            dots = [
                a.dot(b) for (a, b) in product(data, data)
                if a.p_meas == pi and b.p_meas == pj and a.circ_id == b.circ_id
            ][sel]
            avg = sum(dots) / len(dots)
            comp[i, j] = avg.val
            comp[j, i] = avg.val
            comp_err[i, j] = avg.err
            comp_err[j, i] = avg.err
    return (p, comp, comp_err)

def main():
    FS = pd.pp.rcParams["figure.figsize"]
    data = [
        Data.from_file(indir.joinpath(f)) for f in os.listdir(indir)
        if infile_pat.match(f)
    ]
    (p, comp, comp_err) = pcomp(data)
    (
        pd.Plotter.new(
            ncols=2,
            sharey=True,
            figsize=[1.8 * FS[0], FS[1]],
            as_plotarray=True,
        )
        [0]
        .imshow(comp, origin="lower", aspect="equal")
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$p'$")
        .set_xticks(list(range(len(p))), [f"{pk}" for pk in p])
        .set_ylabel("$p$")
        .set_yticks(list(range(len(p))), [f"{pk}" for pk in p])
        .set_clabel("$\\langle Z(t, x, p) \\, Z(t, x, p') \\rangle$")
        [1]
        .imshow(comp_err, origin="lower", aspect="equal")
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$p'$")
        .set_xticks(list(range(len(p))), [f"{pk}" for pk in p])
        .set_clabel("Error")
        .tight_layout(h_pad=0.25)
        .savefig(outdir.joinpath("fixed_conserv_dot.png"))
    )

if __name__ == "__main__":
    main()

