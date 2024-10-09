from dataclasses import dataclass
from itertools import product
from math import ceil
import os
from pathlib import Path
import re
from typing import Self, Callable, Iterable, Optional
import numpy as np
from whooie.analysis import ExpVal
import whooie.pyplotdefs as pd

indir = Path("output")
infile_pat = re.compile(r'n=(\d+)_d=(\d+)_p=([0-9.]+)_chi=(\d+)_runs=(\d+)_(\d+)\.npz')
outdir = Path("output")

def qubit_kl(
    pset: np.ndarray[np.int8, 1],
    qset: np.ndarray[np.int8, 1],
) -> ExpVal:
    Np = (pset != 0).sum()
    Np_up = (pset > 0).sum()
    Np_dn = (pset < 0).sum()
    p_up = ExpVal(Np_up / Np, np.sqrt(Np_up) / Np)
    p_dn = ExpVal(Np_dn / Np, np.sqrt(Np_dn) / Np)

    Nq = (qset != 0).sum()
    Nq_up = (qset > 0).sum()
    Nq_dn = (qset < 0).sum()
    q_up = ExpVal(Nq_up / Nq, np.sqrt(Nq_up) / Nq)
    q_dn = ExpVal(Nq_dn / Nq, np.sqrt(Nq_dn) / Nq)

    # # (pset | qset)
    # return p_up * (p_up.ln() - q_up.ln()) + p_dn * (p_dn.ln() - q_dn.ln())

    # (qset | pset)
    return q_up * (q_up.ln() - p_up.ln()) + q_dn * (q_dn.ln() - p_dn.ln())

@dataclass
class Data:
    size: int
    depth: int
    p_meas: float
    chi: int
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
            chi = float(data["chi"][0])
            runs = int(data["runs"][0])
            outcomes = data["outcomes"][:, 2 * size:, :]
            circ_id = int(m.group(6))
            return Data(size, depth, p_meas, chi, runs, outcomes, circ_id)
        else:
            raise Exception(f"Data.from_file: invalid file name {str(infile)}")

    def kl_interval(self, other: Self) -> np.ndarray[ExpVal, 2]:
        assert self.circ_id == other.circ_id
        assert self.outcomes.shape == other.outcomes.shape
        meas_locs = list(zip(*np.where(abs(self.outcomes[0, :, :]) != 0)))
        pairs = product(meas_locs, meas_locs)
        occ = np.zeros((2 * self.depth - 1, 2 * self.size - 1), dtype=np.uint32)
        kl = np.array(occ.shape[0] * [occ.shape[1] * [ExpVal.nan()]])
        for ((t0, x0), (t1, x1)) in pairs:
            dt = t1 - t0
            dx = x1 - x0
            if occ[dt, dx] > 0:
                kl[dt, dx] = (
                    occ[dt, dx] * kl[dt, dx]
                    + qubit_kl(
                        self.outcomes[:, t0, x0], other.outcomes[:, t1, x1])
                ) / (occ[dt, dx] + 1)
            else:
                kl[dt, dx] = (
                    qubit_kl(
                        self.outcomes[:, t0, x0], other.outcomes[:, t1, x1])
                )
            occ[dt, dx] += 1
        return np.roll(np.roll(kl, -self.size, axis=1), -self.depth, axis=0)

    def kl_x(self, other: Self, d: int) -> np.ndarray[ExpVal, 2]:
        assert self.circ_id == other.circ_id
        assert self.outcomes.shape == other.outcomes.shape
        meas_locs = list(zip(*np.where(abs(self.outcomes[0, :, :]) != 0)))
        pairs = (
            (tx0, tx1) for (tx0, tx1) in product(meas_locs, meas_locs)
            if abs(tx0[0] - tx1[0]) == d
        )
        occ = np.zeros((self.size, self.size), dtype=np.uint32)
        kl = np.array(occ.shape[0] * [occ.shape[1] * [ExpVal.nan()]])
        for ((t0, x0), (t1, x1)) in pairs:
            if occ[x0, x1] > 0:
                kl[x0, x1] = (
                    occ[x0, x1] * kl[x0, x1]
                    + qubit_kl(
                        self.outcomes[:, t0, x0], other.outcomes[:, t1, x1])
                ) / (occ[x0, x1] + 1)
            else:
                kl[x0, x1] = (
                    qubit_kl(
                        self.outcomes[:, t0, x0], other.outcomes[:, t1, x1])
                )
            occ[x0, x1] += 1
        return kl

    def kl_t(self, other: Self, d: int) -> np.ndarray[ExpVal, 2]:
        assert self.circ_id == other.circ_id
        assert self.outcomes.shape == other.outcomes.shape
        meas_locs = list(zip(*np.where(abs(self.outcomes[0, :, :]) != 0)))
        pairs = (
            (tx0, tx1) for (tx0, tx1) in product(meas_locs, meas_locs)
            if abs(tx0[1] - tx1[1]) == d
        )
        occ = np.zeros((self.depth, self.depth), dtype=np.uint32)
        kl = np.array(occ.shape[0] * [occ.shape[1] * [ExpVal.nan()]])
        for ((t0, x0), (t1, x1)) in pairs:
            if occ[t0, t1] > 0:
                kl[t0, t1] = (
                    occ[t0, t1] * kl[t0, t1]
                    + qubit_kl(
                        self.outcomes[:, t0, x0], other.outcomes[:, t1, x1])
                ) / (occ[t0, t1] + 1)
            else:
                kl[t0, t1] = (
                    qubit_kl(
                        self.outcomes[:, t0, x0], other.outcomes[:, t1, x1])
                )
            occ[t0, t1] += 1
        return kl

@dataclass
class Dataset:
    data: list[Data]

    @staticmethod
    def from_dir(path: Path) -> Self:
        return Dataset([
            Data.from_file(path.joinpath(f)) for f in os.listdir(path)
            if infile_pat.match(f)
        ])

    def filter(self, f: Callable[[Data], bool]) -> Iterable[Data]:
        filter(f, self.data)

    def filter_circ(self, circ_id: int) -> Iterable[Data]:
        filter(lambda d: d.circ_id == circ_id, self.data)

    def filter_p(self, p: float) -> Iterable[Data]:
        filter(lambda d: d.p_meas == p, self.data)

    def filter_chi(self, chi: int) -> Iterable[Data]:
        filter(lambda d: d.chi == chi, self.data)

    def all_p(self) -> np.ndarray[float, 1]:
        return np.array(list(set(d.p_meas for d in self.data)))

    def all_chi(self) -> np.ndarray[int, 1]:
        return np.array(list(set(d.chi for d in self.data)))

    def find_by(self, f: Callable[[Data], bool]) -> Optional[Data]:
        for d in self.data:
            if f(d):
                return d
        return None

def map_arr(arr: np.ndarray, f: Callable) -> np.ndarray:
    return np.array([f(a) for a in arr.flatten()]).reshape(arr.shape)

def doit(data: Dataset, circ_id: int, chi: int, p: float, d: int):
    FS = pd.pp.rcParams["figure.figsize"]
    fmt = f"circ={circ_id:03}_chi={chi}_p={p:.3f}_dxt={d}"
    title = f"$p={p:.3f}; \\chi={chi}; \\Delta={d}; [{circ_id:03}]$"

    data_q = data.find_by(
        lambda d: d.p_meas == p and d.chi == 0 and d.circ_id == circ_id)
    data_c = data.find_by(
        lambda d: d.p_meas == p and d.chi == chi and d.circ_id == circ_id)
    assert data_q is not None and data_c is not None

    kl_interval = data_q.kl_interval(data_c)
    kl = map_arr(kl_interval, lambda v: v.val)
    mean_interval = kl[np.where(True ^ np.isnan(kl))].mean()
    kl_x = data_q.kl_x(data_c, d)
    kl = map_arr(kl_x, lambda v: v.val)
    mean_x = kl[np.where(True ^ np.isnan(kl))].mean()
    kl_t = data_q.kl_t(data_c, d)
    kl = map_arr(kl_t, lambda v: v.val)
    mean_t = kl[np.where(True ^ np.isnan(kl))].mean()

    (
        pd.Plotter.new_gridspec(
            dict(nrows=2, ncols=4),
            [
                pd.S[0, 0:2], pd.S[0, 2:4],
                pd.S[1, 0], pd.S[1, 1],
                pd.S[1, 2], pd.S[1, 3],
            ],
            {
                1: {"y": 0},
                3: {"y": 2},
                5: {"y": 4},
            },
            figsize=[3 * FS[0], 2 * FS[1]],
        )
        .to_plotarray()
        .suptitle(title)
        [0]
        .imshow(
            map_arr(kl_interval, lambda v: np.log10(v.val)).T,
            origin="lower",
            aspect="equal",
            extent=[
                -data_q.depth + 0.5, data_q.depth - 0.5,
                -data_q.size + 0.5, data_q.size - 0.5,
            ],
            vmin=-1,
            vmax=-5,
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$\\Delta t$")
        .set_ylabel("$\\Delta x$")
        .set_clabel("$\\log_{{10}} \\langle \\mathregular{KL}(p_Q | p_C) \\rangle$")
        .set_title(f"avg = {mean_interval:.3e}", fontsize="small")
        [1]
        .imshow(
            map_arr(kl_interval, lambda v: np.log10(v.err)).T,
            origin="lower",
            aspect="equal",
            extent=[
                -data_q.depth + 0.5, data_q.depth - 0.5,
                -data_q.size + 0.5, data_q.size - 0.5,
            ],
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$\\Delta t$")
        .set_clabel("Error")
        [2]
        .imshow(
            map_arr(kl_x, lambda v: np.log10(v.val)),
            origin="lower",
            aspect="equal",
            extent=[-0.5, data_q.size - 0.5, -0.5, data_q.size - 0.5],
            vmin=-1,
            vmax=-5,
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$x_1$")
        .set_ylabel("$x_0$")
        .set_clabel(f"$\\log_{{10}} \\langle \\mathregular{{KL}}(p_Q | p_C) \\rangle_{{|\\Delta t| = {d}}}$")
        .set_title(f"avg = {mean_x:.3e}", fontsize="small")
        [3]
        .imshow(
            map_arr(kl_x, lambda v: np.log10(v.err)),
            origin="lower",
            aspect="equal",
            extent=[-0.5, data_q.size - 0.5, -0.5, data_q.size - 0.5],
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$x_1$")
        .set_clabel("Error")
        [4]
        .imshow(
            map_arr(kl_t, lambda v: np.log10(v.val)),
            origin="lower",
            aspect="equal",
            extent=[-0.5, data_q.depth - 0.5, -0.5, data_q.depth - 0.5],
            vmin=-5,
            vmax=-1,
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$t_1$")
        .set_ylabel("$t_0$")
        .set_clabel(f"$\\log_{{10}} \\langle \\mathregular{{KL}}(p_Q | p_C) \\rangle_{{|\\Delta x| = {d}}}$")
        .set_title(f"avg = {mean_t:.3e}", fontsize="small")
        [5]
        .imshow(
            map_arr(kl_t, lambda v: np.log10(v.err)),
            origin="lower",
            aspect="equal",
            extent=[-0.5, data_q.depth - 0.5, -0.5, data_q.depth - 0.5],
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$t_1$")
        .set_clabel("Error")
        .tight_layout(w_pad=0.2, h_pad=0.2)
        .savefig(outdir.joinpath(f"klqc_{fmt}.png"))
        .close()
    )

def main():
    data = Dataset.from_dir(indir)
    it = product(
        [0, 1, 2, 3, 4],
        [16, 32, 64, 128],
        [0.10, 0.15, 0.20, 0.26, 0.27, 0.30, 0.35],
        [1],
    )
    for params in it:
        print(params)
        doit(data, *params)

if __name__ == "__main__":
    main()

