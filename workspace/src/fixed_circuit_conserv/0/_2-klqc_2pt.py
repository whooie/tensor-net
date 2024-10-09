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
    """
    Computes the KL divergence `KL(q | p)`, where `q` and `p` are computed from
    the raw data sets `qset` and `pset` of binary ±1 values, respectively.
    """
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

def qubit2_kl(
    pset: np.ndarray[np.int8, 2],
    qset: np.ndarray[np.int8, 2],
) -> ExpVal:
    """
    Computes the KL divergence `KL(q | p)`, where `q` and `p` are computed over
    2-bit pairs from the raw data sets `qset` and `pset` of ±1 values,
    respectively.
    """
    Np = ((pset[:, 0] != 0) & (pset[:, 1] != 0)).sum()
    Np_0 = ((pset[:, 0] < 0) & (pset[:, 1] < 0)).sum()
    Np_1 = ((pset[:, 0] < 0) & (pset[:, 1] > 0)).sum()
    Np_2 = ((pset[:, 0] > 0) & (pset[:, 1] < 0)).sum()
    Np_3 = ((pset[:, 0] > 0) & (pset[:, 1] > 0)).sum()
    # p_0 = ExpVal(Np_0 / Np, np.sqrt(Np_0) / Np)
    # p_1 = ExpVal(Np_1 / Np, np.sqrt(Np_1) / Np)
    # p_2 = ExpVal(Np_2 / Np, np.sqrt(Np_2) / Np)
    # p_3 = ExpVal(Np_3 / Np, np.sqrt(Np_3) / Np)
    Pp_0 = Np_0 / Np
    Pp_1 = Np_1 / Np
    Pp_2 = Np_2 / Np
    Pp_3 = Np_3 / Np
    p_0 = ExpVal(Pp_0, np.sqrt(Pp_0 * (1 - Pp_0) / Np))
    p_1 = ExpVal(Pp_1, np.sqrt(Pp_1 * (1 - Pp_1) / Np))
    p_2 = ExpVal(Pp_2, np.sqrt(Pp_2 * (1 - Pp_2) / Np))
    p_3 = ExpVal(Pp_3, np.sqrt(Pp_3 * (1 - Pp_3) / Np))
    p = [p_0, p_1, p_2, p_3]

    Nq = ((qset[:, 0] != 0) & (qset[:, 1] != 0)).sum()
    Nq_0 = ((qset[:, 0] < 0) & (qset[:, 1] < 0)).sum()
    Nq_1 = ((qset[:, 0] < 0) & (qset[:, 1] > 0)).sum()
    Nq_2 = ((qset[:, 0] > 0) & (qset[:, 1] < 0)).sum()
    Nq_3 = ((qset[:, 0] > 0) & (qset[:, 1] > 0)).sum()
    # q_0 = ExpVal(Nq_0 / Nq, np.sqrt(Nq_0) / Nq)
    # q_1 = ExpVal(Nq_1 / Nq, np.sqrt(Nq_1) / Nq)
    # q_2 = ExpVal(Nq_2 / Nq, np.sqrt(Nq_2) / Nq)
    # q_3 = ExpVal(Nq_3 / Nq, np.sqrt(Nq_3) / Nq)
    Pq_0 = Nq_0 / Np
    Pq_1 = Nq_1 / Np
    Pq_2 = Nq_2 / Np
    Pq_3 = Nq_3 / Np
    q_0 = ExpVal(Pq_0, np.sqrt(Pq_0 * (1 - Pq_0) / Np))
    q_1 = ExpVal(Pq_1, np.sqrt(Pq_1 * (1 - Pq_1) / Np))
    q_2 = ExpVal(Pq_2, np.sqrt(Pq_2 * (1 - Pq_2) / Np))
    q_3 = ExpVal(Pq_3, np.sqrt(Pq_3 * (1 - Pq_3) / Np))
    q = [q_0, q_1, q_2, q_3]

    # # (pset | qset)
    # return sum(p_k * (p_k.ln() - q_k.ln()) for (p_k, q_k) in zip(p, q))

    # (qset | pset)
    return sum(
        q_k * (q_k.ln() - p_k.ln())
        for (q_k, p_k) in zip(q, p)
        if q_k.val != 0 and p_k.val != 0
    )

def qubit1_moms(
    outcomes: np.ndarray[np.int8, 3],
) -> (np.ndarray[np.float64, 2], np.ndarray[np.float64, 2]):
    zero_locs = np.where(np.all(outcomes == 0, axis=0))
    mean = outcomes.mean(axis=0)
    var = outcomes.std(axis=0) ** 2
    mean[zero_locs] = np.nan
    var[zero_locs] = np.nan
    return (mean, var)

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

    def kl_interval(
        self,
        tx0: (int, int),
        other: Self,
    ) -> ((int, int), np.ndarray[ExpVal, 2]):
        assert self.circ_id == other.circ_id
        assert self.outcomes.shape == other.outcomes.shape
        (t0, x0) = tx0
        if (
            np.any(self.outcomes[:, t0, x0] == 0)
            or np.any(other.outcomes[:, t0, x0] == 0)
        ):
            raise ValueError("Data.kl_interval: invalid fixed position")
        meas_locs = list(zip(*np.where(abs(self.outcomes[0, :, :]) != 0)))
        pset0 = self.outcomes[:, t0, x0]
        qset0 = other.outcomes[:, t0, x0]
        kl = np.array(self.depth * [self.size * [ExpVal.nan()]])
        for (t1, x1) in meas_locs:
            dt = t1 - t0
            dx = x1 - x0
            pset = np.array([pset0, self.outcomes[:, t1, x1]]).T
            qset = np.array([qset0, other.outcomes[:, t1, x1]]).T
            kl[dt, dx] = qubit2_kl(pset, qset)
            if np.isnan(kl[dt, dx].val):
                raise ValueError(f"nan detected in kl_interval, {dt=} {dx=}")
        return (tx0min, kl)

def min_by(f: Callable[[...], float], items: Iterable) -> Optional[...]:
    score = np.inf
    it = None
    for item in items:
        s = f(item)
        if s < score:
            it = item
            score = s
    return it

@dataclass
class Dataset:
    data: list[Data]

    def __iter__(self):
        return iter(self.data)

    @staticmethod
    def from_dir(path: Path) -> Self:
        return Dataset([
            Data.from_file(path.joinpath(f)) for f in sorted(os.listdir(path))
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

@dataclass
class Processed:
    kl: np.ndarray[ExpVal, 2]
    mean: float
    depth: int
    size: int
    t0: int
    x0: int
    circ_id: int
    chi: int
    p_meas: float

def map_arr(arr: np.ndarray, f: Callable) -> np.ndarray:
    return np.array([f(a) for a in arr.flatten()]).reshape(arr.shape)

def doit(
    data: Dataset,
    circ_id: int,
    chi: int,
    p: float,
    plotflag: bool=True,
) -> Processed:
    print(circ_id, chi, p)
    FS = pd.pp.rcParams["figure.figsize"]
    fmt = f"circ={circ_id:03}_chi={chi}_p={p:.3f}"
    title = f"$p={p:.3f}; \\chi={chi}; [{circ_id:03}]$"

    data_q = data.find_by(
        lambda d: d.p_meas == p and d.chi == 0 and d.circ_id == circ_id)
    data_c = data.find_by(
        lambda d: d.p_meas == p and d.chi == chi and d.circ_id == circ_id)
    assert data_q is not None and data_c is not None
    assert data_q.outcomes.shape == data_c.outcomes.shape

        # tx0min = min_by(
        #     lambda loc: 2 * abs(loc[0] - tx0[0]) + abs(loc[1] - tx0[1]),
        #     meas_locs,
        # )
        # assert tx0min is not None
        # (t0, x0) = tx0min

    depth = data_q.depth
    size = data_q.size
    t0 = depth - 1
    x0 = size // 2

    ((t0, x0), kl) = data_q.kl_interval((t0, x0), data_c)
    kl_val = map_arr(kl, lambda v: v.val)
    mean = kl_val[np.where(True ^ np.isnan(kl_val))].mean()
    res = Processed(kl, mean, depth, size, t0, x0, circ_id, chi, p)

    kl = np.roll(np.roll(kl, x0, axis=1), t0, axis=0)
    extent = [-t0 - 0.5, depth - t0 - 0.5, -x0 - 0.5, size - x0 - 0.5]

    if plotflag:
        (
            pd.Plotter.new(ncols=2, sharey=True, figsize=[1.5 * FS[0], FS[1]])
            .to_plotarray()
            .suptitle(title)
            [0]
            .imshow(
                # map_arr(kl, lambda v: v.val).T,
                map_arr(kl, lambda v: np.log10(v.val)).T,
                origin="lower",
                aspect="equal",
                extent=extent,
                vmin=-1,
                vmax=-5,
            )
            .colorbar()
            .ggrid().grid(False, which="both")
            .set_xlabel("$\\Delta t$")
            .set_ylabel("$\\Delta x$")
            .set_clabel(
                "$\\log_{10}"
                " \\mathregular{KL}("
                "p^C_{\\vec{x}_0, \\vec{x}_0 + \\vec{\\Delta}}"
                " | "
                "p^Q_{\\vec{x}_0, \\vec{x}_0 + \\vec{\\Delta}}"
                ")$"
            )
            .set_title(
                f"avg = {mean:.3e}; $t_0 = {t0}$; $x_0 = {x0}$",
                fontsize="small",
            )
            [1]
            .imshow(
                map_arr(kl, lambda v: np.log10(v.err)).T,
                origin="lower",
                aspect="equal",
                extent=extent,
            )
            .colorbar()
            .ggrid().grid(False, which="both")
            .set_xlabel("$\\Delta t$")
            .set_clabel("$\\log_{10} \\mathregular{Error}$")
            .tight_layout(w_pad=0.2, h_pad=0.2)
            .savefig(outdir.joinpath(f"klqc_{fmt}.png"))
            .close()
        )

    return res

def qubit2_bins(
    set0: np.ndarray[np.int8, 1],
    set1: np.ndarray[np.int8, 1],
) -> np.ndarray[np.float64, 1]:
    vals = (1 + set0) + (1 + set1) / 2
    probs = np.histogram(
        vals,
        bins=np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
        density=True,
    )
    return probs

def shannon(p: np.ndarray[np.float64, 1]) -> np.float64:
    return sum(-pk * np.log(pk) for pk in p if p > 0.0)

def main():
    plotflag = False
    FS = pd.pp.rcParams["figure.figsize"]
    data = Dataset.from_dir(indir)
    circs = [0, 1, 2]
    # chi = [16, 32, 64, 128]
    chi = [16, 32, 64]
    # p = [0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]
    p = [
        0.02, 0.03, 0.04, 0.05, 0.06,
        0.07, 0.08, 0.10, 0.12, 0.14,
        0.15, 0.16, 0.17, 0.18, 0.20,
        0.22, 0.25, 0.28, 0.31, 0.34,
        0.37, 0.40,
    ]
    it = product(circs, chi, p)
    processed = [doit(data, *params, plotflag) for params in it]
    circ_ls = ["-", "--", ":"]

    # if plotflag:
    #     for d in data:
    #         print(d.circ_id, d.chi, d.p_meas)
    #         fmt = f"circ={d.circ_id:03.0f}_chi={d.chi:.0f}_p={d.p_meas:.3f}"
    #         (mean, var) = qubit1_moms(d.outcomes)
    #         (
    #             pd.Plotter.new(
    #                 ncols=2,
    #                 sharey=True,
    #                 figsize=[1.5 * FS[0], FS[1]],
    #             )
    #             .to_plotarray()
    #             .suptitle(
    #                 f"$p = {d.p_meas:.3f}$; "
    #                 f"$\\chi = {d.chi:.0f}$; "
    #                 f"circ: {d.circ_id:03.0f}"
    #             )
    #             [0]
    #             .imshow(
    #                 mean.T,
    #                 origin="lower",
    #                 aspect="equal",
    #                 vmin=-1,
    #                 vmax=+1,
    #             )
    #             .colorbar()
    #             .ggrid().grid(False, which="both")
    #             .set_xlabel("$t$")
    #             .set_ylabel("$x$")
    #             .set_clabel("$\\langle Z_{t, x} \\rangle$")
    #             [1]
    #             .imshow(
    #                 var.T,
    #                 origin="lower",
    #                 aspect="equal",
    #                 vmin=0,
    #                 vmax=1.25,
    #             )
    #             .colorbar()
    #             .ggrid().grid(False, which="both")
    #             .set_xlabel("$t$")
    #             .set_clabel("$\\mathregular{Var} Z_{t, x}$")
    #             .tight_layout(w_pad=0.2, h_pad=0.2)
    #             .savefig(outdir.joinpath(f"moments_{fmt}.png"))
    #             .close()
    #         )

    kl00 = (
        np.array([d.kl[0, 0] for d in processed])
        .reshape((len(circs), len(chi), len(p)))
    )
    P = pd.Plotter()
    for (k, x) in enumerate(chi):
        P.plot(
            [], [],
            marker="", linestyle="-", color=f"C{k % 10}",
            label=f"$\\chi = {x}$",
        )
    for (c, ls) in zip(circs, circ_ls):
        P.plot(
            [], [],
            marker="", linestyle=ls, color="k",
            label=f"circ: {c}",
        )
    for (c, kl00_c, ls) in zip(circs, kl00, circ_ls):
        for (k, (x, kl00_cx)) in enumerate(zip(chi, kl00_c)):
            (
                P
                # .errorbar(
                .plot(
                    p,
                    map_arr(kl00_cx, lambda v: v.val),
                    # map_arr(kl00_cx, lambda v: v.err),
                    marker="o", linestyle=ls, color=f"C{k % 10}",
                )
            )
    (
        P
        .ggrid()
        .legend(fontsize="xx-small")
        .set_xlabel("$p$")
        .set_ylabel(
            "$\\mathregular{KL}("
            "p^C_{\\vec{x}_0, \\vec{x}_0}"
            " | "
            "p^Q_{\\vec{x}_0, \\vec{x}_0}"
            ")$"
        )
        .savefig(outdir.joinpath(f"klqc_00.png"))
        .semilogy([], [])
        .savefig(outdir.joinpath(f"klqc_00_log.png"))
        .close()
    )

    p0 = min(p)
    for circ in circs:
        of_circ = [proc for proc in processed if proc.circ_id == circ]
        proc_p0 = next((proc for proc in of_circ if proc.p_meas == p0))
        where = np.where(True ^ np.isnan(map_arr(proc_p0.kl, lambda v: v.val)))
        kl_dpos = [dtx for dtx in zip(*where) if dtx != (0, 0)]
        kl_dpos_sh = [
            (
                dt if dt < proc_p0.depth - proc_p0.t0 else dt - proc_p0.depth,
                dx if dx < proc_p0.size - proc_p0.x0 else dx - proc_p0.size,
            )
            for (dt, dx) in kl_dpos
        ]
        dtx0 = min(
            kl_dpos_sh,
            key=lambda dtx: np.inf if dtx[0] > 0 else abs(dtx[0]) + abs(dtx[1])
        )
        kl_dtx0 = (
            np.array([proc.kl[*dtx0] for proc in of_circ])
            .reshape((len(chi), len(p)))
        )

        P = pd.Plotter()#.new(figsize=[1.5 * FS[0], FS[1]])
        for (k, (x, kl_dtx0_x)) in enumerate(zip(chi, kl_dtx0)):
            (
                P
                .plot(
                    [], [],
                    marker="", linestyle="-", color=f"C{k % 10}",
                    label=f"$\\chi = {x}$",
                )
                # .errorbar(
                .plot(
                    p,
                    map_arr(kl_dtx0_x, lambda v: v.val),
                    # map_arr(kl_dtx0_x, lambda v: v.err),
                    marker="o", linestyle="-", color=f"C{k % 10}",
                )
            )
        (
            P
            .ggrid()
            .legend(fontsize="xx-small")
            .set_xlabel("$p$")
            .set_ylabel(
                "$\\mathregular{KL}("
                "p^C_{\\vec{x}_0, \\vec{x}_0 + \\vec{\\Delta}_0}"
                " | "
                "p^Q_{\\vec{x}_0, \\vec{x}_0 + \\vec{\\Delta}_0}"
                ")$"
            )
            .set_title(
                f"circ: {circ}; "
                f"$\\Delta t_0 = {dtx0[0]:+.0f}$; "
                f"$\\Delta x_0 = {dtx0[1]:+.0f}$"
            )
            .savefig(outdir.joinpath(f"klqc_dtx0_{circ:03.0f}.png"))
            .semilogy([], [])
            .savefig(outdir.joinpath(f"klqc_dtx0_{circ:03.0f}_log.png"))
            .close()
        )


if __name__ == "__main__":
    main()

