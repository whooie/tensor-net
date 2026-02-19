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

def main() -> None:
    datadir = Path("output").joinpath("haar_coev_probs")
    infile0 = datadir.joinpath("haar_coev_probs_nd_seed=10546_n=18_depth=180_fitavg.npz")
    infile1 = datadir.joinpath("haar_coev_probs_seed=10546_n=18_depth=180_circ=processed_id=comb.npz")

    data0 = np.load(str(infile0))
    seed0 = data0["seed"][0]
    size0 = data0["size"][0]
    depth0 = data0["depth"][0]
    p0 = data0["p_meas"]
    chi0 = data0["chi"]
    assert chi0[-1] == 0
    slopes0 = data0["fits"][:, :, :, 1].mean(axis=0) # :: { p, chi }
    slope_diffs0 = map_outer(
        lambda slopes0_p: slopes0_p - slopes0_p[-1], slopes0)

    data1 = np.load(str(infile1))
    seed1 = data1["seed"][0]
    size1 = data1["size"][0]
    depth1 = data1["depth"][0]
    p1 = data1["p_meas"]
    chi1 = data1["chi"]
    assert chi1[-1] == 0
    slopes1 = data1["fits"][:, :, :, 1].mean(axis=0) # :: { p, chi }
    slope_diffs1 = map_outer(
        lambda slopes1_p: slopes1_p - slopes1_p[-1], slopes1)

    assert seed0 == seed1
    seed = seed0
    assert size0 == size1
    size = size0
    assert depth0 == depth1
    depth = depth0

    p_idx = np.array([
        [np.abs(p0 - p1_i).argmin(), i]
        for (i, p1_i) in enumerate(p1)
        if any(abs(p0_k - p1_i) < 1e-6 for p0_k in p0)
    ])
    chi_idx = np.array([
        [np.abs(chi0 - chi1_j).argmin(), j]
        for (j, chi1_j) in enumerate(chi1)
        if chi1_j in chi0
    ])

    p = p1[p_idx[:, 1]]
    chi = chi1[chi_idx[:, 1]]

    slope_errs = np.array([
        [
            slopes1_ij - slopes0_ij
            for (slopes0_ij, slopes1_ij)
            in zip(slopes0_i[chi_idx[:, 0]], slopes1_i[chi_idx[:, 1]])
        ]
        for (slopes0_i, slopes1_i)
        in zip(slopes0[p_idx[:, 0], :], slopes1[p_idx[:, 1], :])
    ])
    slopediff_errs = np.array([
        [
            slope_diffs1_ij - slope_diffs0_ij
            for (slope_diffs0_ij, slope_diffs1_ij)
            in zip(slope_diffs0_i[chi_idx[:, 0]], slope_diffs1_i[chi_idx[:, 1]])
        ]
        for (slope_diffs0_i, slope_diffs1_i)
        in zip(slope_diffs0[p_idx[:, 0], :], slope_diffs1[p_idx[:, 1], :])
    ])

    P = pd.Plotter()
    where_c = np.where(chi != 0)[0]
    chi_c = chi[where_c]
    chi_min = chi_c.min()
    chi_max = chi_c.max()
    dx = chi_max - chi_min
    colors = [pd.colormaps["vibrant"]((x - chi_min) / dx) for x in chi_c]
    it = enumerate(zip(chi_c, slope_errs.T[where_c, :], colors))
    for (k, (x, slope_err_x, c)) in it:
        P.plot(
            p, slope_err_x,
            marker=".", linestyle="-", color=c,
            label=f"$\\chi = {x if x > 0 else '\\infty'}$",
        )
    where_q = np.where(chi == 0)[0][0]
    P.plot(
        p, slope_errs[:, where_q],
        marker=".", linestyle="-", color="r",
        label=f"$\\chi = \\infty$",
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
        .set_ylabel("Slope error from old SVD")
        .savefig(
            datadir.joinpath(f"haar_coev_probs_nd_seed={seed}_n={size}_depth={depth}_svd_error.png")
        )
        .close()
    )

    P = pd.Plotter()
    it = enumerate(zip(chi_c, slopediff_errs.T[where_c, :], colors))
    for (k, (x, slopediff_err_x, c)) in it:
        P.plot(
            p, slopediff_err_x,
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
        .set_ylabel("Slope diff. error from old SVD")
        .savefig(
            datadir.joinpath(f"haar_coev_probs_nd_seed={seed}_n={size}_depth={depth}_svd_slopediff_error.png")
        )
        .close()
    )

if __name__ == "__main__":
    main()

