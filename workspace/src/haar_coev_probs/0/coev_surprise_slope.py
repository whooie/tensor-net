from __future__ import annotations
from pathlib import Path
from typing import *

from lib.coev import (Processed, map_outer)
import lib.pyplotdefs as pd

def main() -> None:
    indir = (
        Path("output")
        .joinpath("haar_coev_probs")
        .joinpath("combflat")
    )
    outdir = (
        Path("output")
        .joinpath("haar_coev_probs")
        .joinpath("surprise_slope")
    )
    manifest_file = (
        Path("n=21")
        .joinpath("manifest_seed=10546_nqubits=21_depth=210_id=5b52503e59a064a6.cbor")
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
    cmap = pd.colormaps["vibrant"]
    colors = [cmap((x - chi_min) / dx) for x in chi_c]
    it = enumerate(zip(chi, avg_slopes_diff.T, colors))
    for (k, (x, slopes_diff_x, c)) in it:
        if x == 0:
            continue
        P.semilogy(
            p_meas, slopes_diff_x,
            marker=".", linestyle="-", color=c,
            label=f"$\\chi = {x}$",
        )
    (
        P
        .ggrid()
        .llegend()
        .set_xlabel("$p$")
        .set_ylabel("$\\Delta Q_{N, p, \\chi}^\\infty$")
        .set_title(f"{size = }; {depth = }")
        .savefig(
            outdir
            .joinpath(f"slopediffs_seed={seed}_n={size}_depth={depth}.png")
        )
        .close()
    )

if __name__ == "__main__":
    main()

