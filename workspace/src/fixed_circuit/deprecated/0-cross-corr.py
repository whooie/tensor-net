# deprecated in favor of averaging over spacetime interval instances before
# averaging over runs -- this one averages both together

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
import numpy as np
import whooie.pyplotdefs as pd
FS = pd.pp.rcParams["figure.figsize"]

@dataclass
class Data:
    size: int
    depth: int
    p_meas: float
    runs: int
    outcomes: np.ndarray[np.int8, 3]

def get_data(infile: Path) -> Data:
    data = np.load(str(infile))
    size = int(data["size"][0])
    depth = int(data["depth"][0])
    p_meas = float(data["p_meas"][0])
    runs = int(data["runs"][0])
    outcomes = data["outcomes"]
    return Data(size, depth, p_meas, runs, outcomes)

def doit(infile: Path):
    outfile = infile.with_suffix(".png")
    data = get_data(infile)
    size = data.size
    depth = data.depth
    p_meas = data.p_meas
    runs = data.runs
    outcomes = data.outcomes

    P = (
        pd.Plotter.new_gridspec(
            dict(nrows=3, ncols=4),
            [
                pd.S[0, 1:3],
                pd.S[1, 0:2],
                pd.S[2, 0:2],
                pd.S[1, 2],
                pd.S[2, 2],
                pd.S[1, 3],
                pd.S[2, 3],
            ],
            figsize=[3.0 * FS[0], 1.50 * FS[0]],
        )
        .to_plotarray()
        .suptitle(f"$p = {p_meas:.3}; \\mathregular{{runs}} = {runs}$")
    )

    (
        P[0]
        .imshow(
            1 - np.abs(outcomes[0, :, :]).T,
            origin="lower",
            aspect="equal",
            extent=[
                -0.5, depth + 0.5,
                -0.5, size + 0.5,
            ],
            cmap="gray",
        )
        .ggrid().grid(False, which="both")
        .set_xlabel("$t$")
        .set_ylabel("$x$")
    )

    meas_locs = list(zip(*np.where(abs(outcomes[0, :, :]) != 0)))
    pairs = ((tx0, tx1) for (tx0, tx1) in product(meas_locs, meas_locs) if tx0 != tx1)

    pairdiff = np.zeros((2 * depth - 1, 2 * size - 1), dtype=np.uint32)
    corrdiff = np.nan * np.zeros((2 * depth - 1, 2 * size - 1), dtype=np.float64)
    pairx = np.zeros((size, size), dtype=np.uint32)
    corrx = np.nan * np.zeros((size, size), dtype=np.float64)
    pairt = np.zeros((depth, depth), dtype=np.uint32)
    corrt = np.nan * np.zeros((depth, depth), dtype=np.float64)
    for ((t0, x0), (t1, x1)) in pairs:
        dt = t1 - t0
        dx = x1 - x0

        if pairdiff[dt, dx] > 0:
            corrdiff[dt, dx] = (
                pairdiff[dt, dx] * runs * corrdiff[dt, dx]
                + (outcomes[:, t0, x0] * outcomes[:, t1, x1]).sum()
            ) / ((pairdiff[dt, dx] + 1) * runs)
        else:
            corrdiff[dt, dx] = (
                (outcomes[:, t0, x0] * outcomes[:, t1, x1]).sum()
            ) / runs
        pairdiff[dt, dx] += 1

        if dt == 0:
            if pairx[x0, x1] > 0:
                corrx[x0, x1] = (
                    pairx[x0, x1] * runs * corrx[x0, x1]
                    + (outcomes[:, t0, x0] * outcomes[:, t1, x1]).sum()
                ) / ((pairx[x0, x1] + 1) * runs)
            else:
                corrx[x0, x1] = (
                    (outcomes[:, t0, x0] * outcomes[:, t1, x1]).sum()
                ) / runs
            pairx[x0, x1] += 1

        if dx == 0:
            if pairt[t0, t1] > 0:
                corrt[t0, t1] = (
                    pairt[t0, t1] * runs * corrt[t0, t1]
                    + (outcomes[:, t0, x0] * outcomes[:, t1, x1]).sum()
                ) / ((pairt[t0, t1] + 1) * runs)
            else:
                corrt[t0, t1] = (
                    (outcomes[:, t0, x0] * outcomes[:, t1, x1]).sum()
                ) / runs
            pairt[t0, t1] += 1

    corrdiff = np.roll(np.roll(corrdiff, -size, axis=1), -depth, axis=0)
    pairdiff = (
        np.roll(np.roll(pairdiff, -size, axis=1), -depth, axis=0)
        .astype(np.float64)
    )
    pairdiff[np.where(pairdiff == 0)] = np.nan
    vmaxdiff = abs(corrdiff[np.where(True ^ np.isnan(corrdiff))]).max()
    vmindiff = -vmaxdiff
    (
        P[1]
        .imshow(
            corrdiff.T,
            origin="lower",
            aspect="equal",
            extent=[
                -depth + 0.5, depth - 0.5,
                -size + 0.5, size - 0.5,
            ],
            vmin=vmindiff,
            vmax=vmaxdiff,
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$\\Delta t$")
        .set_ylabel("$\\Delta x$")
        .set_clabel("$\\langle Z(t, x) Z(t', x') \\rangle$")
        [2]
        .imshow(
            pairdiff.T,
            origin="lower",
            aspect="equal",
            extent=[
                -depth + 0.5, depth - 0.5,
                -size + 0.5, size - 0.5,
            ],
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$\\Delta t$")
        .set_ylabel("$\\Delta x$")
        .set_clabel("$\\#(\\Delta t, \\Delta x)$")
    )

    pairx = pairx.astype(np.float64)
    pairx[np.where(pairx == 0)] = np.nan
    vmaxx = abs(corrx[np.where(True ^ np.isnan(corrx))]).max()
    vminx = -vmaxx
    (
        P[3]
        .imshow(
            corrx,
            origin="lower",
            aspect="equal",
            extent=[-0.5, size - 0.5, -0.5, size - 0.5],
            vmin=vminx,
            vmax=vmaxx,
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$x_1$")
        .set_ylabel("$x_0$")
        .set_clabel("$\\langle Z(x_0) Z(x_1) \\rangle_{{\\Delta t = 0}}$")
        [4]
        .imshow(
            pairx,
            origin="lower",
            aspect="equal",
            extent=[-0.5, size - 0.5, -0.5, size - 0.5],
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$x_1$")
        .set_ylabel("$x_0$")
        .set_clabel("$\\#(x_0, x_1)_{{\\Delta t = 0}}$")
    )

    pairt = pairt.astype(np.float64)
    pairt[np.where(pairt == 0)] = np.nan
    vmaxt = abs(corrt[np.where(True ^ np.isnan(corrt))]).max()
    vmint = -vmaxt
    (
        P[5]
        .imshow(
            corrt,
            origin="lower",
            aspect="equal",
            extent=[-0.5, depth - 0.5, -0.5, depth - 0.5],
            vmin=vmint,
            vmax=vmaxt,
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$t_1$")
        .set_ylabel("$t_0$")
        .set_clabel("$\\langle Z(t_0) Z(t_1) \\rangle_{{\\Delta x = 0}}$")
        [6]
        .imshow(
            pairt,
            origin="lower",
            aspect="equal",
            extent=[-0.5, depth - 0.5, -0.5, depth - 0.5],
        )
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$t_1$")
        .set_ylabel("$t_0$")
        .set_clabel("$\\#(t_0, t_1)_{{\\Delta x = 0}}$")
    )

    (
        P
        .tight_layout(w_pad=0.05, h_pad=0.05)
        .savefig(outfile)
        .close()
    )

def main():
    if len(sys.argv) < 2:
        print("missing input file(s)")
        sys.exit(1)

    infiles = [Path(f) for f in sys.argv[1:]]
    for infile in infiles:
        print(infile)
        doit(infile)

if __name__ == "__main__":
    main()


