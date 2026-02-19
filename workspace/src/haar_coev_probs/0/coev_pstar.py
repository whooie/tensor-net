from __future__ import annotations
from typing import *

from lib.coev import (Processed, map_outer)

def find_last[T](
    items: Iterable[T],
    pred: Callable[[T], bool],
) -> Optional[tuple[int, T]]:
    acc: Optional[tuple[int, T]] = None
    for (k, item) in enumerate(items):
        if pred(item):
            acc = (k, item)
        else:
            return acc
    return acc

def interp_thresh(
    x: np.ndarray[(...,), float],
    y: np.ndarray[(...,), float],
    y0: float,
) -> Optional[float]:
    left = find_last(y, lambda yk: yk > y0)
    if left is None:
        return None
    else:
        kmin = left[0]
    right = find_last(y[::-1], lambda yk: yk < y0)
    if right is None:
        return None
    else:
        kmax = len(y) - 1 - right[0]
    xmin = x[kmin]
    xmax = x[kmax]
    ymin = y[kmin]
    ymax = y[kmax]
    return xmin + (xmax - xmin) / (ymax - ymin) * (y0 - ymin)

def find_pstar(data: Processed, threshold: float) -> np.ndarray[(...,), float]:
    avg_slopes = data.fits[:, :, :, 1].mean(axis=0) # :: { p, chi }
    avg_slopes_diff = map_outer(
        lambda slopes_p: slopes_p - slopes_p[-1],
        avg_slopes,
    ) # :: { p, chi }

    pstar = np.array([
        p_ if (
            p_ := interp_thresh(
                data.p_meas[slope_diff > 0],
                np.log10(slope_diff[slope_diff > 0]),
                threshold,
            )
        ) is not None else np.nan
        for slope_diff in avg_slopes_diff[:, :-1].T
    ]) # :: { chi }
    return pstar

def plot_by_threshold(
    proc: list[Processed],
    thresh: float,
    outdir: Path,
) -> None:
    P = pd.Plotter()
    nmin = min(d.size for d in proc)
    nmax = max(d.size for d in proc)
    nrange = nmax - nmin
    cmap = pd.colormaps["vibrant"]
    colors = (
        [f"C{k % 10}" for k in range(len(proc))]
        if nrange == 0
        else [cmap((d.size - nmin) / nrange) for d in proc]
    )
    for (k, (data, color)) in enumerate(zip(proc, colors)):
        pstar = find_pstar(data, thresh)
        P.plot(
            data.chi, pstar,
            marker=".", c=color,
            label=f"$N = {data.size}$",
        )
    (
        P
        .ggrid()
        .llegend()
        .set_xlabel("$\\chi$")
        .set_ylabel("$p^*$")
        .set_title(f"threshold = {thresh:g}")
        .savefig(outdir.joinpath(f"pstar_thresh={thrshold:g}.png"))
        .close()
    )

def plot_by_size(
    proc: Processed,
    thresholds: list[float],
    outdir: Path,
) -> None:
    P = pd.Plotter()
    tlogmin = min(np.log10(t) for t in thresholds)
    tlogmax = max(np.log10(t) for t in thresholds)
    tlogrange = tlogmax - tlogmin
    cmap = pd.colormaps["vibrant"]
    colors = (
        [f"C{k % 10}" for k in range(len(thresholds))]
        if tlogrange == 0
        else [cmap((np.log10(t) - tlogmin) / tlogmax) for t in thresholds]
    )
    for (k, (thresh, color)) in enumerate(zip(thresholds, colors)):
        pstar = find_pstar(proc, thresh)
        P.plot(
            proc.chi, pstar,
            marker=".", c=color,
            label=f"threshold = {thresh:g}",
        )
    (
        P
        .ggrid()
        .llegend()
        .set_xlabel("$\\chi$")
        .set_ylabel("$p^*$")
        .set_title(f"$N = {proc.size}$")
        .savefig(outdir.joinpath(f"pstar_n={n}.png"))
        .close()
    )

def plot_by_chis(
    proc: list[Processed],
    thresh: float,
    outdir: Path,
) -> None:
    if len(proc) == 0:
        return
    chis = set(proc[0].chi)
    for data in proc[1:]:
        chis ^= set(data.chi)
    if len(chi) == 0:
        return
    chis = list(chis)
    pstar_calcs = [find_pstar(data, thresh) for data in proc]
    xmin = min(chis)
    xmax = max(chis)
    xrange = xmax - xmin
    cmap = pd.colormaps["vibrant"]
    colors = [cmap((x - xmin) / xrange) for x in chis]
    P = pd.Plotter()
    Pfit = pd.Plotter()
    for (chi0, color) in zip(chis, colors):
        n = np.array([data.size for data in proc])
        pstar = np.array([
            ps[np.argmin(abs(data.chi - chi0))]
            for (data, ps) in zip(proc, pstar_calcs)
        ])
        pstar_fit = pstar[np.isfinite(pstar)]
        if len(pstar_fit) >= 2:
            over_n_fit = 1 / n[np.isfinite(pstar)]
            b = np.polyfit(over_n_fit, pstar_fit)[::-1]
            over_n_fplot = np.array([0.0, over_n_fit.max()])
            pstar_fplot = b[0] + b[1] * over_n_fplot
            P.plot(over_n_fplot, pstar_fplot, marker="", ls="--", c=color)
            Pfit.plot([chi0], [b[0]], marker="o", ls="", c=color)
        P.plot(
            1 / n, pstar,
            marker=".", ls="", c=color,
            label=f"$\\chi_0 = {chi0}$",
        )
    (
        P
        .ggrid()
        .llegend()
        .set_xlabel("$1 / N$")
        .set_ylabel("$p^*$")
        .set_title(f"threshold = {thresh:g}")
        .savefig(outdir.joinpath(f"pstar_chs_thresh={thesh:g}.png"))
        .close()
    )
    (
        Pfit
        .set_xlabel("$\\chi$")
        .set_ylabel("$p^* @ 1 / N = 0$")
        .set_title(f"threshold = {thresh:g}")
        .savefig(outdir.joinpath("pstar_chs_extrap_thresh={thresh:g}.png"))
        .close()
    )

def main() -> None:
    infile_base = (
        Path("output")
        .joinpath("haar_coev_probs")
        .joinpath("surprise_slope")
    )
    infiles = [
        infile_base.joinpath(f"avg_fits_seed=10546_n={n}_depth={10 * n}.npz")
        for n in [21, 22]
    ]
    proc = [Processed.load(infile) for infile in infiles]
    thresholds = np.logspace(-6, -2, 10)

    for thresh in thresholds:
        plot_by_threshold(proc, thresh, infile_base)
        plot_by_chis(proc, thresh, infile_base)
    for data in proc:
        plot_by_size(data, thresholds, infile_base)

if __name__ == "__main__":
    main()

