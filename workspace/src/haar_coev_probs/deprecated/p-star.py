from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable, Iterator, Optional
import numpy as np
import numpy.linalg as la
try:
    import whooie.pyplotdefs as pd
except ModuleNotFoundError:
    import lablib.plotting.pyplotdefs as pd
from cumulative_avg import (Processed, fname_adjust, map_outer)

def find_last[T](items: Iterable[T], pred: Callable[[T], bool]) -> Optional[T]:
    """
    Find the last item (with its position) of a leading contiguous subsequence
    that satisfies a predicate.
    """
    acc = None
    for item in items:
        if pred(item):
            acc = item
        else:
            return acc
    return acc

def interp_thresh(
    x: np.ndarray[float],
    y: np.ndarray[float],
    y0: float,
) -> Optional[float]:
    """
    Interpolate to find the approximate `x` value for which `y` crosses `y0`.
    """
    kmin = find_last(enumerate(y), lambda pair: pair[1] > y0)
    if kmin is None:
        return None
    else:
        kmin = kmin[0]
    kmax = find_last(enumerate(reversed(y)), lambda pair: pair[1] < y0)
    if kmax is None:
        return None
    else:
        kmax = len(y) - 1 - kmax[0]
    xmin = x[kmin]
    xmax = x[kmax]
    ymin = y[kmin]
    ymax = y[kmax]
    return xmin + (xmax - xmin) / (ymax - ymin) * (y0 - ymin)

def find_pstar(
    infile: Path,
    threshold: float,
) -> (int, np.ndarray[int, 1], np.ndarray[float, 1]):
    """
    Find p* as a function of chi.
    """
    processed = Processed.load(infile)

    avg_slopes = processed.fits[:, :, :, 1].mean(axis=0) # :: { p, chi }
    avg_slopes_err = processed.fits[:, :, :, 1].std(axis=0) # :: { p, chi }
    size = processed.size
    depth = processed.depth
    p_meas = processed.p_meas
    chi = processed.chi

    avg_slopes_diff = map_outer(
        lambda slopes_p: slopes_p - slopes_p[-1],
        avg_slopes,
    ) # :: { p, chi }

    pstar = np.array([
        p_ if (
            p_ := interp_thresh(
                p_meas[slope_diff > 0],
                np.log10(slope_diff[slope_diff > 0]),
                threshold,
            )
        ) is not None else np.nan
        for slope_diff in avg_slopes_diff.T
    ]) # :: { chi }
    return (size, chi, pstar)

def by_threshold(threshold: float):
    outdir = Path("output").joinpath("haar_coev_probs")

    P = pd.Plotter()
    for (k, infile) in enumerate(infiles):
        (size, chi, pstar) = find_pstar(outdir.joinpath(infile), threshold)
        P.plot(
            chi, pstar,
            marker="o", color=f"C{k % 10}", label=f"$N = {size}$"
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
        .set_xlabel("$\\chi$")
        .set_ylabel("$p^*$")
        .set_title(f"Threshold = $10^{{{threshold:g}}}$")
        .savefig(outdir.joinpath(f"pstar_thresh={threshold:g}.png"))
        .close()
    )

def by_infile(infile: str):
    npat = re.compile(r"haar_coev_probs.+n=([0-9]+).+")
    outdir = Path("output").joinpath("haar_coev_probs")
    n = int(npat.match(infile).group(1))

    P = pd.Plotter()
    for (k, thresh) in enumerate(thresholds):
        (size, chi, pstar) = find_pstar(outdir.joinpath(infile), thresh)
        P.plot(
            chi, pstar,
            marker="o", color=f"C{k % 10}",
            label=f"threshold = $10^{{{thresh:g}}}$",
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
        .set_xlabel("$\\chi$")
        .set_ylabel("$p^*$")
        .set_title(f"$N = {n}$")
        .savefig(outdir.joinpath(f"pstar_n={n}.png"))
        .close()
    )

def by_chis(chis: list[int], thresh: float):
    npat = re.compile(r"haar_coev_probs.+n=([0-9]+).+")
    outdir = Path("output").joinpath("haar_coev_probs")
    P = pd.Plotter()
    Pfit = pd.Plotter()
    for (c, chi0) in enumerate(chis):
        n_acc = list()
        pstar_acc = list()
        for (k, infile) in enumerate(infiles):
            n = int(npat.match(infile).group(1))
            (size, chi, pstar) = find_pstar(outdir.joinpath(infile), thresh)
            if chi0 not in chi:
                continue
            k0 = np.argmin(abs(chi - chi0))
            n_acc.append(n)
            pstar_acc.append(pstar[k0])
        n_acc = np.array(n_acc)
        pstar_acc = np.array(pstar_acc)
        pstar_fit = pstar_acc[np.isfinite(pstar_acc)]
        if len(pstar_fit) >= 2:
            over_n_fit = 1 / n_acc[np.isfinite(pstar_acc)]
            A = np.array([np.ones(over_n_fit.shape), over_n_fit]).T
            b = la.solve(A.T @ A, A.T @ pstar_fit)
            over_n_fplot = np.array([0.0, over_n_fit.max()])
            pstar_fplot = b[0] + b[1] * over_n_fplot
            P.plot(
                over_n_fplot, pstar_fplot,
                marker="", linestyle="--", color=f"C{c % 10}",
            )
            Pfit.plot(
                [chi0], [b[0]],
                marker="o", linestyle="", color=f"C{c % 10}",
            )
        P.plot(
            1 / n_acc, pstar_acc,
            marker="o", linestyle="", color=f"C{c % 10}",
            label=f"$\\chi_0 = {chi0}$",
        )

    (
        P
        .set_xlabel("$1 / N$")
        .set_ylabel("$p^*$")
        .set_title(f"$\\mathregular{{Threshold}} = 10^{{{thresh:g}}}$")
        .legend(
            fontsize="xx-small",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=1.0,
        )
        .savefig(outdir.joinpath(f"pstar_chs_thresh={thresh:g}.png"))
        .close()
    )
    (
        Pfit
        .set_xlabel("$\\chi$")
        .set_ylabel("$p^* @ 1 / N = 0$")
        .set_title(f"$\\mathregular{{Threshold}} = 10^{{{thresh:g}}}$")
        .savefig(outdir.joinpath(f"pstar_chs_extrap_thresh={thresh:g}.png"))
        .close()
    )

infiles = [
    # "haar_coev_probs_n=6_d=60_runs=2000_seed=10546_circ=processed.npz",
    # "haar_coev_probs_n=8_d=80_runs=2000_seed=10546_circ=processed.npz",
    # "haar_coev_probs_n=10_d=100_runs=2000_seed=10546_circ=processed.npz",
    # "haar_coev_probs_n=12_d=120_runs=500_seed=10546_circ=processed.npz",
    # "haar_coev_probs_n=14_d=140_runs=500_seed=10546_circ=processed.npz",
    # "haar_coev_probs_seed=10546_n=15_depth=150_circ=processed_id=comb.npz",
    # "haar_coev_probs_seed=10546_n=16_depth=160_circ=processed_id=comb.npz",
    # "haar_coev_probs_seed=10546_n=17_depth=170_circ=processed_id=comb.npz",

    "haar_coev_probs_seed=10546_n=14_depth=140_circ=processed_id=comb.npz",
    "haar_coev_probs_seed=10546_n=15_depth=150_circ=processed_id=comb.npz",
    "haar_coev_probs_seed=10546_n=16_depth=160_circ=processed_id=comb.npz",
    "haar_coev_probs_seed=10546_n=17_depth=170_circ=processed_id=comb.npz",
    "haar_coev_probs_seed=10546_n=18_depth=180_circ=processed_id=comb.npz",
    "haar_coev_probs_seed=10546_n=19_depth=190_circ=processed_id=comb.npz",
    "haar_coev_probs_seed=10546_n=20_depth=200_circ=processed_id=comb.npz",
]
thresholds = [-2, -3, -4, -4.25, -4.5, np.log10(2e-5), -5, -6]
chis = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72]

if __name__ == "__main__":
    for threshold in thresholds:
        by_threshold(threshold)
        by_chis(chis, threshold)
    for infile in infiles:
        by_infile(infile)

