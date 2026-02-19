from pathlib import Path
from typing import Callable, Optional
import numpy as np
import numpy.linalg as la
import whooie.pyplotdefs as pd

def curry(f: Callable, *args):
    return lambda *rest: f(*args, *rest)

def fname_adjust(
    path: Path,
    adj: str | Callable[[str], str],
    suffix: Optional[str],
) -> Path:
    suff = (
        "" if suffix is None
        else ("." + suffix) if not suffix.startswith(".")
        else suffix
    )
    if isinstance(adj, str):
        return path.with_stem(path.stem + adj).with_suffix(suff)
    elif isinstance(adj, Callable):
        return path.with_stem(adj(path.stem)).with_suffix(suff)
    else:
        raise Exception()

def map_outer[T, U](
    f: Callable[[np.ndarray[T]], U],
    a: np.ndarray[T],
    num_axes: int = 1
) -> np.ndarray[U]:
    assert num_axes != 0, "cannot map on zero axes"
    shape = a.shape
    to_map = a.reshape((np.prod(shape[:num_axes]), *shape[num_axes:]))
    mapped = np.array([f(a_k) for a_k in to_map])
    reshape = (*shape[:num_axes], *mapped.shape[1:])
    return mapped.reshape(reshape)

def surprise_t(
    after: int,
    prob_rec: np.ndarray[float, 2],
) -> np.ndarray[float, 1]:
    sel_after = prob_rec[after:, :]
    where = np.where(sel_after > 0.0)
    return np.cumsum(-np.log(sel_after[where]))

def linfit(y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    x = np.arange(y.shape[0])
    A = np.array([np.ones(y.shape), x])
    return la.solve(A @ A.T, A @ y)

def slope(y: np.ndarray[float, 1]) -> float:
    return linfit(y)[1]

def main():
    outdir = Path("output").joinpath("haar_coev_probs")

    # infile = outdir.joinpath("haar_coev_probs_n=6_d=18_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_probs_n=8_d=24_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_probs_n=10_d=30_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_probs_n=12_d=36_runs=5000_seed=10546.npz")
    infile = outdir.joinpath("haar_coev_probs_n=14_d=42_runs=1000_seed=10546.npz")

    plotdir = fname_adjust(infile, "", None)
    plotdir.mkdir(exist_ok=True)

    data = np.load(str(infile))
    size = int(data["size"][0])
    depth = int(data["depth"][0])
    p_meas = data["p_meas"]
    chi = data["chi"]
    traj = data["traj"] # :: { p, run, t, x }
    prob = data["prob"] # :: { p, run, chi, t, x }
    seed = int(data["seed"][0])

    chi = np.roll(chi, -1)
    prob = np.roll(prob, -1, axis=2)

    k_q = np.where(chi == 0)[0][0]
    k_c = np.where(chi != 0)[0]
    assert np.sum(chi == 0) == 1

    t0 = 2 * size
    # t0 = 0
    # surp_t :: { p }{ chi, meas_t }
    surp_t = [
        map_outer(curry(surprise_t, t0), prob_p, num_axes=2).mean(axis=0)
        for prob_p in prob
    ]
    # fits :: { p, chi, poly_param }
    fits = np.array([
        map_outer(linfit, surp_t_p, num_axes=1)
        for surp_t_p in surp_t
    ])

    for (p, surp_t_p, fits_p) in zip(p_meas, surp_t, fits):
        t = np.arange(surp_t_p.shape[1])
        P = pd.Plotter()
        for (k, (x, surp_t_px, fits_px)) in enumerate(zip(chi, surp_t_p, fits_p)):
            (
                P
                .plot(
                    t, surp_t_px,
                    marker=".", ls="-", c=f"C{k % 10}",
                    label=f"$\\chi = {x}$",
                )
                .plot(
                    t, fits_px[0] + fits_px[1] * t,
                    marker="", ls="--", c=f"C{k % 10}",
                    label=f"$a_0 = {fits_px[0]:+.5f}; a_1 = {fits_px[1]:+.5f}$",
                )
            )
        (
            P
            .plot(
                t, surp_t_p[-1],
                marker=".", ls="-", c="k",
                label=f"$\\chi = \\infty$",
            )
            .plot(
                t, fits_p[-1, 0] + fits_p[-1, 1] * t,
                marker="", ls="--", c="k",
                label=f"$a_0 = {fits_p[-1, 0]:+.5f}; a_1 = {fits_p[-1, 1]:+.5f}$",
            )
            .ggrid()
            .legend(
                fontsize="xx-small",
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
                framealpha=1.0,
            )
            .set_xlabel("$t$")
            .set_ylabel(r"$\langle \log P_\chi(t) \rangle_{\vec{b}}$")
            .set_title(f"{size = }; {depth = }\n${p = :.3f}$")
            .savefig(plotdir.joinpath(f"avg_t_{p=:.3f}.png"))
            .close()
        )

    for (p, surp_t_p, fits_p) in zip(p_meas, surp_t, fits):
        t = np.arange(surp_t_p.shape[1])
        P = pd.Plotter()
        for (k, (x, surp_t_px, fits_px)) in enumerate(zip(chi[k_c], surp_t_p[k_c, :], fits_p[k_c, :])):
            rel_fit = fits_px - fits_p[-1, :]
            (
                P
                .plot(
                    t, surp_t_px - surp_t_p[-1, :],
                    marker=".", ls="-", c=f"C{k % 10}",
                    label=f"$\\chi = {x}$",
                )
                .plot(
                    t, rel_fit[0] + rel_fit[1] * t,
                    marker="", ls="--", c=f"C{k % 10}",
                    label=f"$a_0 = {rel_fit[0]:+.5f}; a_1 = {rel_fit[1]:+.5f}$",
                )
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
            .set_xlabel("$t$")
            .set_ylabel(
                r"$\langle \log P_\chi(t) - \log P_\infty(t) \rangle_{\vec{b}}$"
            )
            .set_title(f"{size = }; {depth = }\n${p = :.3f}$")
            .savefig(plotdir.joinpath(f"avg_t_diff_{p=:.3f}.png"))
            .close()
        )

    P = (
        pd.Plotter()
        .colorplot(np.arange(len(chi)), p_meas, fits[:, :, 1])
        .colorbar()
    )
    for p in p_meas:
        P.axhline(p, color="0.85", lw=0.1)
    for x in np.arange(len(chi)):
        P.axvline(x, color="0.85", lw=0.1)
    (
        P
        .ggrid().grid(False, which="both")
        .set_xlabel("$\\chi$")
        .set_xticks(np.arange(len(chi)), ["$\\infty$" if x == 0 else str(x) for x in chi])
        .set_ylabel("$p$")
        .set_clabel(r"Slope of $\langle \log P_\chi(t) \rangle_{\vec{b}}$")
        .set_title(f"{size = }; {depth = }")
        .savefig(plotdir.joinpath("avg_t_slopes.png"))
        .close()
    )

if __name__ == "__main__":
    main()

