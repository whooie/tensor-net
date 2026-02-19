from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional
import numpy as np
import numpy.linalg as la
try:
    import whooie.pyplotdefs as pd
except ModuleNotFoundError:
    import lablib.plotting.pyplotdefs as pd

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
    if num_axes == 1:
        return np.array([f(a_k) for a_k in a])
    else:
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
    fits: np.ndarray[float, 4] # :: { circ, p_meas, chi, linfit_param }

def process_single(infile: Path, d0: int) -> Optional[np.ndarray[float, 3]]:
    print(infile)
    data = np.load(str(infile))
    size = int(data["size"][0])
    chi = data["chi"]
    traj = data["traj"] # :: { p, run, d, x }
    prob = data["prob"] # :: { p, run, chi, d, x }

    # rotate arrays so that chi == 0 (inf) is last position
    # assume that chi == 0 exists uniquely
    assert np.sum(chi == 0) == 1
    rot = len(chi) - np.argmin(chi) - 1
    chi = np.roll(chi, rot)
    prob = np.roll(prob, rot, axis=2)

    if np.any(abs(traj[:, :, d0:, :]).sum(axis=(2, 3)) < 2):
        print("discarding due to too few measurements")
        return None

    k_q = np.where(chi == 0)[0][0]
    k_c = np.where(chi != 0)[0]

    # surp_t :: { p }{ chi, meas_t }
    surp_t = [
        map_outer(curry(surprise_t, d0), prob_p, num_axes=2).mean(axis=0)
        for prob_p in prob
    ]
    del prob
    # fits :: { p, chi, linfit_param }
    fits = np.array([
        map_outer(linfit, surp_t_p, num_axes=1)
        for surp_t_p in surp_t
    ])
    return fits

def process_set(indir: Path, file_fmt: str) -> Processed:
    data_first = np.load(str(indir.joinpath(file_fmt.format(0))))
    num_circs = int(data_first["circs"][0])
    seed = int(data_first["seed"][0])
    size = int(data_first["size"][0])
    depth = int(data_first["depth"][0])
    p_meas = data_first["p_meas"]
    chi = data_first["chi"]

    # rotate arrays so that chi == 0 (inf) is last position
    # assume that chi == 0 exists uniquely
    assert np.sum(chi == 0) == 1
    rot = len(chi) - np.argmin(chi) - 1
    chi = np.roll(chi, rot)

    d0 = 4 * size
    assert depth > d0
    fits = np.array([
        f for c in range(num_circs)
        if (f := process_single(indir.joinpath(file_fmt.format(c)), d0))
        is not None
    ])
    return Processed(seed, size, depth, d0, p_meas, chi, fits)

def main():
    outdir = Path("output").joinpath("haar_coev_probs")

    # infile_fmt = "haar_coev_probs_n=6_d=60_runs=2000_seed=10546_circ={}.npz"
    # infile_fmt = "haar_coev_probs_n=8_d=80_runs=2000_seed=10546_circ={}.npz"
    # infile_fmt = "haar_coev_probs_n=10_d=100_runs=2000_seed=10546_circ={}.npz"
    infile_fmt = "haar_coev_probs_n=12_d=120_runs=500_seed=10546_circ={}.npz"

    processed = process_set(outdir, infile_fmt)
    avg_slopes = processed.fits[:, :, :, 1].mean(axis=0)
    avg_slopes_err = processed.fits[:, :, :, 1].std(axis=0)
    size = processed.size
    depth = processed.depth
    p_meas = processed.p_meas
    chi = processed.chi

    chi_mat, p_meas_mat = np.meshgrid(np.arange(len(chi)), p_meas)
    avg_slopes_diff = map_outer(
        lambda slopes_p: slopes_p - slopes_p[-1],
        avg_slopes,
    )

    P = (
        pd.Plotter()
        .colorplot(np.arange(len(chi)), p_meas, avg_slopes)
        .colorbar()
        .contour(chi_mat, p_meas_mat, avg_slopes, colors="k")
    )
    for p in p_meas:
        P.axhline(p, color="0.85", lw=0.1)
    for x in np.arange(len(chi)):
        P.axvline(x, color="0.85", lw=0.1)
    (
        P
        .ggrid().grid(False, which="both")
        .set_xlabel("$\\chi$")
        .set_xticks(
            np.arange(len(chi)),
            ["$\\infty$" if x == 0 else str(x) for x in chi],
        )
        .set_ylabel("$p$")
        .set_clabel(
            "$"
            r"\left\langle "
            r"\mathregular{Slope}~\mathregular{of}~ "
            r"\langle \log P_\chi(t) \rangle_{\vec{b}} "
            r"\right\rangle_{\vec{U}} "
            "$"
        )
        .set_title(f"{size = }; {depth = }")
        .savefig(outdir.joinpath(infile_fmt.format("avg")).with_suffix(".png"))
        .close()
    )

    P = (
        pd.Plotter()
        .colorplot(np.arange(len(chi)), p_meas, avg_slopes_err)
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
        .set_xticks(
            np.arange(len(chi)),
            ["$\\infty$" if x == 0 else str(x) for x in chi],
        )
        .set_ylabel("$p$")
        .set_clabel(
            "$"
            r"\mathregular{StDev}~\mathregular{of}~ "
            r"\left\langle "
            r"\mathregular{Slope}~\mathregular{of}~ "
            r"\langle \log P_\chi(t) \rangle_{\vec{b}} "
            r"\right\rangle_{\vec{U}} "
            "$"
        )
        .set_title(f"{size = }; {depth = }")
        .savefig(
            fname_adjust(
                outdir.joinpath(infile_fmt.format("avg")),
                lambda stem: stem + "_err",
                "png",
            )
        )
        .close()
    )

    P = pd.Plotter()
    for (k, (x, slopes_x)) in enumerate(zip(chi, avg_slopes.T)):
        P.semilogy(
            p_meas, slopes_x,
            marker=".", linestyle="-", color=f"C{k % 10}",
            label=f"$\\chi = {x}$",
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
        .set_ylabel(
            "$"
            r"\left\langle "
            r"\mathregular{Slope}~\mathregular{of}~ "
            r"\langle \log P_\chi(t) \rangle_{\vec{b}} "
            r"\right\rangle_{\vec{U}} "
            "$"
        )
        .set_title(f"{size = }; {depth = }")
        .savefig(
            fname_adjust(
                outdir.joinpath(infile_fmt.format("avg")),
                lambda stem: stem + "_lines_log",
                "png",
            )
        )
        .close()
    )

    P = (
        pd.Plotter()
        .colorplot(np.arange(len(chi)), p_meas, avg_slopes_diff)
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
        .set_xticks(
            np.arange(len(chi)),
            ["$\\infty$" if x == 0 else str(x) for x in chi],
        )
        .set_ylabel("$p$")
        .set_clabel("Diff. from $\\chi = \\infty$")
        .set_title(f"{size = }; {depth = }")
        .savefig(
            fname_adjust(
                outdir.joinpath(infile_fmt.format("avg")),
                lambda stem: stem + "_diff",
                "png",
            )
        )
        .close()
    )

    P = pd.Plotter()
    for (k, (x, slopes_diff_x)) in enumerate(zip(chi, avg_slopes_diff.T)):
        P.plot(
            p_meas, slopes_diff_x,
            marker=".", linestyle="-", color=f"C{k % 10}",
            label=f"$\\chi = {x}$",
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
        .set_ylabel("Diff. from $\\chi = \\infty$")
        .set_title(f"{size = }; {depth = }")
        .savefig(
            fname_adjust(
                outdir.joinpath(infile_fmt.format("avg")),
                lambda stem: stem + "_diff_lines",
                "png",
            )
        )
        .close()
    )

    P = pd.Plotter()
    for (k, (x, slopes_diff_x)) in enumerate(zip(chi, avg_slopes_diff.T)):
        P.semilogy(
            p_meas, slopes_diff_x,
            marker=".", linestyle="-", color=f"C{k % 10}",
            label=f"$\\chi = {x}$",
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
        .set_ylabel("Diff. from $\\chi = \\infty$")
        .set_title(f"{size = }; {depth = }")
        .savefig(
            fname_adjust(
                outdir.joinpath(infile_fmt.format("avg")),
                lambda stem: stem + "_diff_lines_log",
                "png",
            )
        )
        .close()
    )

    pd.pp.rcParams["savefig.pad_inches"] = 0.3
    slope_range = avg_slopes.max() - avg_slopes.min()
    contour_z_level = avg_slopes.min() - slope_range / 3
    (
        pd.Plotter.new_3d()
        .plot_surface(chi_mat, p_meas_mat, avg_slopes, alpha=0.75)
        .contourf(
            chi_mat, p_meas_mat, avg_slopes,
            zdir="z",
            offset=contour_z_level,
            cmap="vibrant"
        )
        .view_init(azim=20)
        .set_xlabel("$\\chi$")
        .set_xticks(
            np.arange(len(chi) - 1),
            ["$\\infty$" if x == 0 else str(x) for x in chi[:-1]],
        )
        .set_ylabel("$p$")
        .set_zlabel(
            "$"
            r"\left\langle "
            r"\mathregular{Slope}~\mathregular{of}~ "
            r"\langle \log P_\chi(t) \rangle_{\vec{b}} "
            r"\right\rangle_{\vec{U}} "
            "$"
        )
        .set_zlim(bottom=contour_z_level)
        .set_title(f"{size = }; {depth = }")
        .savefig(
            fname_adjust(
                outdir.joinpath(infile_fmt.format("avg")),
                lambda stem: stem + "_3d",
                "png",
            )
        )
        .close()
    )

    slope_range = avg_slopes_diff.max() - avg_slopes_diff.min()
    contour_z_level = avg_slopes_diff.min() - slope_range / 3
    (
        pd.Plotter.new_3d()
        .plot_surface(
            chi_mat, p_meas_mat, avg_slopes_diff,
            alpha=0.75
        )
        .contourf(
            chi_mat, p_meas_mat, avg_slopes_diff,
            zdir="z",
            offset=contour_z_level,
            cmap="vibrant"
        )
        .view_init(azim=20)
        .set_xlabel("$\\chi$")
        .set_xticks(
            np.arange(len(chi)),
            ["$\\infty$" if x == 0 else str(x) for x in chi],
        )
        .set_ylabel("$p$")
        .set_zlabel("Diff. from $\\chi = \\infty$")
        .set_zlim(bottom=contour_z_level)
        .set_title(f"{size = }; {depth = }")
        .savefig(
            fname_adjust(
                outdir.joinpath(infile_fmt.format("avg")),
                lambda stem: stem + "_diff_3d",
                "png",
            )
        )
        .close()
    )

if __name__ == "__main__":
    main()

