from pathlib import Path
from typing import Callable, Optional
import numpy as np
try:
    import whooie.pyplotdefs as pd
except ModuleNotFoundError:
    import lablib.plotting.pyplotdefs as pd

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

def main():
    outdir = Path("output").joinpath("haar_coev_entropy")

    infile = outdir.joinpath("haar_coev_entropy_n=6_d=24_runs=500_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_entropy_n=8_d=32_runs=500_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_entropy_n=10_d=40_runs=500_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_entropy_n=12_d=48_runs=500_seed=10546.npz")

    data = np.load(str(infile))
    size = int(data["size"][0])
    depth = int(data["depth"][0])
    circs = int(data["circs"][0])
    runs = int(data["runs"][0])
    p_meas = data["p_meas"]
    chi = data["chi"]
    entropy = data["entropy"] # :: { circ, p, run, chi }
    seed = int(data["seed"][0])

    assert np.sum(chi == 0) == 1
    rot = len(chi) - np.argmin(chi) - 1
    chi = np.roll(chi, rot)
    entropy = np.roll(entropy, rot)
    avg_entropy = entropy.mean(axis=(0, 2)) # :: { p, chi }
    avg_entropy_diff = np.array([row - row[-1] for row in avg_entropy])

    P = (
        pd.Plotter()
        .colorplot(np.arange(len(chi)), p_meas, avg_entropy)
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
            r"\left\langle "
            r"S_{N / 2}^\mathregular{vn} "
            r"\right\rangle_{\vec{b}, \vec{U}} "
            "$"
        )
        .set_title(f"{size = }; {depth = }")
        .savefig(fname_adjust(infile, lambda stem: stem, "png"))
        .close()
    )

    P = (
        pd.Plotter()
        .colorplot(np.arange(len(chi)), p_meas, avg_entropy_diff)
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
        .savefig(fname_adjust(infile, lambda stem: stem + "_diff", "png"))
        .close()
    )

    pd.pp.rcParams["savefig.pad_inches"] = 0.3
    chi_mat, p_meas_mat = np.meshgrid(np.arange(len(chi)), p_meas)
    entropy_range = abs(avg_entropy.max() - avg_entropy.min())
    contour_z_level = avg_entropy.min() - entropy_range / 3
    (
        pd.Plotter.new_3d()
        .plot_surface(chi_mat, p_meas_mat, avg_entropy, alpha=0.75)
        .contourf(
            chi_mat, p_meas_mat, avg_entropy,
            zdir="z",
            offset=contour_z_level,
            cmap="vibrant"
        )
        .view_init(azim=200)
        .set_xlabel("$\\chi$")
        .set_xticks(
            np.arange(len(chi)),
            ["$\\infty$" if x == 0 else str(x) for x in chi],
        )
        .set_ylabel("$p$")
        .set_zlabel(
            "$"
            r"\left\langle "
            r"S_{N / 2}^\mathregular{vn} "
            r"\right\rangle_{\vec{b}, \vec{U}} "
            "$"
        )
        .set_zlim(bottom=contour_z_level)
        .set_title(f"{size = }; {depth = }")
        .savefig(fname_adjust(infile, lambda stem: stem + "_3d", "png"))
        .close()
    )

    entropy_range = abs(avg_entropy_diff.max() - avg_entropy_diff.min())
    contour_z_level = avg_entropy_diff.min() - entropy_range / 3
    (
        pd.Plotter.new_3d()
        .plot_surface(chi_mat, p_meas_mat, avg_entropy_diff, alpha=0.75)
        .contourf(
            chi_mat, p_meas_mat, avg_entropy_diff,
            zdir="z",
            offset=contour_z_level,
            cmap="vibrant"
        )
        .view_init(azim=200)
        .set_xlabel("$\\chi$")
        .set_xticks(
            np.arange(len(chi)),
            ["$\\infty$" if x == 0 else str(x) for x in chi],
        )
        .set_ylabel("$p$")
        .set_zlabel("Diff. from $\\chi = \\infty$")
        .set_zlim(bottom=contour_z_level)
        .set_title(f"{size = }; {depth = }")
        .savefig(fname_adjust(infile, lambda stem: stem + "_diff_3d", "png"))
        .close()
    )

if __name__ == "__main__":
    main()
