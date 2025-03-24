from pathlib import Path
from typing import Callable
import numpy as np
import numpy.linalg as la
import whooie.pyplotdefs as pd

def fname_adjust(
    path: Path,
    adj: str | Callable[[str], str],
    suffix: str,
) -> Path:
    suff = ("." + suffix) if not suffix.startswith(".") else suffix
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

def do_avg_div(prob_rec: np.ndarray[float, 2]) -> float:
    return np.mean(-np.log(prob_rec[np.where(prob_rec > 0.0)]))

def main():
    outdir = Path("output").joinpath("haar_coev_probs")
    # infile = outdir.joinpath("haar_coev_probs_n=6_d=18_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_probs_n=8_d=24_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_probs_n=10_d=30_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_coev_probs_n=12_d=36_runs=5000_seed=10546.npz")
    infile = outdir.joinpath("haar_coev_probs_n=14_d=42_runs=1000_seed=10546.npz")

    data = np.load(str(infile))
    size = int(data["size"][0])
    depth = int(data["depth"][0])
    p_meas = data["p_meas"]
    chi = data["chi"]
    traj = data["traj"] # :: { p, run, t, x }
    prob = data["prob"] # :: { p, run, chi, t, x }
    seed = int(data["seed"][0])

    k_q = np.where(chi == 0)[0][0]
    k_c = np.where(chi != 0)[0]
    print(k_q, k_c)

    # avg_div :: { chi, p }
    avg_div = map_outer(do_avg_div, prob, num_axes=3).mean(axis=1).swapaxes(0, 1)

    P = pd.Plotter()
    for (k, (x, div_x)) in enumerate(zip(chi[k_c], avg_div[k_c, :])):
        P.plot(
            p_meas, div_x,
            marker=".", ls="-", c=f"C{k % 10}",
            label=f"$\\chi = {x}$",
        )
    (
        P
        .plot(
            p_meas, avg_div[k_q, :],
            marker=".", ls="-", c="k",
            label=f"$\\chi = \\infty$",
        )
        .ggrid()
        .legend(
            fontsize="xx-small",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=1.0,
        )
        .set_xlabel("$p$")
        .set_ylabel("$\\langle \\log p_\\chi(\\vec{b}) \\rangle_\\vec{b}$")
        .set_title(
            f"{size = }"
            f"; {depth = }"
        )
        .savefig(fname_adjust(infile, "_avg_div", "png"))
        .close()
    )

    P = pd.Plotter()
    div_q = avg_div[k_q, :]
    for (k, (x, div_x)) in enumerate(zip(chi[k_c], avg_div[k_c, :])):
        P.plot(
            p_meas, div_x - div_q,
            marker=".", ls="-", c=f"C{k % 10}",
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
            "\\langle \\log p_\\chi(\\vec{b}) \\rangle_\\vec{b}"
            "- \\langle \\log p_\\infty(\\vec{b}) \\rangle_\\vec{b}"
            "$"
        )
        .set_title(
            f"{size = }"
            f"; {depth = }"
        )
        .savefig(fname_adjust(infile, "_avg_div_diff", "png"))
        .close()
    )

    p_idx = 16
    prob_p = prob[p_idx, :, :, :, :] # :: { run, chi, x, t }
    shape = prob_p.shape
    prob_p_flat = prob_p.reshape((shape[0] * shape[1], shape[2] * shape[3]))
    prob_p_flat_sel = np.array([
        -np.log(sel[np.where(sel > 0.0)])
        for sel in prob_p_flat
    ])
    post_shape = prob_p_flat_sel.shape
    cumdiv = prob_p_flat_sel.reshape((shape[0], shape[1], post_shape[-1])).cumsum(axis=-1).mean(axis=0)
    # cumdiv :: { chi, meas_t }

    meas_t = np.arange(cumdiv.shape[1])
    A = np.array([np.ones(meas_t.shape), meas_t]).T
    cumdiv_inf = cumdiv[0, :]
    b = la.solve(A.T @ A, A.T @ cumdiv_inf)
    print(b)

    P = pd.Plotter()
    for (k, (x, cumdiv_x)) in enumerate(zip(chi[k_c], cumdiv[k_c, :])):
        P.plot(
            cumdiv_x,
            marker=".", ls="-", c=f"C{k % 10}",
            label=f"$\\chi = {x}$",
        )
    (
        P
        .plot(
            cumdiv[k_q, :],
            marker=".", ls="-", c="k",
            label=f"$\\chi = \\infty$",
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
        .set_ylabel(
            "$"
            "\\langle -\\sum_k^t \\log p_\\chi(b_k) \\rangle_\\vec{b}"
            "$"
        )
        .set_title(
            f"{size = }"
            f"; {depth = }"
        )
        .savefig(fname_adjust(infile, f"_avg_t_p={p_meas[p_idx]:.3f}", "png"))
        .close()
    )

    P = pd.Plotter()
    for (k, (x, cumdiv_x)) in enumerate(zip(chi[k_c[1:]], cumdiv[k_c[1:], :])):
        P.plot(
            cumdiv_x - cumdiv[k_q, :],
            marker=".", ls="-", c=f"C{k % 10}",
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
        .set_xlabel("$t$")
        .set_ylabel(
            "$"
            "\\langle -\\sum_k^t \\log p_\\chi(b_k) \\rangle_\\vec{b}"
            "- \\langle -\\sum_k^t \\log p_\\infty(b_k) \\rangle_\\vec{b}"
            "$"
        )
        .set_title(
            f"{size = }"
            f"; {depth = }"
        )
        .savefig(fname_adjust(infile, f"_avg_t_diff_p={p_meas[p_idx]:.3f}", "png"))
        .close()
    )

if __name__ == "__main__":
    main()

