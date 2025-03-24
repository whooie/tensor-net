from pathlib import Path
from typing import Callable
import numpy as np
import numpy.linalg as la
import whooie.pyplotdefs as pd

def curry(f: Callable, *args: ...) -> Callable:
    return lambda *rem: f(*args, *rem)

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

def logm(a: np.ndarray[complex, 2]) -> np.ndarray[complex, 2]:
    assert a.ndim == 2 and a.shape[0] == a.shape[1], \
        "encountered non-square matrix"
    (e, v) = la.eigh(a)
    return v.conj().T @ np.diag(np.log(e)) @ v

def entropy(alpha: float, rho: np.ndarray[complex, 2]) -> float:
    if alpha == 1:
        return -np.diag(rho @ logm(rho)).sum().real
    else:
        return np.log(np.diag(rho ** alpha).sum().real) / (1 - alpha)

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
        raise Exception

def main():
    outdir = Path("output").joinpath("haar_state_gen")
    # infile = outdir.joinpath("haar_state_gen_n=6_d=12_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_state_gen_n=8_d=16_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_state_gen_n=10_d=20_runs=5000.npz")
    # infile = outdir.joinpath("haar_state_gen_n=10_d=20_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_state_gen_n=10_d=20_runs=5000_seed=10547.npz")
    # infile = outdir.joinpath("haar_state_gen_n=10_d=20_runs=5000_seed=10548.npz")
    # infile = outdir.joinpath("haar_state_gen_n=12_d=24_runs=5000_seed=10546.npz")
    # infile = outdir.joinpath("haar_state_gen_n=14_d=28_runs=500_seed=10546.npz")
    infile = outdir.joinpath("haar_state_gen_n=14_d=28_runs=1500_seed=10546.npz")

    data = np.load(str(infile))
    size = int(data["size"][0])
    depth = int(data["depth"][0])
    p_meas = data["p_meas"]
    chi = data["chi"]
    # target_x = int(data["target_x"][0])
    target_range = tuple(data["target_range"])
    subsize = target_range[1] - target_range[0]
    rho = data["rho"] # :: { p, chi, run, i, j }
    # meas = data["meas"] # :: { p, chi, run, t, x }

    k_q = np.where(chi == 0)[0][0]
    k_c = np.where(chi != 0)[0]
    rho = rho.mean(axis=2).swapaxes(0, 1) # :: { chi, p, i, j }
    alpha = 2
    s_a = curry(entropy, alpha)
    ent = map_outer(s_a, rho, num_axes=2)
    P = pd.Plotter()
    for (k, (x, ent_x)) in enumerate(zip(chi[k_c], ent[k_c, :])):
        P.plot(
            p_meas, ent_x,
            marker=".", ls="-", c=f"C{k % 10}", label=f"$\\chi = {x}$"
        )
    (
        P
        .plot(
            p_meas, ent[k_q, :],
            marker=".", ls="-", c="k", label=f"$\\chi = \\infty$",
        )
        .axhline(
            np.log(2 ** subsize),
            ls="--", c="0.5", label=f"$\\log 2^{subsize}$",
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
        .set_ylabel(f"$S_{{\\alpha}}(\\langle \\rho_{{U, b}}^{{\\chi}} \\rangle_b)$")
        .set_title(
            f"{size = }"
            f"; {depth = }"
            f"\n target = {target_range[0]}..{target_range[1] - 1}"
            f"; $\\alpha$ = {alpha}"
        )
        .savefig(fname_adjust(infile, f"_entropy_{alpha=:.1f}", "png"))
        .close()
    )

if __name__ == "__main__":
    main()

