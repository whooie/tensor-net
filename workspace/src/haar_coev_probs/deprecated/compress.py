from pathlib import Path
import sys
from typing import *
import numpy as np

def fname_adjust(
    path: Path,
    adj: Optional[str | Callable[[str], str]],
    suffix: Optional[str],
) -> Path:
    suff = (
        "" if suffix is None
        else ("." + suffix) if not suffix.startswith(".")
        else suffix
    )
    if isinstance(adj, type(None)):
        return path.with_suffix(suff)
    elif isinstance(adj, str):
        return path.with_stem(path.stem + adj).with_suffix(suff)
    elif isinstance(adj, Callable):
        return path.with_stem(adj(path.stem)).with_suffix(suff)
    else:
        raise Exception()

def write_out(infile: Path) -> None:
    data = np.load(str(infile))
    size = data["size"]
    depth = data["depth"]
    circ = data["circ"]
    circs = data["circs"]
    runs = data["runs"]
    p_meas = data["p_meas"]
    chi = data["chi"]
    traj = data["traj"] # :: { p, run, t, x }
    prob = data["prob"] # :: { p, run, chi, t, x }
    seed = data["seed"]

    for (p, traj_p, prob_p) in zip(p_meas, traj, prob):
        meas_selector = np.where(traj_p[0, :, :] != 0)
        meas_locs = np.array(meas_selector).T
        traj_p_flat = np.array([
            traj_p_run[meas_selector] for traj_p_run in traj_p
        ])
        prob_p_flat = np.array([
            [
                prob_p_run_chi[meas_selector] for prob_p_run_chi in prob_p_run
            ] for prob_p_run in prob_p
        ])
        fname = (
            f"haar_coev_probs"
            f"_seed={seed[0]}"
            f"_n={size[0]}"
            f"_depth={depth[0]}"
            f"_circ={circ[0]}"
            f"_p={p:.6f}"
            f"_id=combflat.npz"
        )
        outfile = (
            infile.parent
            .joinpath("flattened")
            .joinpath(f"n={size[0]}")
            .joinpath(f"c={circ[0]}")
            .joinpath(fname)
        )
        np.savez(
            str(outfile),
            size=size,
            depth=depth,
            circ=circ,
            circs=circs,
            runs=runs,
            p=np.array([p]),
            p_meas=p_meas,
            chi=chi,
            meas_locs=meas_locs,
            traj=traj_p_flat,
            prob=prob_p_flat,
            seed=seed,
        )

def main() -> None:
    infile = sys.argv[1]
    write_out(Path(infile))

if __name__ == "__main__":
    main()

