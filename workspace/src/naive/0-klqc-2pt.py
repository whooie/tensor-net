from math import prod
from pathlib import Path
from typing import Callable
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
<<<<<<< Updated upstream
# infile = outdir.joinpath("fixed_conserv_exact_n=15_d=30_mc=150.npz")
# infile = outdir.joinpath("fixed_conserv_exact_n=10_d=20_mc=1.npz")
# infile = outdir.joinpath("fixed_conserv_exact_n=10_d=20_mc=20.npz")
# infile = outdir.joinpath("fixed_conserv_exact_n=10_d=20_mc=50.npz")
infile = outdir.joinpath("fixed_conserv_exact_n=11_d=22_mc=50.npz")
=======
infile = outdir.joinpath("fixed_conserv_exact_n=10_d=20_circs=25_avg=100.npz")
# infile = outdir.joinpath("fixed_conserv_exact_n=15_d=30_circs=25_avg=30.npz")
>>>>>>> Stashed changes
data = np.load(str(infile))

size = int(data["size"][0])
depth = int(data["depth"][0])
circs = int(data["circs"][0])
p_meas = data["p_meas"]
chi = data["chi"]
dt = int(data["dt"][0])
target_x = data["target_x"].astype(int)
P_qc = data["dists"].mean(axis=3) # :: { circ, p, chi, dist }

check = np.abs(P_qc.sum(axis=3) - 1).sum()
assert check < 1e-9, f"something weird with probabilities: {check}"

def kl(p: np.ndarray[float, 1], q: np.ndarray[float, 1]) -> float:
    """
    KL divergence, KL(p||q).
    """
    assert p.shape == q.shape, \
        f"kl: incompatible array shapes {p.shape}, {q.shape}"
    return sum(
        pk * (np.log(pk) - np.log(qk))
        for (pk, qk) in zip(p, q)
        if pk > 0 and qk > 0
    )

def sh(p: np.ndarray[float, 1]) -> float:
    """
    Shannon entropy, H(p).
    """
    assert p.ndim == 1, \
        f"sh: expected one dimension, got {p.ndim}"
    return sum(-pk * np.log(pk) for pk in p if pk > 0)

k_q = np.where(chi == 0)[0][0]
k_c = np.where(chi != 0)[0]
print(k_q, k_c)

P_q = P_qc[:, :, k_q, :] # :: { circ, p, dist }
P_c = P_qc[:, :, k_c, :].swapaxes(1, 2).swapaxes(0, 1) # :: { chi, circ, p, dist }

P_q_flat = P_q.reshape((prod(P_q.shape[:2]), P_q.shape[2])) # :: { circ <> p, dist }
P_c_flat = P_c.reshape((P_c.shape[0], prod(P_c.shape[1:3]), P_c.shape[3])) # :: { chi, circ <> p, dist }
data_kl = np.array([
    [kl(c_xcp, q_cp) for (c_xcp, q_cp) in zip(c_x, P_q_flat)]
    for c_x in P_c_flat
]).reshape(P_c.shape[:3]).mean(axis=1) # :: { chi, p }

P_qc_flat = P_qc.reshape((prod(P_qc.shape[:3]), P_qc.shape[3])) # :: { circ <> p <> chi, dist }
data_sh = np.array([sh(P_qc_cpx) for P_qc_cpx in P_qc_flat]) \
    .reshape(P_qc.shape[:3]) \
    .swapaxes(1, 2).swapaxes(0, 1) \
    .mean(axis=1) # :: { chi, p }

P = pd.Plotter.new(nrows=2, sharex=True, as_plotarray=True)

for (x, kl_x) in zip(chi[k_c], data_kl):
    P[0].semilogy(p_meas, kl_x, marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[0] \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$\\mathregular{KL}(C_\\chi || Q)$")

for (x, sh_x) in zip(chi[k_c], data_sh[k_c]):
    P[1].plot(p_meas, sh_x, marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[1] \
    .plot(p_meas, data_sh[k_q], marker=".", linestyle="-", color="k", label=f"$\\chi = \\infty$") \
    .axhline(np.log(4), linestyle="--", color="0.5") \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$H(P_\\chi)$") \
    .set_xlabel("$p$")

P \
    .suptitle(

        f"$\\mathregular{{circs}} = {circs}; \\mathregular{{size}} = {size}; \\mathregular{{depth}} = {depth}$\n"
        f"$\\Delta t = {dt}; x_0 = {target_x[0]}; x_1 = {target_x[1]}$",
        fontsize="small",
    ) \
    .savefig(infile.with_suffix(".png")) \
    .close()

