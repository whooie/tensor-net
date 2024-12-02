from math import prod
from pathlib import Path
from typing import Callable
import numpy as np
import whooie.pyplotdefs as pd
FS = pd.pp.rcParams["figure.figsize"]

outdir = Path("output")
# infile = outdir.joinpath("naive_n=12_d=24_circs=50_avg=4000.npz")
# infile = outdir.joinpath("naive_n=10_d=20_circs=50_avg=10000.npz")
# infile = outdir.joinpath("naive_n=10_d=20_circs=50_runs=2000.npz")
# infile = outdir.joinpath("naive_n=10_d=20_circs=20_runs=500.npz")
# infile = outdir.joinpath("naive_n=20_d=40_circs=20_runs=500.npz")
infile = outdir.joinpath("naive_n=10_d=20_circs=50_runs=500.npz")
data = np.load(str(infile))

size = int(data["size"][0])
depth = int(data["depth"][0])
circs = int(data["circs"][0])
p_meas = data["p_meas"]
chi = data["chi"]
dt = int(data["dt"][0])
target_x = data["target_x"].astype(int)
P_qc = data["dists"] # :: { circ, p, chi, dist }

check = np.abs(P_qc.sum(axis=3) - 1).sum()
assert check < 1e-9, f"something weird with probabilities: {check}"

def kl(p: np.ndarray[float, 1], q: np.ndarray[float, 1]) -> float:
    """
    KL divergence, KL(p||q).
    """
    assert p.shape == q.shape, \
        f"kl: incompatible array shapes {p.shape}, {q.shape}"
    return sum(
        pk * np.log(pk / qk)
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

P = pd.Plotter.new(
    nrows=3,
    sharex=True,
    figsize=[FS[0], 1.5 * FS[1]],
    as_plotarray=True,
)
for (x, kl_x) in zip(chi[k_c], data_kl):
    # print(kl_x)
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
    P[1].plot(p_meas, np.log(4) - sh_x, marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[1] \
    .plot(p_meas, np.log(4) - data_sh[k_q], marker=".", linestyle="-", color="k", label=f"$\\chi = \\infty$") \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$\\log 4 - H(P_\\chi)$")
for (x, sh_x) in zip(chi[k_c], data_sh[k_c]):
    P[2].semilogy(p_meas, abs(sh_x - data_sh[k_q]), marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[2] \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$|H(C_\\chi) - H(Q)|$") \
    .set_xlabel("$p$")
P \
    .suptitle(

        f"$\\mathregular{{circs}} = {circs}; \\mathregular{{size}} = {size}; \\mathregular{{depth}} = {depth}$\n"
        f"$\\Delta t = {dt}; x_0 = {target_x[0]}; x_1 = {target_x[1]}$",
        fontsize="small",
    ) \
    .savefig(infile.with_suffix(".png")) \
    .close()

probs = P_qc.mean(axis=0).swapaxes(0, 1) # :: { chi, p, dist }
P = pd.Plotter.new(
    nrows=4,
    sharex=True,
    figsize=[FS[0], 1.5 * FS[1]],
    as_plotarray=True,
)
for (x, p00) in zip(chi[k_c], probs[k_c, :, 0]):
    P[0].plot(p_meas, p00, marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[0] \
    .plot(p_meas, probs[k_q, :, 0], marker=".", linestyle="-", color="k", label="$\\chi = \\infty$") \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$P_\\chi(00)$")
for (x, p01) in zip(chi[k_c], probs[k_c, :, 1]):
    P[1].plot(p_meas, p01, marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[1] \
    .plot(p_meas, probs[k_q, :, 1], marker=".", linestyle="-", color="k", label="$\\chi = \\infty$") \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$P_\\chi(01)$")
for (x, p10) in zip(chi[k_c], probs[k_c, :, 2]):
    P[2].plot(p_meas, p10, marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[2] \
    .plot(p_meas, probs[k_q, :, 2], marker=".", linestyle="-", color="k", label="$\\chi = \\infty$") \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$P_\\chi(10)$")
for (x, p11) in zip(chi[k_c], probs[k_c, :, 3]):
    P[3].plot(p_meas, p11, marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[3] \
    .plot(p_meas, probs[k_q, :, 3], marker=".", linestyle="-", color="k", label="$\\chi = \\infty$") \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$P_\\chi(11)$") \
    .set_xlabel("$p$")
P \
    .suptitle(

        f"$\\mathregular{{circs}} = {circs}; \\mathregular{{size}} = {size}; \\mathregular{{depth}} = {depth}$\n"
        f"$\\Delta t = {dt}; x_0 = {target_x[0]}; x_1 = {target_x[1]}$",
        fontsize="small",
    ) \
    .savefig(infile.with_stem(infile.stem + "_hist").with_suffix(".png")) \
    .close()

P = pd.Plotter.new(
    nrows=4,
    sharex=True,
    figsize=[FS[0], 1.5 * FS[1]],
    as_plotarray=True,
)
p00q = probs[k_q, :, 0]
for (x, p00) in zip(chi[k_c], probs[k_c, :, 0]):
    P[0].semilogy(p_meas, abs(p00 - p00q), marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[0] \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$|P_\\chi(00) - Q(00)|$", fontsize="x-small")
p01q = probs[k_q, :, 1]
for (x, p01) in zip(chi[k_c], probs[k_c, :, 1]):
    P[1].semilogy(p_meas, abs(p01 - p01q), marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[1] \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$|P_\\chi(01) - Q(01)|$", fontsize="x-small")
p10q = probs[k_q, :, 2]
for (x, p10) in zip(chi[k_c], probs[k_c, :, 2]):
    P[2].semilogy(p_meas, abs(p10 - p10q), marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[2] \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$|P_\\chi(10) - Q(10)|$", fontsize="x-small")
p11q = probs[k_q, :, 3]
for (x, p11) in zip(chi[k_c], probs[k_c, :, 3]):
    P[3].semilogy(p_meas, abs(p11 - p11q), marker=".", linestyle="-", label=f"$\\chi = {x}$")
P[3] \
    .ggrid() \
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    ) \
    .set_ylabel("$|P_\\chi(11) - Q(11)|$", fontsize="x-small") \
    .set_xlabel("$p$")
P \
    .suptitle(

        f"$\\mathregular{{circs}} = {circs}; \\mathregular{{size}} = {size}; \\mathregular{{depth}} = {depth}$\n"
        f"$\\Delta t = {dt}; x_0 = {target_x[0]}; x_1 = {target_x[1]}$",
        fontsize="small",
    ) \
    .savefig(infile.with_stem(infile.stem + "_histdiff").with_suffix(".png")) \
    .close()

