from math import prod
from pathlib import Path
from typing import Callable
import numpy as np
import whooie.pyplotdefs as pd
FS = pd.pp.rcParams["figure.figsize"]

outdir = Path("output")
infile = outdir.joinpath("clifford_mixing_n=10_d=20_circs=50_runs=5000.npz")
data = np.load(str(infile))

size = int(data["size"][0])
depth = int(data["depth"][0])
circs = int(data["circs"][0])
p_meas = data["p_meas"]
chi = data["chi"]
dt = int(data["dt"][0])
target_x = int(data["target_x"][0])
target_range = data["target_range"].astype(int)
tr_qc = data["tr"] # :: { circ, p, chi }

nqubits = target_range[1] - target_range[0]
tr_mixed = 0.5 ** nqubits

k_q = np.where(chi == 0)[0][0]
k_c = np.where(chi != 0)[0]
print(k_q, k_c)

tr_q = tr_qc[:, :, k_q].mean(axis=0) # :: { p }
tr_c = tr_qc[:, :, k_c].mean(axis=0).swapaxes(0, 1) # :: { chi, p }

P = pd.Plotter.new()
for (x, tr_x) in zip(chi[k_c], tr_c):
    P.plot(p_meas, tr_x, marker=".", linestyle="-", label=f"$\\chi = {x}$")
(
    P
    .plot(p_meas, tr_q, marker=".", linestyle="-", color="k", label=f"$\\chi = \\infty$")
    .axhline(tr_mixed, color="0.5", linestyle="--", label="$1 / 2^N$")
    .ggrid()
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    .set_xlabel("$p$")
    .set_ylabel("$\\langle \\langle \\mathregular{tr} \\rho^2_{\\vec{U}, \\vec{b}} \\rangle_\\vec{b} \\rangle_\\vec{U}$")
    .set_ylim(0.0, 1.0)
    .set_title(
        f"$\\mathregular{{circs}} = {circs}; \\mathregular{{size}} = {size}; \\mathregular{{depth}} = {depth}$\n"
        f"$\\Delta t = {dt}; x_0 = {target_x}; \\mathregular{{part}} = {target_range[0]}..{target_range[1] - 1}$",
        fontsize="small",
    )
    .savefig(infile.with_stem(infile.stem + "_tr2").with_suffix(".png"))
    .close()
)

