from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
infile = outdir.joinpath("entropy.npz")
data = np.load(str(infile))
s = data["entropy"]
size = data["size"][0]
p_meas = data["p_meas"][0]

t = np.arange(s.shape[0])

(
    pd.Plotter()
    .plot(t, s)
    .ggrid()
    .set_title(f"N = {size}, p_meas = {p_meas:.3f}")
    .set_xlabel("Time")
    .set_ylabel("Entanglement entropy")
    .savefig(outdir.joinpath("entropy.png"))
    .show()
)

