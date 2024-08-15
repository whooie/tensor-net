import sys
import numpy as np
import whooie.pyplotdefs as pd

if len(sys.argv) < 2:
    print("missing filename")
    sys.exit(1)

infile = sys.argv[1]

data = np.load(infile)
size = data["size"][0]
depth = data["depth"][0]
p_meas = data["p_meas"][0]
runs = data["runs"][0]
outcomes = data["outcomes"]

(
    pd.Plotter()
    .imshow(np.abs(outcomes).sum(axis=0))
    .colorbar()
    .set_xlabel("$x$")
    .set_ylabel("$t$")
    .ggrid().grid(False, which="both")
    .show()
)

