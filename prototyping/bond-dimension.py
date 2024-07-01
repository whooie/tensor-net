import numpy as np
from numpy.random import random
import whooie.pyplotdefs as pd
from schmidt_mps import *

def random_state(n: int) -> np.ndarray[complex]:
    psi = np.array([random() + 1j * random() for _ in range(2**n)])
    norm = (abs(psi)**2).sum()
    return psi / np.sqrt(norm)

def bond_dimensions(n: int) -> np.ndarray[int]:
    mps = canonicalize(n, random_state(n))
    return np.array([len(bond) for bond in mps.L])

def main():
    N = np.array(list(range(2, 16)))
    chi = [bond_dimensions(n) for n in N]
    max_chi = np.array([max(x) for x in chi])

    P = pd.Plotter.new_3d()
    for (n, x) in zip(N, chi):
        # P.plot_trisurf(np.arange(n - 1), n * np.ones(len(x)), x, color="C0")
        P.plot(np.arange(n - 1), n * np.ones(len(x)), x, color="k")
    (
        P
        .set_xlabel("Bond index")
        .set_ylabel("Qubits")
        .set_zlabel("Bond dimension")
        .savefig("bond-dimensions-3d.png")
    )

    P = pd.Plotter()
    colors = [
        pd.colormaps["vibrant"](k / (N.max() - 1)) for k in range(N.shape[0])]
    for (n, x, c) in zip(N[::-1], chi[::-1], colors[::-1]):
        P.plot(
            np.arange(n - 1), x,
            marker="o", linestyle="-", color=c,
            label=f"$N = {n}$",
        )
    (
        P
        .ggrid()
        .legend(fontsize="x-small")
        .set_xlabel("Bond index")
        .set_ylabel("Bond dimension")
        .savefig("bond-dimensions.png")
    )

    (
        pd.Plotter()
        .plot(N, max_chi, marker="o", linestyle="-", color="C0")
        # .plot(N, [2**(n // 2) for n in N], marker=".", color="k")
        .ggrid()
        .set_xlabel("Qubits")
        .set_ylabel("Max. bond dimension")
        .savefig("bond-dimensions-max.png")
    )

    # pd.show()

if __name__ == "__main__":
    main()

