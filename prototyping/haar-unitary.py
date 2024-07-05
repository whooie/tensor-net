import numpy as np
import whooie.pyplotdefs as pd

def random_haar(n: int) -> np.ndarray[complex, 2]:
    z = np.array([
        np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
        for _ in range((2**n)**2)
    ]).reshape((2**n, 2**n))
    (q, r) = np.linalg.qr(z)
    for j in range(2**n):
        q[:, j] *= r[j, j] / abs(r[j, j])
    return q

def random_haar_state(
    n: int,
    fid: np.ndarray[complex, 1],
) -> np.ndarray[complex, 1]:
    assert len(fid) == 2**n
    return random_haar(n) @ fid

def bloch_sphere(state: np.ndarray[complex, 1]) -> (float, float):
    global_phase = np.arctan2(state[0].imag, state[0].real)
    state /= np.exp(1j * global_phase) * np.sqrt((abs(state)**2).sum())
    theta = 2 * np.arccos(state[0].real)
    phi = np.arctan2(state[1].imag, state[1].real)
    return (theta, phi)

def to_cartesian(theta: float, phi: float) -> (float, float, float):
    return (
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta),
    )

MC = 10000
psi0 = np.array([1, 0], dtype=complex)
coords = np.array([
    to_cartesian(*bloch_sphere(random_haar_state(1, psi0)))
    for _ in range(MC)
])

(
    pd.Plotter.new_3d()
    .plot(
        coords[:, 0], coords[:, 1], coords[:, 2],
        marker=".", linestyle="", color="k", alpha=0.5,
    )
    .show()
)


