import numpy as np
import whooie.pyplotdefs as pd

data = np.load("test-haar.npz")
q = data["q"]

global_phase = np.arctan2(q[:, 0].imag, q[:, 0].real)
q = np.array([row / np.exp(1j * ph) for (row, ph) in zip(q, global_phase)])
theta = 2 * np.arccos(q[:, 0].real)
phi = np.arctan2(q[:, 1].imag, q[:, 1].real)

x = np.cos(phi) * np.sin(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(theta)

mean_x = x.mean()
std_x = x.std()
mean_y = y.mean()
std_y = y.std()
mean_z = z.mean()
std_z = z.std()

print(f"<x> = {mean_x:.3f} ± {std_x:.3f}")
print(f"<y> = {mean_y:.3f} ± {std_y:.3f}")
print(f"<z> = {mean_z:.3f} ± {std_z:.3f}")

(
    pd.Plotter.new_3d()
    .plot(
        x, y, z,
        marker=".", linestyle="", color="k", alpha=0.1 / np.log10(q.shape[0]),
    )
    .show()
)

