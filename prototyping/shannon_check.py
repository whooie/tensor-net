import numpy as np
import whooie.pyplotdefs as pd

def sh(p: np.ndarray[float, 1]) -> float:
    return sum(-pk * np.log(pk) for pk in p if pk > 0)

p0 = np.linspace(0.0, 1.0, 100)
dp0 = p0[1] - p0[0]
p1 = np.linspace(0.0, 1.0, 100)
dp1 = p1[1] - p1[0]

H = np.array([
    [
        sh([p0k, p1k, 1 - p0k - p1k]) if p0k + p1k <= 1 else np.nan
        for p1k in p1
    ] for p0k in p0
])

P1, P0 = np.meshgrid(p1, p0)

extent = [
    p1.min() - dp1 / 2, p1.max() + dp1 / 2,
    p0.min() - dp0 / 2, p0.max() + dp0 / 2,
]

pd.Plotter() \
    .imshow(H, extent=extent, cmap="vibrant", origin="lower") \
    .colorbar() \
    .contour(P1, P0, H, levels=[np.log(3) * 0.999], colors=["k"]) \
    .plot([1 / 3], [1 / 3], marker="o", color="r") \
    .show()

