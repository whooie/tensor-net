import numpy as np
import whooie.pyplotdefs as pd

p0 = 0.3
p1 = 0.35
M = 1000
N = 1000

m0 = np.random.random(size=(M, N)) < p0
mp0 = m0.mean()
print(mp0)
pd.Plotter().imshow(True ^ m0[:25, :25], cmap="gray")

p01 = (p1 - p0) / (1 - p0)
# m1 = np.array([
#     m0k if m0k else np.random.random() < p01 for m0k in m0.flatten()
# ]).reshape((M, N))
m1 = np.array([
    [m0ij if m0ij else np.random.random() < p01 for m0ij in m0i]
    for m0i in m0
])
mp1 = m1.mean()
print(mp1)
pd.Plotter().imshow(True ^ m1[:25, :25], cmap="gray")

pd.show()

