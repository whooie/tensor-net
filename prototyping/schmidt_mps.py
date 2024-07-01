from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import numpy.linalg as la
from numpy.random import random
from scipy.linalg import logm

i = 1j

cis = lambda a: np.cos(a) + i * np.sin(a)

I = np.array([[ 1,  0], [ 0,  1]])
XRot = lambda a: cis(a / 2) * np.array([
    [np.cos(a / 2),       -1j * np.sin(a / 2)],
    [-1j * np.sin(a / 2),  np.cos(a / 2)     ],
])
YRot = lambda a: cis(a / 2) * np.array([
    [ np.cos(a / 2), np.sin(a / 2)],
    [-np.sin(a / 2), np.cos(a / 2)],
])
ZRot = lambda a: np.array([
    [1, 0     ],
    [0, cis(a)],
])
U = lambda a, b, c: ZRot(c) @ XRot(b) @ ZRot(a)
H = np.array([[ 1,  1], [ 1, -1]]) / np.sqrt(2)
X = np.array([[ 0,  1], [ 1,  0]])
Y = np.array([[ 0, -i], [ i,  0]])
Z = np.array([[ 1,  0], [ 0, -1]])
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0,  1], [0, 0,  1,  0]])
CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -i], [0, 0, +i,  0]])
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1,  0], [0, 0,  0, -1]])

def kron(*X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.ndarray([[1]])
    elif len(X) == 1:
        return X[0]
    else:
        return np.kron(X[0], kron(*X[1:]))

@dataclass
class LRDecomp:
    l: np.ndarray
    s: np.ndarray
    r: np.ndarray

def lrdecomp(q: np.ndarray) -> LRDecomp:
    (u, s, vh) = la.svd(q, full_matrices=False)
    rank = len([sk for sk in s if sk > 0])
    u = u[:, :rank]
    r = np.array([sv * vhv for (sv, vhv) in zip(s[:rank], vh[:rank, :])])
    s = s[:rank]
    # print(u)
    # print(s)
    # print(r)
    # print()
    return LRDecomp(u, s, r)

@dataclass
class MPS:
    G: list[np.ndarray[complex, 3]]
    L: list[np.ndarray[float, 1]]

def canonicalize(n: int, state: np.ndarray) -> MPS:
    assert state.shape == (2**n,)
    udim = 1
    data = list()
    svals = list()
    q = state.reshape((1, len(state)))
    for k in range(n - 1):
        outdim = 2
        reshape_m = udim * outdim
        reshape_n = q.shape[0] * q.shape[1] // reshape_m
        q = q.reshape((reshape_m, reshape_n))
        lr = lrdecomp(q)
        rank = len(lr.s)
        g = lr.l.reshape((udim, outdim, rank))
        if len(svals) > 0:
            g = np.array([
                gv / sv if sv > 0 else gv for (gv, sv) in zip(g, svals[-1])
            ])
        data.append(g)
        svals.append(lr.s)
        udim = rank
        q = lr.r
    g = np.array([
        qv / sv if sv > 0 else qv for (qv, sv) in zip(q, svals[-1])
    ])
    data.append(g.reshape((udim, outdim, 1)))
    return MPS(data, svals)

def contract(mps: MPS) -> np.ndarray:
    psi = mps.G[0].reshape((2, mps.G[0].shape[-1]))
    for k in range(len(mps.L)):
        psi = psi @ np.diag(mps.L[k])
        sh = mps.G[k + 1].shape
        psi = np.array([[[
            sum(psi[ss, u] * mps.G[k + 1][u, s, v] for u in range(sh[0]))
            for v in range(sh[2])
            ] for s in range(sh[1])
            ] for ss in range(psi.shape[0])
        ]).reshape((psi.shape[0] * sh[1], sh[2]))
    assert psi.shape[1] == 1
    return psi.flatten()

def apply_1q(mps: MPS, k: int, gate: np.ndarray):
    assert gate.shape == (2, 2)
    sh = mps.G[k].shape
    new = np.array([[[
        sum(mps.G[k][v, s, u] * gate[sp, s] for s in range(2))
        for u in range(sh[2])
        ] for sp in range(2)
        ] for v in range(sh[0])
    ])
    mps.G[k] = new

def apply_2q(mps: MPS, k: int, gate: np.ndarray):
    assert gate.shape == (4, 4)
    g = gate.reshape((2, 2, 2, 2)) # [s1p, s2p, s1, s2]
    shk = mps.G[k].shape
    shkp1 = mps.G[k + 1].shape
    qq = np.array([[[[
        sum(
            mps.G[k][v, s1, u]
            * mps.L[k][u]
            * mps.G[k + 1][u, s2, w]
            * g[s1p, s2p, s1, s2]
            for u in range(shk[2])
            for s1 in range(2)
            for s2 in range(2)
        )
        for w in range(shkp1[2])
        ] for s2p in range(2)
        ] for s1p in range(2)
        ] for v in range(shk[0])
    ])
    u, s, vh = la.svd(qq.reshape((shk[0] * 2, 2 * shkp1[2])), full_matrices=False)
    norm = np.sqrt((s**2).sum())
    s = [sk / norm for sk in s if sk > 0.0]
    rank = len(s)
    vh *= norm
    mps.G[k] = u[:, :rank].reshape((shk[0], 2, rank))
    mps.L[k] = s
    mps.G[k + 1] = vh[:rank, :].reshape((rank, 2, shkp1[2]))

def project(mps: MPS, k: int, p: int):
    assert p in {0, 1}
    shk = mps.G[k].shape
    if k == 0:
        prob = sum(
            abs(mps.G[k][0, p, u])**2 * mps.L[k][u]**2
            for u in range(shk[2])
        )
        assert prob > 0.0
        shkp1 = mps.G[k + 1].shape
        qq = np.array([[[[
            sum(
                mps.G[k][v, sk, u]
                * mps.L[k][u]
                * mps.G[k + 1][u, skp1, w]
                / np.sqrt(prob)
                if sk == p else 0.0
                for u in range(shk[2])
            )
            for w in range(shkp1[2])
            ] for skp1 in range(2)
            ] for sk in range(2)
            ] for v in range(shk[0])
        ])
        u, s, vh = la.svd(qq.reshape((shk[0] * 2, 2 * shkp1[2])), full_matrices=False)
        norm = np.sqrt((s**2).sum())
        s = [sk / norm for sk in s if sk > 0.0]
        rank = len(s)
        vh *= norm
        mps.G[k] = u[:, :rank].reshape((shk[0], 2, rank))
        mps.L[k] = s
        mps.G[k + 1] = vh[:rank, :].reshape((rank, 2, shkp1[2]))
    elif k == len(mps.G) - 1:
        prob = sum(
            mps.L[k - 1][v]**2 * abs(mps.G[k][v, p, 0])**2
            for v in range(shk[0])
        )
        assert prob > 0.0
        shkm1 = mps.G[k - 1].shape
        qq = np.array([[[[
            sum(
                mps.G[k - 1][v, skm1, u]
                * mps.L[k - 1][u]
                * mps.G[k][u, sk, w]
                / np.sqrt(prob)
                if sk == p else 0.0
                for u in range(shk[0])
            )
            for w in range(shk[2])
            ] for sk in range(2)
            ] for skm1 in range(2)
            ] for v in range(shkm1[0])
        ])
        u, s, vh = la.svd(qq.reshape((shkm1[0] * 2, 2 * shk[2])), full_matrices=False)
        norm = np.sqrt((s**2).sum())
        s = [sk / norm for sk in s if sk > 0.0]
        rank = len(s)
        vh *= norm
        mps.G[k - 1] = u[:, :rank].reshape((shkm1[0], 2, rank))
        mps.L[k - 1] = s
        mps.G[k] = vh[:rank, :].reshape((rank, 2, shk[2]))
    else:
        prob = sum(
            mps.L[k - 1][v]**2 * abs(mps.G[k][v, p, u])**2 * mps.L[k][u]**2
            for v in range(shk[0])
            for u in range(shk[2])
        )
        assert prob > 0.0
        shkm1 = mps.G[k - 1].shape
        shkp1 = mps.G[k + 1].shape
        qqq = np.array([[[[[
            sum(
                mps.G[k - 1][v, skm1, u1]
                * mps.L[k - 1][u1]
                * mps.G[k][u1, sk, u2]
                * mps.L[k][u2]
                * mps.G[k + 1][u2, skp1, w]
                / np.sqrt(prob)
                if sk == p else 0.0
                for u1 in range(shkm1[2])
                for u2 in range(shkp1[0])
            )
            for w in range(shkp1[2])
            ] for skp1 in range(2)
            ] for sk in range(2)
            ] for skm1 in range(2)
            ] for v in range(shkm1[0])
        ])
        u, s, vh = la.svd(qqq.reshape((shkm1[0] * 2, 2 * 2 * shkp1[2])), full_matrices=False)
        norm = np.sqrt((s**2).sum())
        s = [sk / norm for sk in s if sk > 0.0]
        rank = len(s)
        # vh *= norm
        mps.G[k - 1] = u[:, :rank].reshape((shkm1[0], 2, rank))
        mps.L[k - 1] = s
        qq = np.array([sv * vhv for (sv, vhv) in zip(s, vh[:rank, :])])

        u, s, vh = la.svd(qq.reshape((rank * 2, 2 * shkp1[2])), full_matrices=False)
        norm = np.sqrt((s**2).sum())
        s = [sk / norm for sk in s if sk > 0.0]
        rank = len(s)
        vh *= norm
        mps.G[k] = u[:, :rank].reshape((len(mps.L[k - 1]), 2, rank))
        mps.L[k] = s
        mps.G[k + 1] = vh[:rank, :].reshape((rank, 2, shkp1[2]))

def main():
    psi = np.array([1] + 15 * [0], dtype=complex)
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[4] = 1 / np.sqrt(2)
    print(psi)
    mps = canonicalize(3, psi)
    # print(mps)
    # apply_1q(mps, 0, H)
    # apply_2q(mps, 0, CX)
    # apply_1q(mps, 1, H)
    # apply_2q(mps, 1, CX)
    # project(mps, 3, 0)
    psi = contract(mps)
    # print(psi)

if __name__ == "__main__":
    main()

