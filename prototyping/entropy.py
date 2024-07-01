import numpy as np
from scipy.linalg import logm, svd

def qq(x):
    print(x)
    return x

def trace(rho: np.ndarray) -> float:
    return sum(rk for rk in np.diag(rho) if abs(rk) > 1e-12)

def trace_qubit(n: int, k: int, rho: np.ndarray) -> np.ndarray:
    T0 = np.kron(
        np.kron(np.eye(2 ** k), np.array([[1, 0]]).T),
        np.eye(2 ** (n - k - 1)),
    )
    T1 = np.kron(
        np.kron(np.eye(2 ** k), np.array([[0, 1]]).T),
        np.eye(2 ** (n - k - 1)),
    )
    return (T0.T @ rho @ T0) + (T1.T @ rho @ T1)

def entropy_log(rho: np.ndarray) -> float:
    return -trace(rho @ logm(rho))

def entropy_svd(rho: np.ndarray) -> float:
    s = svd(rho)[1]
    return -sum(sk ** 2 * np.log(sk**2) for sk in s if abs(sk**2) > 1e-12)

psi_w3 = np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=complex) / np.sqrt(3)
rho_w3 = np.outer(psi_w3, psi_w3)

psi_b01 = np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=complex) / np.sqrt(2)
rho_b01 = np.outer(psi_b01, psi_b01)

psi_b12 = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=complex) / np.sqrt(2)
rho_b12 = np.outer(psi_b12, psi_b12)

rho_b3 = 0.5 * rho_b01 + 0.5 * rho_b12

rho = rho_b01
print(entropy_log(rho))
print(entropy_log(trace_qubit(3, 2, rho)))
print(entropy_log(trace_qubit(2, 0, trace_qubit(3, 0, rho))))
print(entropy_svd(rho))
print(entropy_svd(trace_qubit(3, 2, rho)))
print(entropy_svd(trace_qubit(2, 0, trace_qubit(3, 0, rho))))

