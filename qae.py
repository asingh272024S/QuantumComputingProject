import numpy as np
from qiskit import transpile


def run(qc, backend, shots, seed=None, memory=False):
    tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
    result = backend.run(tqc, shots=shots, seed_simulator=seed, memory=memory).result()
    return (result.get_memory() if memory else result.get_counts()), tqc


def estimate(y, m):
    return np.sin(np.pi * np.asarray(y) / 2**m) ** 2


def counts_to_outcomes(counts, m):
    ys = np.array([int(key, 2) for key in counts])
    return ys, np.array([counts[key] for key in counts])


def theory_distribution(a, m):
    # P(y) = (F(y - M theta/pi) + F(y + M theta/pi)) / 2,  F(d) = sin^2(pi d) / (M^2 sin^2(pi d / M))
    M = 2**m
    theta = np.arcsin(np.sqrt(a))
    y = np.arange(M)

    def F(d):
        den = (M * np.sin(np.pi * d / M)) ** 2
        return np.where(den < 1e-15, 1.0, np.sin(np.pi * d) ** 2 / np.maximum(den, 1e-15))

    return 0.5 * F(y - M * theta / np.pi) + 0.5 * F(y + M * theta / np.pi)


def error_bound(a, m):
    M = 2**m
    return 2 * np.pi * np.sqrt(a * (1 - a)) / M + np.pi**2 / M**2


def oracle_calls(m):
    # one A for the initial state plus A and A^dagger in each of the 2^m - 1 Grover iterations
    return 2 ** (m + 1) - 1
