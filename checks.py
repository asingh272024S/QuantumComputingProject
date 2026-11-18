import numpy as np
from qiskit.quantum_info import Operator, Statevector

import model
from circuits import a_operator, uniformly_controlled_ry

TOL = 1e-9


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    x, p = model.discretize()
    a, fmax = model.amplitude(x, p)
    f_norm = model.payoff(x) / fmax
    A = a_operator(p, f_norm)
    n = len(p).bit_length() - 1
    results = []

    rng = np.random.default_rng(0)
    for k in range(1, 4):
        N = 2**k
        alphas = rng.uniform(0, 2 * np.pi, N)
        ref = np.zeros((2 * N, 2 * N))
        for j, al in enumerate(alphas):
            c, s = np.cos(al / 2), np.sin(al / 2)
            ref[np.ix_([j, j + N], [j, j + N])] = [[c, -s], [s, c]]
        ok = np.allclose(Operator(uniformly_controlled_ry(alphas)).data, ref, atol=1e-10)
        results.append(report(f"uniformly controlled RY == block-diagonal RY(α_j) (k={k})", ok, f"{N} CX"))

    probs = Statevector(A).probabilities()
    a_sim = probs[2**n:].sum()
    p_sim = probs[:2**n] + probs[2**n:]
    results.append(report("A loads the distribution", np.allclose(p_sim, p, atol=TOL),
                          f"max |p_sim - p| = {np.abs(p_sim - p).max():.2e}"))
    results.append(report("A encodes a = E[f]/f_max", abs(a_sim - a) < TOL,
                          f"a_sim = {a_sim:.12f}, a = {a:.12f}"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
