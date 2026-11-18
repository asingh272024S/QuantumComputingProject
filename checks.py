import numpy as np
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

import model
from circuits import a_operator, controlled_grover, grover_operator, qae_circuit, qft, uniformly_controlled_ry
from qae import theory_distribution

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

    for m in range(1, 7):
        ok = Operator(qft(m)).equiv(Operator(QFTGate(m)))
        results.append(report(f"hand-built QFT == QFTGate (m={m})", ok, "unitaries equal up to global phase"))

    Q = grover_operator(A)
    U = Operator(Q).data
    psi = Statevector(A).data
    vals, vecs = np.linalg.eig(U)
    weights = np.abs(vecs.conj().T @ psi) ** 2
    theta = np.arcsin(np.sqrt(a))
    support = weights > 1e-6
    phases = np.angle(vals[support])
    ok = (support.sum() == 2 and np.allclose(np.sort(np.abs(phases)), 2 * theta, atol=1e-7)
          and np.allclose(weights[support], 0.5, atol=1e-7))
    results.append(report("A|0> splits evenly over Q eigenvectors with phases +/-2θ", ok,
                          f"phases = {np.round(phases, 6)}, 2θ = {2 * theta:.6f}, weights = {np.round(weights[support], 6)}"))

    cU = Operator(controlled_grover(A)).data
    on, off = cU[1::2, 1::2], cU[0::2, 0::2]
    ok = np.allclose(on, U, atol=1e-8) and np.allclose(off, np.eye(len(U)), atol=1e-8)
    results.append(report("c-Q is Q (with its -1 phase) when control=1 and I when control=0", ok,
                          f"max errors {np.abs(on - U).max():.2e}, {np.abs(off - np.eye(len(U))).max():.2e}"))

    for m in range(1, 6):
        sv = Statevector(qae_circuit(A, m, measure=False))
        sim = sv.probabilities(qargs=list(range(m)))
        th = theory_distribution(a, m)
        tvd = 0.5 * np.abs(sim - th).sum()
        results.append(report(f"QAE output == Brassard distribution (m={m})", tvd < 1e-8, f"TVD = {tvd:.2e}"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
