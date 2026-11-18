import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def uniformly_controlled_ry(alphas):
    # RY(alphas[j]) on the target (last qubit) when the controls are in state |j>; 2^k RY + 2^k CX (Gray code)
    N = len(alphas)
    k = int(np.log2(N))
    qc = QuantumCircuit(k + 1, name="UCRY")
    if k == 0:
        qc.ry(alphas[0], 0)
        return qc
    gray = [i ^ (i >> 1) for i in range(N)]
    signs = np.array([[(-1) ** bin(j & g).count("1") for g in gray] for j in range(N)])
    thetas = signs.T @ np.asarray(alphas) / N
    for i in range(N):
        qc.ry(thetas[i], k)
        changed = gray[i] ^ gray[(i + 1) % N]
        qc.cx(changed.bit_length() - 1, k)
    return qc


def state_prep(p):
    # binary tree of uniformly controlled RY: qubit n-1 is the most significant bit of the price index
    n = int(np.log2(len(p)))
    qc = QuantumCircuit(n, name="A_p")
    for level in range(n):
        target, block = n - 1 - level, 2 ** (n - level)
        alphas = []
        for prefix in range(2**level):
            seg = p[prefix * block:(prefix + 1) * block]
            ratio = seg[:block // 2].sum() / seg.sum() if seg.sum() > 0 else 1.0
            alphas.append(2 * np.arccos(np.sqrt(np.clip(ratio, 0, 1))))
        qc.compose(uniformly_controlled_ry(alphas), list(range(target + 1, n)) + [target], inplace=True)
    return qc


def payoff_rotation(f_norm):
    # |i>|0> -> |i>(sqrt(1-f_i)|0> + sqrt(f_i)|1>)
    n = int(np.log2(len(f_norm)))
    qc = QuantumCircuit(n + 1, name="A_f")
    qc.compose(uniformly_controlled_ry(2 * np.arcsin(np.sqrt(f_norm))), range(n + 1), inplace=True)
    return qc


def a_operator(p, f_norm):
    n = int(np.log2(len(p)))
    qc = QuantumCircuit(QuantumRegister(n, "price"), QuantumRegister(1, "anc"), name="A")
    qc.append(state_prep(p).to_gate(), range(n))
    qc.append(payoff_rotation(f_norm).to_gate(), range(n + 1))
    return qc


def zero_reflection(k):
    # S_0 = I - 2|0><0|
    qc = QuantumCircuit(k, name="S_0")
    qc.x(range(k))
    qc.h(k - 1)
    qc.mcx(list(range(k - 1)), k - 1)
    qc.h(k - 1)
    qc.x(range(k))
    return qc


def grover_operator(a_circ):
    # Q = -A S_0 A^dagger S_chi, good states have ancilla (last qubit) = 1
    k = a_circ.num_qubits
    qc = QuantumCircuit(*a_circ.qregs, name="Q")
    qc.z(k - 1)
    qc.append(a_circ.inverse().to_gate(label="A†"), range(k))
    qc.append(zero_reflection(k).to_gate(), range(k))
    qc.append(a_circ.to_gate(), range(k))
    qc.global_phase += np.pi
    return qc


def controlled_grover(a_circ):
    # qubit 0 is the control; A and A^dagger need no control since A A^dagger = I when it is off
    k = a_circ.num_qubits
    st = list(range(1, k + 1))
    qc = QuantumCircuit(k + 1, name="c-Q")
    qc.cz(0, k)
    qc.append(a_circ.inverse().to_gate(label="A†"), st)
    qc.x(st)
    qc.h(k)
    qc.mcx([0] + st[:-1], k)
    qc.h(k)
    qc.x(st)
    qc.append(a_circ.to_gate(), st)
    qc.z(0)
    return qc
