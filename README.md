# European option pricing with quantum amplitude estimation

CS4134 (Fall 2026) project. Prices a European call with canonical quantum amplitude estimation (QAE) built from scratch in Qiskit: state preparation, payoff encoding, Grover operator, controlled powers, and phase estimation with a hand-built inverse QFT. Results are compared with the exact discretized payoff, Black-Scholes, and classical Monte Carlo at the same number of oracle calls.

## Setup

Python 3.13 (3.10+ should work).

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Run

```
python checks.py          # 18 correctness checks against theory (about 10 s)
python run_noiseless.py   # noiseless sweep m = 1..7, writes results/ (about 15 s)
```

## Outputs (`results/`)

| File | Contents |
|---|---|
| `noiseless_table.png`, `noiseless_table.csv` | QAE estimate, error, theoretical bound, MC and direct-sampling error, CX count per m |
| `noiseless_error_scaling.png` | Median price error vs oracle calls |
| `noiseless_histogram_m5.png` | QAE output distribution vs the Brassard et al. formula |
| `circuit_qae_m3.png`, `circuit_A.png`, `circuit_Q.png` | Circuit diagrams |

## Files

- `model.py`: market parameters, log-normal discretization, Black-Scholes, Monte Carlo
- `circuits.py`: A operator (uniformly controlled RY), Grover operator, controlled-Q, QFT, QAE circuit
- `qae.py`: simulation helper, estimate mapping, theoretical output distribution and error bound
- `checks.py`: verifies each component against theory
- `run_noiseless.py`: the sweep, table and figures

Market parameters are at the top of `model.py`. Run settings (m range, shots, seed) are at the top of `run_noiseless.py`. Results are reproducible with the fixed seed.
