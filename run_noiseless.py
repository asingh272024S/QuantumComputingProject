import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

import model
from circuits import a_operator, direct_sampling_circuit, qae_circuit
from qae import counts_to_outcomes, error_bound, estimate, oracle_calls, run, theory_distribution

M_VALUES = range(1, 8)
HIST_M = 5
SHOTS = 8192
MC_TRIALS = 2000
SEED = 7
OUT = "results"

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
QAE_C, MC_C, DS_C = "#2a78d6", "#eb6834", "#1baf7a"


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return values[order][np.searchsorted(cum, 0.5)]


def sweep(backend, tag):
    rng = np.random.default_rng(SEED)
    x, p = model.discretize()
    a, fmax = model.amplitude(x, p)
    exact = float(model.to_price(a, fmax))
    bs = model.black_scholes()
    A = a_operator(p, model.payoff(x) / fmax)

    print(f"grid S_T = {np.round(x, 2)}\np = {np.round(p, 4)}")
    print(f"a = {a:.6f}, f_max = {fmax:.4f}, exact discretized price = {exact:.4f}, Black-Scholes = {bs:.4f}\n")

    rows, hists = [], {}
    for m in M_VALUES:
        counts, _ = run(qae_circuit(A, m), backend, SHOTS, SEED)
        ys, ws = counts_to_outcomes(counts, m)
        prices = model.to_price(estimate(ys, m), fmax)
        errs = np.abs(prices - exact)
        a_errs = np.abs(estimate(ys, m) - a)

        values = np.round(prices, 10)
        uniq = np.unique(values)
        mass = np.array([ws[values == u].sum() for u in uniq])
        mode_price = uniq[mass.argmax()]
        hists[m] = (uniq, mass / SHOTS)

        n = oracle_calls(m)
        mc_err = np.abs(model.monte_carlo(x, p, n, MC_TRIALS, rng) - exact)
        mem, _ = run(direct_sampling_circuit(A), backend, n * MC_TRIALS, SEED, memory=True)
        hits = np.array([s == "1" for s in mem]).reshape(MC_TRIALS, n)
        ds_err = np.abs(model.to_price(hits.mean(axis=1), fmax) - exact)

        hw = transpile(qae_circuit(A, m), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=SEED)
        rows.append({
            "m": m,
            "oracle_calls": n,
            "mode_price": mode_price,
            "mode_abs_err": abs(mode_price - exact),
            "qae_median_abs_err": weighted_median(errs, ws),
            "bound_price": float(model.to_price(error_bound(a, m), fmax)),
            "p_within_bound": ws[a_errs <= error_bound(a, m)].sum() / SHOTS,
            "mc_median_abs_err": np.median(mc_err),
            "direct_median_abs_err": np.median(ds_err),
            "cx_count": hw.count_ops().get("cx", 0),
            "depth": hw.depth(),
        })
        r = rows[-1]
        print(f"m={m}: calls={n:4d}  mode=${r['mode_price']:.4f}  |err|={r['mode_abs_err']:.4f}  "
              f"QAE med={r['qae_median_abs_err']:.4f}  MC med={r['mc_median_abs_err']:.4f}  "
              f"direct med={r['direct_median_abs_err']:.4f}  P(in bound)={r['p_within_bound']:.3f}  cx={r['cx_count']}")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{tag}_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    ctx = dict(a=a, fmax=fmax, exact=exact, bs=bs, x=x, p=p)
    plot_scaling(rows, ctx, tag)
    plot_histogram(hists[HIST_M], ctx, tag)
    plot_table(rows, ctx, tag)
    return rows, ctx


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scaling(rows, ctx, tag):
    calls = np.array([r["oracle_calls"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    style(ax)
    sd = model.discount() * np.sqrt(ctx["p"] @ model.payoff(ctx["x"]) ** 2 - (ctx["p"] @ model.payoff(ctx["x"])) ** 2)
    nn = np.logspace(0, np.log10(calls.max() * 1.3), 100)
    ax.plot(nn, 0.6745 * sd / np.sqrt(nn), color=MC_C, lw=1.2, ls="--", alpha=0.7, label="MC theory 0.674·σ/√N")
    ax.plot(calls, [r["bound_price"] for r in rows], color=QAE_C, lw=1.2, ls="--", alpha=0.7,
            label="QAE bound (Brassard, prob. ≥ 8/π²)")
    series = [("qae_median_abs_err", QAE_C, "o", "QAE (canonical)"),
              ("mc_median_abs_err", MC_C, "s", "Classical Monte Carlo"),
              ("direct_median_abs_err", DS_C, "^", "Direct sampling of A")]
    for z, (key, c, mk, label) in zip((4, 3, 2), series):
        y = [r[key] for r in rows]
        ax.plot(calls, y, color=c, lw=2, marker=mk, ms=8, mec=SURFACE, mew=2, label=label, zorder=z)
        ax.annotate(label, (calls[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(right=calls.max() * 6)
    ax.set_ylim(top=3 * max(r[k] for r in rows for k, *_ in series))
    ax.set_xlabel("Oracle calls N (applications of A or A†)", color=INK)
    ax.set_ylabel("Median |price error| vs exact discretized price ($)", color=INK)
    ax.set_title(f"Price error vs oracle calls ({tag})", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}_error_scaling.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_histogram(hist, ctx, tag):
    uniq, prob = hist
    M = 2**HIST_M
    th_y = theory_distribution(ctx["a"], HIST_M)
    th_prices = np.round(model.to_price(estimate(np.arange(M), HIST_M), ctx["fmax"]), 10)
    th_mass = np.array([th_y[th_prices == u].sum() for u in uniq])
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    style(ax)
    width = 0.8 * np.min(np.diff(uniq)) if len(uniq) > 1 else 0.5
    ax.bar(uniq, prob, width=width, color=QAE_C, edgecolor=SURFACE, linewidth=2, label=f"Simulated ({SHOTS} shots)")
    ax.plot(uniq, th_mass, ls="none", marker="D", ms=8, color=INK, mec=SURFACE, mew=1.5, label="Theory (Brassard et al.)")
    ax.axvline(ctx["exact"], color=MC_C, lw=2, label=f"Exact discretized \\${ctx['exact']:.3f}")
    ax.axvline(ctx["bs"], color=DS_C, lw=2, ls="--", label=f"Black-Scholes \\${ctx['bs']:.3f}")
    ax.set_xlim(uniq.min() - 2, min(uniq.max(), 4 * ctx["exact"]) + 2)
    ax.set_xlabel("QAE price estimate e^(−rT) f_max sin²(πy/2^m)  ($)", color=INK)
    ax.set_ylabel("Probability", color=INK)
    ax.set_title(f"QAE output distribution, m = {HIST_M} evaluation qubits ({tag})", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}_histogram_m{HIST_M}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_table(rows, ctx, tag):
    cols = ["m", "Oracle\ncalls", "QAE estimate\n(mode, $)", "|Error|\n(mode, $)", "QAE median\n|error| ($)",
            "Theory bound\n($)", "P(within\nbound)", "MC median\n|error| ($)", "Direct sampl.\nmedian\n|error| ($)", "CX\ngates"]
    cells = [[r["m"], r["oracle_calls"], f"{r['mode_price']:.3f}", f"{r['mode_abs_err']:.3f}",
              f"{r['qae_median_abs_err']:.3f}", f"{r['bound_price']:.3f}", f"{r['p_within_bound']:.3f}",
              f"{r['mc_median_abs_err']:.3f}", f"{r['direct_median_abs_err']:.3f}", r["cx_count"]] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 1.2), facecolor=SURFACE)
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#f0efec" if r == 0 else SURFACE)
        if r == 0:
            cell.set_height(cell.get_height() * 1.7)
    ax.set_title(f"Canonical QAE, European call ({tag}). S₀={model.S0:g}, K={model.K:g}, r={model.R:g}, "
                 f"σ={model.SIGMA:g}, T={model.T:g}, 3 price qubits, {SHOTS} shots per m.\n"
                 f"Exact discretized price \\${ctx['exact']:.4f} (a = {ctx['a']:.4f}),  Black-Scholes \\${ctx['bs']:.4f}.  "
                 f"MC and direct sampling use the same number of oracle calls, median over {MC_TRIALS} trials.",
                 color=INK, fontsize=9.5, loc="left")
    fig.savefig(f"{OUT}/{tag}_table.png", dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def draw_circuits():
    x, p = model.discretize()
    a, fmax = model.amplitude(x, p)
    from circuits import grover_operator
    A = a_operator(p, model.payoff(x) / fmax)
    os.makedirs(OUT, exist_ok=True)
    for name, qc in [("circuit_qae_m3", qae_circuit(A, 3)), ("circuit_A", A.decompose()),
                     ("circuit_Q", grover_operator(A))]:
        qc.draw("mpl", fold=-1, filename=f"{OUT}/{name}.png")
        plt.close("all")


if __name__ == "__main__":
    sweep(AerSimulator(), "noiseless")
    draw_circuits()
