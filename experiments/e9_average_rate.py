"""E9: the lead tolerance (exponential wealth, Theorem 5.7) against the betting tolerance
(Theorem 5.1) on i.i.d. honest records with no attack.

For each margin delta and tie share, REPS repetitions of T records are drawn once and both
certificates are computed on the same records, with the insertion and the replacement cost and
the default 14-bet grid (caps for the betting wealth, gammas for the exponential wealth). A record
is a win with probability pi/2 + delta, a loss with probability pi/2 - delta and a tie otherwise,
so the mean outcome is 1/2 + delta and pi is the share of records without a tie (pi = 1: binary
records; pi = 0.7: 30% ties, about the share on Chatbot Arena).

Saved: quantiles and means of Bhat_t / t at the checkpoints, the limit rates
max_k g(lam_k)^+ / c(lam_k) of Theorem 5.3 (betting) and Theorem 5.7 (exponential), and the share
of (repetition, record) cells on which the two tolerances differ in value or in certified status.

Outputs: results/e9_average_rate.csv (one row per setting, wealth, cost and checkpoint),
results/raw/e9_average.npz (Bhat_t / t of every repetition at the checkpoints).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import ccb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")

SETTINGS = [(0.05, 1.0), (0.10, 1.0), (0.05, 0.7), (0.10, 0.7)]   # (delta, pi)
REPS, T, BATCH, SEED = 2000, 5000, 250, 90_000
CHECKS = [1000, 2000, 5000]
FAMILIES = ("bet", "exp")
KINDS = ("insertion", "replacement")


def limit_rate(delta, pi, family, kind):
    """max_k g(lam_k)^+ / c(lam_k): Theorem 5.3 for the betting wealth, Theorem 5.7 for the exponential one."""
    lams, _ = ccb.default_grid()
    if family == "exp":
        g = lams * delta - lams ** 2 / 8
        ci, cr = ccb.exp_costs(lams)
    else:
        ci, cr = ccb.costs(lams)
        g = delta * cr - (pi / 2) * (-np.log1p(-lams ** 2 / 4))
    c = ci if kind == "insertion" else cr
    return float(np.max(np.maximum(g, 0) / c))


def run():
    rng = np.random.default_rng(SEED)
    rows, saved = [], {}
    for delta, pi in SETTINGS:
        a, loss = pi / 2 + delta, pi / 2 - delta
        rate = {(f, k): np.empty((REPS, len(CHECKS))) for f in FAMILIES for k in KINDS}
        differ_value = {k: 0 for k in KINDS}
        differ_status = {k: 0 for k in KINDS}
        for s in range(0, REPS, BATCH):
            U = rng.random((BATCH, T))
            W = np.where(U < a, 1.0, np.where(U < a + loss, 0.0, 0.5))
            for k in KINDS:
                B = {f: ccb.avg_certified_budget(W, k, family=f) for f in FAMILIES}
                for f in FAMILIES:
                    rate[(f, k)][s:s + BATCH] = B[f][:, np.array(CHECKS) - 1] / np.array(CHECKS)
                differ_value[k] += int((B["bet"] != B["exp"]).sum())
                differ_status[k] += int(((B["bet"] >= 0) != (B["exp"] >= 0)).sum())
            print(f"delta={delta} pi={pi}: reps {s + BATCH}/{REPS}", flush=True)
        cells = REPS * T
        for f in FAMILIES:
            for k in KINDS:
                r = rate[(f, k)]
                tag = f"d{delta}_pi{pi}_{f}_{k[:3]}"
                saved[tag] = r
                for j, t in enumerate(CHECKS):
                    q10, q50, q90 = np.percentile(r[:, j], [10, 50, 90])
                    rows.append(dict(delta=delta, pi=pi, ties=round(1 - pi, 2), family=f, kind=k, t=t,
                                     q10=q10, median=q50, q90=q90, mean=float(r[:, j].mean()),
                                     limit_rate=limit_rate(delta, pi, f, k),
                                     frac_cells_value_differs=differ_value[k] / cells,
                                     frac_cells_status_differs=differ_status[k] / cells))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e9_average_rate.csv"), index=False)
    np.savez_compressed(os.path.join(RAW, "e9_average.npz"), checks=np.array(CHECKS), reps=REPS, T=T, seed=SEED,
                        settings=np.array(SETTINGS), **saved)
    pd.set_option("display.width", 250)
    print(df.to_string())
    return df


if __name__ == "__main__":
    run()
