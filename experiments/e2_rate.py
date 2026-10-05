"""E2: growth of the certified budget and time to certify, under honest data
W ~ Bern(1/2 + delta), i.i.d., no attack. Main grid certificate (deflate each bet by its
cost, then average); the union-over-bets variant is recorded for comparison (tau_union).

Saves, for each delta:
  * results/raw/e2_delta{d}.npz : tau[kind][rep, b] (first t with Bhat_t >= b; inf if never)
    and Bhat_t / t at checkpoints for every repetition;
  * results/e2_tau.csv   : mean / median time to certify and the theoretical bounds;
  * results/e2_rate.csv  : quantiles of Bhat_t / t at checkpoints.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import ccb  # noqa: E402

ALPHA = ccb.ALPHA
DELTAS = [0.02, 0.05, 0.10]
T = {0.02: 60000, 0.05: 20000, 0.10: 8000}
REPS = 2000
BS = [0, 5, 10, 20, 50, 100, 200, 400]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")


def g(lam, delta):
    """Expected log growth of one honest record under Bern(1/2 + delta)."""
    return (0.5 + delta) * np.log1p(lam / 2) + (0.5 - delta) * np.log1p(-lam / 2)


def upper_bound_mean_tau(delta, b, kind):
    """Wald bound E tau_b <= min_k [log(1/(alpha w_k)) + b log c_k + log M_+k] / g(lam_k)
    (Theorem: sample complexity), over grid components with g > 0."""
    lams, w = ccb.default_grid()
    best = np.inf
    for lam, wk in zip(lams, w):
        gi = g(lam, delta)
        if gi <= 0:
            continue
        li, lr = ccb.costs(lam)
        c = li if kind == "insertion" else lr
        best = min(best, (np.log(1 / (ALPHA * wk)) + b * c + li) / gi)
    return best


def lower_bound_mean_tau(delta, b, kind):
    """E tau_b >= max( log(1/alpha)/KL , max_n n (1 - alpha - P(Bin(n, q) > b)) ),
    q = 2 delta (insertion) or delta (replacement). Theorem: lower bound."""
    kl = (0.5 + delta) * np.log(1 + 2 * delta) + (0.5 - delta) * np.log(1 - 2 * delta)
    q = 2 * delta if kind == "insertion" else delta
    n = np.arange(1, int(20 * (b + 1) / q) + 10)
    coup = np.max(n * np.clip(1 - ALPHA - stats.binom.sf(b, n, q), 0, None))
    return max(np.log(1 / ALPHA) / kl, coup)


def main():
    tau_rows, rate_rows = [], []
    for di, delta in enumerate(DELTAS):
        rng = np.random.default_rng(20_000 + di)
        n = T[delta]
        checks = np.unique(np.round(np.geomspace(100, n, 25)).astype(int))
        taus = {k: np.full((REPS, len(BS)), np.inf) for k in ("insertion", "replacement")}
        tausu = {k: np.full((REPS, len(BS)), np.inf) for k in ("insertion", "replacement")}
        ratio = {k: np.zeros((REPS, len(checks))) for k in ("insertion", "replacement")}
        for r0 in range(0, REPS, 50):
            H = (rng.random((50, n)) < 0.5 + delta).astype(float)
            for kind in ("insertion", "replacement"):
                Bh = ccb.avg_certified_budget(H, kind)
                Bu = ccb.grid_certified_budget(H, kind)
                ratio[kind][r0:r0 + 50] = Bh[:, checks - 1] / checks
                for j, b in enumerate(BS):
                    for arr, store in ((Bh, taus), (Bu, tausu)):
                        hit = arr >= b
                        any_ = hit.any(axis=1)
                        first = np.argmax(hit, axis=1) + 1.0
                        store[kind][r0:r0 + 50, j] = np.where(any_, first, np.inf)
        np.savez_compressed(os.path.join(RAW, f"e2_delta{delta}.npz"),
                            tau_ins=taus["insertion"], tau_rep=taus["replacement"],
                            tau_union_ins=tausu["insertion"], tau_union_rep=tausu["replacement"],
                            ratio_ins=ratio["insertion"], ratio_rep=ratio["replacement"],
                            checks=checks, bs=np.array(BS))
        for kind in ("insertion", "replacement"):
            for j, b in enumerate(BS):
                t = taus[kind][:, j]
                tau_rows.append(dict(delta=delta, kind=kind, b=b,
                                     frac_certified=float(np.isfinite(t).mean()),
                                     mean_tau=float(np.mean(t)) if np.isfinite(t).all() else np.nan,
                                     median_tau=float(np.median(t)),
                                     mean_tau_union=float(np.mean(tausu[kind][:, j])),
                                     upper=upper_bound_mean_tau(delta, b, kind),
                                     lower=lower_bound_mean_tau(delta, b, kind)))
            for j, c in enumerate(checks):
                q = np.quantile(ratio[kind][:, j], [0.1, 0.5, 0.9])
                rate_rows.append(dict(delta=delta, kind=kind, t=int(c), q10=q[0], q50=q[1], q90=q[2]))
        print("delta", delta, "done", flush=True)
    pd.DataFrame(tau_rows).to_csv(os.path.join(OUT, "e2_tau.csv"), index=False)
    pd.DataFrame(rate_rows).to_csv(os.path.join(OUT, "e2_rate.csv"), index=False)
    print(pd.DataFrame(tau_rows).to_string())


if __name__ == "__main__":
    main()
