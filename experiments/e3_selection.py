"""E3: best-of-k selection. A developer evaluates k private variants, all of which are
no better than the opponent (p = 1/2), and publishes the first variant whose certificate
becomes nonnegative, so a false claim is made iff any variant's certificate ever does.
We compare the naive claim (use the published variant's own certificate) with the
selection-corrected certificate that divides the wealth by k.
Main grid certificate at budget 0 (no corruption); horizon T records per variant.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import ccb  # noqa: E402

ALPHA = ccb.ALPHA
T = 5000
REPS = 4000
KS = [1, 3, 9, 27]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root


def max_grid_score(H, logk):
    """max_t [log sum_l w_l K^(l)_t - log(k/alpha)] for the main grid certificate at
    budget 0 (no attack), per repetition; a false claim is made iff >= 0."""
    return ccb.avg_violation(H, np.zeros_like(H), "insertion", logk=logk)


def main():
    rng = np.random.default_rng(30_000)
    rows = []
    raw = {}
    for k in KS:
        naive = np.full(REPS, -np.inf)
        corr = np.full(REPS, -np.inf)
        for v in range(k):
            H = (rng.random((REPS, T)) < 0.5).astype(float)
            naive = np.maximum(naive, max_grid_score(H, 0.0))
            corr = np.maximum(corr, max_grid_score(H, np.log(k)))
        raw[f"naive_k{k}"] = naive
        raw[f"corrected_k{k}"] = corr
        for name, s in (("naive", naive), ("corrected", corr)):
            p = float(np.mean(s >= 0))
            rows.append(dict(k=k, method=name, rate=p, se=np.sqrt(p * (1 - p) / REPS)))
        print(k, rows[-2:], flush=True)
    np.savez_compressed(os.path.join(ROOT, "results", "raw", "e3_selection.npz"), **raw)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "e3_selection.csv"), index=False)


if __name__ == "__main__":
    main()
