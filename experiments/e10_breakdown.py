"""E10: fixed-time breakdown counts of standard confidence intervals, beside the certificates.

For each E4a Arena pair and each E7a MMLU pair, the largest number of corrupted records that a
fixed-time interval for the mean outcome, computed once on all n records, could absorb while
still excluding 1/2 (lower bound above 1/2):
  * forged: B inserted wins are removed from the data (n - B records remain);
  * flipped: B wins are turned into losses (n records remain).
Two intervals: the two-sided 95% normal (Wald) interval mean +- 1.96 sd / sqrt(n), the baseline
of Section 6, and the Hoeffding interval mean +- sqrt(log(2/alpha) / (2 n)). Both are fixed-time
quantities: they are what a sensitivity study of the final standings computes, and they are not
valid under repeated looks or adaptive timing (Figure 1, the rigging replay). The deterministic
counts of Proposition B.8 (Delta - 1 forged, ceil(Delta/2) - 1 replaced) are listed as well.

Inputs: results/e4a_pairs.csv (n, mean outcome, tie share) and results/e7a_pairs.csv
(wins, losses, ties). Output: results/e10_breakdown.csv.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
OUT = os.path.join(ROOT, "results")
ALPHA, Z = 0.05, 1.959964


def lower_bound(wins, losses, ties, kind):
    n = wins + losses + ties
    if n <= 0 or wins < 0 or losses < 0:
        return -np.inf
    m = (wins + 0.5 * ties) / n
    if kind == "wald":
        var = (wins * (1 - m) ** 2 + losses * m ** 2 + ties * (0.5 - m) ** 2) / n
        return m - Z * np.sqrt(var / n)
    return m - np.sqrt(np.log(2 / ALPHA) / (2 * n))      # Hoeffding, outcomes in [0, 1]


def breakdown(wins, losses, ties, kind, attack):
    """Largest B >= 0 such that the interval still excludes 1/2 after B corruptions; -1 if it
    does not exclude 1/2 even at B = 0."""
    if lower_bound(wins, losses, ties, kind) <= 0.5:
        return -1
    lo, hi = 0, 1
    def ok(B):
        if attack == "forged":
            return lower_bound(wins - B, losses, ties, kind) > 0.5
        return lower_bound(wins - B, losses + B, ties, kind) > 0.5
    while ok(hi) and hi < wins:
        lo, hi = hi, 2 * hi
    hi = min(hi, wins)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def rows_from(df, name, wins, losses, ties, extra):
    rows = []
    for k in range(len(df)):
        w, l, t = int(round(wins[k])), int(round(losses[k])), int(round(ties[k]))
        D = w - l
        r = dict(dataset=name, pair=extra["pair"][k], n=w + l + t, wins=w, losses=l, ties=t, Delta=D,
                 det_forged=D - 1 if D > 0 else -1, det_replaced=int(np.ceil(D / 2)) - 1 if D > 0 else -1,
                 edge_forged=extra["edge_ins"][k], edge_flipped=extra["edge_rep"][k],
                 lead_forged=extra["lead_ins"][k], lead_flipped=extra["lead_rep"][k])
        for kind in ("wald", "hoeff"):
            for attack in ("forged", "flipped"):
                r[f"{kind}_{attack}"] = breakdown(w, l, t, kind, attack)
        rows.append(r)
    return rows


def run():
    a = pd.read_csv(os.path.join(OUT, "e4a_pairs.csv"))
    ties = a.tie_rate * a.n
    wins = a.win_rate * a.n - 0.5 * ties
    losses = a.n - wins - ties
    rows = rows_from(a, "arena", wins.values, losses.values, ties.values,
                     dict(pair=(a.leader + " > " + a.other).tolist(), edge_ins=a.B_ins.astype(int).tolist(),
                          edge_rep=a.B_rep.astype(int).tolist(), lead_ins=a.B_lead_ins.astype(int).tolist(),
                          lead_rep=a.B_lead_rep.astype(int).tolist()))
    e = pd.read_csv(os.path.join(OUT, "e7a_pairs.csv"))
    e = e[e.D > 0].reset_index(drop=True)
    rows += rows_from(e, "mmlu", e.wins.values, e.losses.values, e.ties.values,
                      dict(pair=(e.i + " > " + e.j).tolist(), edge_ins=e.B_ins.astype(int).tolist(),
                           edge_rep=e.B_rep.astype(int).tolist(), lead_ins=e.B_lead_ins.astype(int).tolist(),
                           lead_rep=e.B_lead_rep.astype(int).tolist()))
    df = pd.DataFrame(rows)
    for c in ("edge", "lead", "wald", "hoeff"):
        df[f"{c}_forged_over_det"] = np.where(df.det_forged > 0, df[f"{c}_forged"] / df.det_forged, np.nan)
    df["edge_over_wald_forged"] = np.where(df.wald_forged > 0, df.edge_forged / df.wald_forged, np.nan)
    df["edge_over_wald_flipped"] = np.where(df.wald_flipped > 0, df.edge_flipped / df.wald_flipped, np.nan)
    df.to_csv(os.path.join(OUT, "e10_breakdown.csv"), index=False)
    pd.set_option("display.width", 300)
    cols = ["dataset", "pair", "n", "Delta", "det_forged", "wald_forged", "hoeff_forged", "edge_forged", "lead_forged",
            "det_replaced", "wald_flipped", "hoeff_flipped", "edge_flipped", "lead_flipped", "edge_over_wald_forged", "edge_over_wald_flipped"]
    print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    run()
