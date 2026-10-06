"""E5: certificate persistence under a regime shift (no attack).

Uncorrupted records are independent and binary (no ties). Model i wins with probability
p1 = 0.6 for the first T1 = 1000 records and p2 = 0.4 afterwards. The null H_0 (i never
has an edge) is false, so the edge claim is correct in the sense of Section 2.1. The
uncorrupted cumulative margin Delta_t (Section 2.1) is positive before t = 2 T1 = 2000 and at most
zero from then on. This experiment measures how long the betting tolerance (Theorem 5.1) stays
nonnegative after the switch and how quickly it returns to -1, for the insertion and replacement
costs and the default 14-bet grid, with no attack (N_t = 0 throughout). The lead tolerance
(exponential wealth, Theorem 5.7) is computed on the same records. Theorem 5.7 bounds by alpha
the probability that it is nonnegative at any t >= 2000. The share of (repetition, record) cells
on which the two tolerances differ is recorded.

Also run: a certificate restarted at the pre-specified time T1 + 1, which sees only the
p2 = 0.4 records. Its null is true, so by Theorem 5.1 it certifies anything with probability at
most alpha; the fraction of runs in which it ever certifies is reported.

Outputs: results/e5_regime.csv (summary), results/raw/e5_regime.npz (paths as int16 and
quantiles), optional preview figure results/e5_regime_preview.pdf (--figure).
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

N, T1, P1, P2 = 5000, 1000, 0.6, 0.4
REPS, BATCH, SEED = 2000, 250, 50_000
CHECKS = [1000, 1100, 1500, 2000, 3000, 5000]
TRAIL = 500


def run():
    rng = np.random.default_rng(SEED)
    p = np.where(np.arange(N) < T1, P1, P2)
    B = {"ins": np.empty((REPS, N), dtype=np.int16), "rep": np.empty((REPS, N), dtype=np.int16),
         "exp_ins": np.empty((REPS, N), dtype=np.int16), "exp_rep": np.empty((REPS, N), dtype=np.int16),
         "ins_restart": np.empty((REPS, N - T1), dtype=np.int16),
         "rep_restart": np.empty((REPS, N - T1), dtype=np.int16)}
    H_all = np.empty((REPS, N), dtype=np.int8)
    for s in range(0, REPS, BATCH):
        H = (rng.random((BATCH, N)) < p).astype(float)
        H_all[s:s + BATCH] = H
        for kind in ("insertion", "replacement"):
            k = kind[:3]
            B[k][s:s + BATCH] = ccb.avg_certified_budget(H, kind)
            B["exp_" + k][s:s + BATCH] = ccb.avg_certified_budget(H, kind, family="exp")
            B[k + "_restart"][s:s + BATCH] = ccb.avg_certified_budget(H[:, T1:], kind)
        print(f"reps {s + BATCH}/{REPS}", flush=True)
    t = np.arange(1, N + 1)
    cum = np.cumsum(H_all, axis=1) / t
    cs = np.cumsum(H_all, axis=1)
    trail = np.full((REPS, N), np.nan)
    trail[:, TRAIL - 1:] = (cs[:, TRAIL - 1:] - np.concatenate([np.zeros((REPS, 1)), cs[:, :-TRAIL]], axis=1)) / TRAIL
    q = {}
    for k in ("ins", "rep", "exp_ins", "exp_rep"):
        q[k] = np.percentile(B[k], [10, 50, 90], axis=0)
    q["cum"] = np.percentile(cum, [10, 50, 90], axis=0)
    q["trail"] = np.nanpercentile(trail, [10, 50, 90], axis=0)
    rows = []
    for k in ("ins", "rep", "exp_ins", "exp_rep"):
        Bk = B[k]
        other = ("exp_" + k) if not k.startswith("exp_") else k[4:]   # the same cost with the other wealth
        after = Bk[:, T1:]
        dropped = after < 0
        has = dropped.any(axis=1)
        t_drop = np.where(has, T1 + 1 + dropped.argmax(axis=1), np.nan)  # first t > T1 with Bhat_t = -1
        delay = t_drop - T1
        row = dict(cost=k, frac_certified_at_T1=float((Bk[:, T1 - 1] >= 0).mean()),
                   median_B_at_T1=float(np.median(Bk[:, T1 - 1])),
                   max_median_B=float(q[k][1].max()), argmax_median_B=int(q[k][1].argmax() + 1),
                   frac_dropped_within_N=float(has.mean()),
                   T_drop_median=float(np.nanmedian(t_drop)), T_drop_q10=float(np.nanpercentile(t_drop, 10)),
                   T_drop_q90=float(np.nanpercentile(t_drop, 90)),
                   delay_median=float(np.nanmedian(delay)), delay_q10=float(np.nanpercentile(delay, 10)),
                   delay_q90=float(np.nanpercentile(delay, 90)),
                   frac_recertified_after_drop=float(np.mean([(Bk[r, int(t_drop[r]):] >= 0).any() for r in range(REPS) if has[r]])) if has.any() else np.nan,
                   restart_frac_ever_certified=float((B[k + "_restart"] >= 0).any(axis=1).mean()) if k + "_restart" in B else np.nan,
                   restart_max_B=int(B[k + "_restart"].max()) if k + "_restart" in B else -1,
                   frac_certified_some_t_ge_2T1=float((Bk[:, 2 * T1 - 1:] >= 0).any(axis=1).mean()),
                   frac_cells_value_differs=float((Bk != B[other]).mean()),
                   frac_cells_status_differs=float(((Bk >= 0) != (B[other] >= 0)).mean()))
        for c in CHECKS:
            row[f"frac_B_ge0_at_{c}"] = float((Bk[:, c - 1] >= 0).mean())
            row[f"median_B_at_{c}"] = float(np.median(Bk[:, c - 1]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e5_regime.csv"), index=False)
    np.savez_compressed(os.path.join(RAW, "e5_regime.npz"), B_ins=B["ins"], B_rep=B["rep"],
                        B_exp_ins=B["exp_ins"], B_exp_rep=B["exp_rep"],
                        B_ins_restart=B["ins_restart"], B_rep_restart=B["rep_restart"],
                        q_ins=q["ins"], q_rep=q["rep"], q_exp_ins=q["exp_ins"], q_exp_rep=q["exp_rep"],
                        q_cum=q["cum"], q_trail=q["trail"],
                        N=N, T1=T1, P1=P1, P2=P2, reps=REPS, seed=SEED, trail=TRAIL)
    pd.set_option("display.width", 200)
    print(df.T.to_string())
    return df


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(RAW, "e5_regime.npz"))
    t = np.arange(1, int(z["N"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    ax = axes[0]
    for k, lab, col in (("q_ins", "insertion certificate", "#1f4e79"), ("q_rep", "replacement certificate", "#b5541c")):
        q = z[k]
        ax.plot(t, q[1], color=col, label=lab)
        ax.fill_between(t, q[0], q[2], color=col, alpha=0.2, lw=0)
    ax.axvline(int(z["T1"]), color="k", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("record $t$"); ax.set_ylabel(r"$\hat B_t$ (median, 10–90%)")
    ax.set_title(f"win probability {z['P1']} then {z['P2']} after $t={int(z['T1'])}$", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.plot(t, z["q_cum"][1], color="#1f4e79", label="cumulative win rate")
    ax.fill_between(t, z["q_cum"][0], z["q_cum"][2], color="#1f4e79", alpha=0.2, lw=0)
    ax.plot(t, z["q_trail"][1], color="#b5541c", label=f"trailing {int(z['trail'])}-record win rate")
    ax.fill_between(t, z["q_trail"][0], z["q_trail"][2], color="#b5541c", alpha=0.2, lw=0)
    ax.axhline(0.5, color="k", lw=0.5); ax.axvline(int(z["T1"]), color="k", ls=":", lw=0.8)
    ax.set_xlabel("record $t$"); ax.set_ylabel("empirical win rate"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e5_regime_preview.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "e5_regime_preview.png"), dpi=160, bbox_inches="tight")
    print("wrote", os.path.join(OUT, "e5_regime_preview.pdf"), "and .png")


if __name__ == "__main__":
    if "--figure-only" not in sys.argv:
        run()
    if "--figure" in sys.argv or "--figure-only" in sys.argv:
        figure()
