"""E7: certified budgets and contamination replay on a static benchmark (HELM MMLU).

Data. Per-item results of four open-weight 7-9B models on MMLU (57 subjects, 5-shot,
multiple_choice_joint) from Stanford HELM's public bucket (release v1.13.0 index):
    meta_llama-3.1-8b-instruct-turbo, qwen_qwen2.5-7b-instruct-turbo,
    google_gemma-2-9b (base model; HELM's MMLU leaderboard does not include gemma-2-9b-it),
    mistralai_mistral-7b-instruct-v0.3. The two Turbo runs used Together AI's FP8-quantized endpoints.
For each run, per_instance_stats.json gives exact_match in {0, 1} per instance. Mode `fetch`
downloads the 4 x 57 files (cached, gzipped, under data/helm_mmlu/ in the repository) and writes the
per-item correctness table results/raw/e7_mmlu_items.csv.gz; the other modes read that table.

Record. One MMLU question. For the ordered pair (i, j): W = 1 if i is correct and j wrong,
0 if i wrong and j correct, 1/2 if both are correct or both wrong. The record order is a
fixed random permutation (seed 0) of all items, drawn before any outcome is looked at; the
final certificate does not depend on the order (App. D.1), the time to certify does.

E7a (mode `a`)  clean certified budgets for all 12 ordered pairs, both costs, with the raw
    breakdown count D = wins - losses (forged items for the weaker model that move the observed
    mean outcome to 1/2) and ceil(D/2) (flips), the ratios Bhat / D and Bhat / ceil(D/2), and the
    records needed to certify b in {0, 10, 25, 50} under the seed-0 order and over 100 random orders.
E7b (mode `b`)  contamination replay, as E4b: for each unordered pair the attacked model is
    the one with the lower accuracy, null streams resample the pair's own items with
    replacement (so 'attacked model is better' is false), and the attacker inserts B forged
    items (W = 1) in one burst at a random position, spreads them, or contaminates the first
    B items the attacked model lost (W = 0 -> 1, the value-dependent attacker). Methods as in
    E4b: Wald interval, undeflated betting, per-step robust, ours insertion, ours replacement.
"""
import gzip
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import ccb  # noqa: E402
import e4_arena as E4  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "data", "helm_mmlu")   # the 228 per_instance_stats.json files, gzipped, are in the repository
ITEMS = os.path.join(RAW, "e7_mmlu_items.csv.gz")

BUCKET = "https://storage.googleapis.com/crfm-helm-public/"
RELEASE = "v1.13.0"
MODELS = {"llama-3.1-8b-instruct": "meta_llama-3.1-8b-instruct-turbo",
          "qwen2.5-7b-instruct": "qwen_qwen2.5-7b-instruct-turbo",
          "gemma-2-9b": "google_gemma-2-9b",
          "mistral-7b-instruct-v0.3": "mistralai_mistral-7b-instruct-v0.3"}
ALPHA = ccb.ALPHA
SEED_ORDER = 0
N_PERM = 100
TAU_B = [0, 10, 25, 50]
BS_INS = [0, 10, 25, 50, 100, 200, 400]
BS_REP = [0, 5, 10, 25, 50, 100, 200]
REPS, BATCH = 1000, 250


# ------------------------------------------------------------------ data
def _get_json(path):
    with urllib.request.urlopen(BUCKET + urllib.parse.quote(path)) as r:
        return json.load(r)


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    idx = _get_json(f"mmlu/benchmark_output/releases/{RELEASE}/runs_to_run_suites.json")
    jobs = []
    for short, helm in MODELS.items():
        runs = [r for r in idx if f"model={helm}," in r]
        assert len(runs) == 57, (helm, len(runs))
        for run in runs:
            subject = run.split("subject=")[1].split(",")[0]
            jobs.append((short, subject, f"mmlu/benchmark_output/runs/{idx[run]}/{run}/per_instance_stats.json"))

    def one(job):
        short, subject, path = job
        local = os.path.join(CACHE, f"{short}__{subject}.json.gz")
        if not os.path.exists(local):
            with urllib.request.urlopen(BUCKET + urllib.parse.quote(path)) as r, gzip.open(local, "wb") as f:
                f.write(r.read())
        return job, local

    rows = []
    with ThreadPoolExecutor(8) as ex:
        for (short, subject, _), local in ex.map(one, jobs):
            for e in json.load(gzip.open(local, "rt")):
                if e.get("train_trial_index", 0) != 0:
                    continue
                em = [s for s in e["stats"] if s["name"]["name"] == "exact_match"]
                assert len(em) == 1 and em[0]["count"] == 1
                rows.append((subject, e["instance_id"], short, float(em[0]["mean"])))
    df = pd.DataFrame(rows, columns=["subject", "instance_id", "model", "correct"])
    tab = df.pivot_table(index=["subject", "instance_id"], columns="model", values="correct")
    assert tab.notna().all().all(), "items missing for some model"
    tab = tab.reset_index()
    tab.to_csv(ITEMS, index=False, compression="gzip")
    print(f"{len(tab)} items x {len(MODELS)} models written to {os.path.relpath(ITEMS, HERE)}")
    print("accuracy:", tab[list(MODELS)].mean().round(4).to_dict())
    return tab


def load_items():
    return pd.read_csv(ITEMS)


def pair_outcomes(tab, i, j, order):
    """W = 1 if i is right and j wrong, 0 for the reverse, 1/2 if both are right or both wrong.
    Per-item scores must be 0 or 1; a fractional score has no right/wrong reading."""
    a, b = tab[i].to_numpy()[order], tab[j].to_numpy()[order]
    assert np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all(), "per-item scores must be 0 or 1"
    return np.where(a == b, 0.5, np.where(a == 1.0, 1.0, 0.0))


def pair_outcomes_graded(tab, i, j, order):
    """W = (1 + s_i - s_j) / 2 for per-item scores s in [0, 1] (used for Omni-MATH in E8, whose
    judged scores are 0, 1/3, 1/2, 2/3 or 1). For 0/1 scores this equals pair_outcomes."""
    a, b = tab[i].to_numpy()[order], tab[j].to_numpy()[order]
    assert ((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)).all(), "per-item scores must lie in [0, 1]"
    return 0.5 * (1.0 + a - b)


# ------------------------------------------------------------------ E7a
def e7a(tab, models=None, tag="e7a", outcomes=None, combined_raw=False):
    """outcomes: the record function (default pair_outcomes). combined_raw: save the paths of
    all pairs in one file results/raw/<tag>_paths.npz instead of one file per pair."""
    outcomes = outcomes or pair_outcomes
    n = len(tab)
    order0 = np.random.default_rng(SEED_ORDER).permutation(n)
    perms = [np.random.default_rng(1000 + s).permutation(n) for s in range(N_PERM)]
    names = list(models or MODELS)
    rows, prow, combined = [], [], {"order": order0}
    for i in names:
        for j in names:
            if i == j:
                continue
            w = outcomes(tab, i, j, order0)
            nwin, nloss = int((w == 1).sum()), int((w == 0).sum())
            D = float(np.sum(2 * w - 1))   # = wins - losses when every record is 0, 1/2 or 1
            D = int(round(D)) if abs(D - round(D)) < 1e-9 else D
            r = dict(i=i, j=j, n=n, wins=nwin, losses=nloss, ties=n - nwin - nloss, win_rate=float(w.mean()),
                     D=D, D_rep=int(np.ceil(D / 2)) if D > 0 else 0)
            paths = {}
            for k, kind in (("ins", "insertion"), ("rep", "replacement")):
                path = ccb.avg_certified_budget(w[None, :], kind)[0]
                paths[k] = path
                r[f"B_{k}"] = int(path[-1])
                raw = D if k == "ins" else r["D_rep"]
                r[f"ratio_{k}"] = float(path[-1] / raw) if raw > 0 and path[-1] >= 0 else np.nan
                for b in TAU_B:
                    hit = np.flatnonzero(path >= b)
                    r[f"tau{b}_{k}"] = int(hit[0] + 1) if len(hit) else np.nan
            for k, kind in (("ins", "insertion"), ("rep", "replacement")):   # lead certificate (Theorem 5.7)
                lead = ccb.avg_certified_budget(w[None, :], kind, family="exp")[0]
                paths[f"lead_{k}"] = lead
                r[f"B_lead_{k}"] = int(lead[-1])
            rows.append(r)
            if D > 0:  # time to certify over random orders, for the pairs that certify
                for s, perm in enumerate(perms):
                    wp = outcomes(tab, i, j, perm)
                    for k, kind in (("ins", "insertion"), ("rep", "replacement")):
                        path = ccb.avg_certified_budget(wp[None, :], kind)[0]
                        q = dict(i=i, j=j, perm=s, cost=k, B_final=int(path[-1]))
                        for b in TAU_B:
                            hit = np.flatnonzero(path >= b)
                            q[f"tau{b}"] = int(hit[0] + 1) if len(hit) else np.nan
                        prow.append(q)
            if combined_raw:
                combined.update({f"{i}__{j}__W": w, f"{i}__{j}__B_ins": paths["ins"].astype(np.int32),
                                 f"{i}__{j}__B_rep": paths["rep"].astype(np.int32),
                                 f"{i}__{j}__B_lead_ins": paths["lead_ins"].astype(np.int32),
                                 f"{i}__{j}__B_lead_rep": paths["lead_rep"].astype(np.int32)})
            else:
                np.savez_compressed(os.path.join(RAW, f"{tag}_{i}__{j}.npz"), order=order0, W=w,
                                    B_ins=paths["ins"].astype(np.int32), B_rep=paths["rep"].astype(np.int32),
                                    B_lead_ins=paths["lead_ins"].astype(np.int32), B_lead_rep=paths["lead_rep"].astype(np.int32))
            print(i, j, r["B_ins"], r["B_rep"], flush=True)
    if combined_raw:
        np.savez_compressed(os.path.join(RAW, f"{tag}_paths.npz"), **combined)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"{tag}_pairs.csv"), index=False)
    pp = pd.DataFrame(prow)
    pp.to_csv(os.path.join(OUT, f"{tag}_perm.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    if len(pp):
        print(pp.groupby(["i", "j", "cost"]).agg(B_final_min=("B_final", "min"), B_final_max=("B_final", "max"),
                                                 tau0_med=("tau0", "median"), tau0_q10=("tau0", lambda x: x.quantile(.1)),
                                                 tau0_q90=("tau0", lambda x: x.quantile(.9)),
                                                 tau50_med=("tau50", "median")).to_string())
    return df, pp


# ------------------------------------------------------------------ E7b
def e7b(tab, models=None, tag="e7b", reps=REPS, outcomes=None):
    outcomes = outcomes or pair_outcomes
    n = len(tab)
    order0 = np.random.default_rng(SEED_ORDER).permutation(n)
    names = list(models or MODELS)
    acc = tab[names].mean()
    rows = []
    pairs = [(a, b) for ai, a in enumerate(names) for b in names[ai + 1:]]
    for pi, (a, b) in enumerate(pairs):
        x, y = (a, b) if acc[a] < acc[b] else (b, a)   # x = attacked model, the weaker one
        w = outcomes(tab, x, y, order0)
        if not w.mean() < 0.5:
            print('skip (equal accuracy):', x, y); continue
        for attack, BS in (("burst_random", BS_INS), ("spread", BS_INS), ("online_replace", BS_REP)):
            for bi, B in enumerate(BS):
                rng = np.random.default_rng(70_000 + 1000 * pi + 100 * bi + len(attack))
                hits = {m: 0 for m in ("wald", "naive", "perstep", "avg_ins", "avg_rep")}
                for s in range(0, reps, BATCH):
                    H = rng.choice(w, size=(BATCH, n), replace=True)
                    if attack == "burst_random":
                        W, F = E4.attack_burst_random(H, B, rng)
                    elif attack == "spread":
                        W, F = ccb.attack_spread(H, B, rng)
                    else:
                        W, F = ccb.attack_online_replace(H, B)
                    v = dict(wald=E4.wald_violation(W, F),
                             naive=ccb.avg_violation(W, np.zeros_like(F)),
                             perstep=ccb.single_violation(W, F, 0.2, "none", eps=B / n),
                             avg_ins=ccb.avg_violation(W, F, "insertion"),
                             avg_rep=ccb.avg_violation(W, F, "replacement"))
                    for m, val in v.items():
                        hits[m] += int(np.sum(val >= -1e-12))
                for m, h in hits.items():
                    p = h / reps
                    rows.append(dict(attacked=x, opponent=y, n=n, win_rate=float(w.mean()), attack=attack, B=B,
                                     method=m, rate=p, se=np.sqrt(p * (1 - p) / reps)))
                print(x, y, attack, B, {m: round(h / reps, 3) for m, h in hits.items()}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{tag}_contamination.csv"), index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["attacked", "attack", "B"], columns="method", values="rate").to_string())
    return df


SHORT = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "qwen2.5-7b-instruct": "Qwen2.5-7B",
         "gemma-2-9b": "Gemma-2-9B", "mistral-7b-instruct-v0.3": "Mistral-7B"}


def figure(tag_a="e7a", tag_b="e7b", out="e7_preview", title="HELM MMLU, 14,042 items", short=None):
    """Preview (not a paper figure): A certified budgets against the raw breakdown counts;
    B, C probability of a false claim under forged items and under contaminated items,
    mean over the six pairs, as in Table 1 of the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = short or SHORT
    a = pd.read_csv(os.path.join(OUT, f"{tag_a}_pairs.csv"))
    b = pd.read_csv(os.path.join(OUT, f"{tag_b}_contamination.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    cert = a[a.B_ins >= 0].sort_values("B_ins", ascending=False)
    x = np.arange(len(cert)); w = 0.2
    ax.bar(x - 1.5 * w, cert.D, w, color="#9fb3c8", label="raw breakdown, forged ($D$)")
    ax.bar(x - 0.5 * w, cert.B_ins, w, color="#1f4e79", label="certified, forged")
    ax.bar(x + 0.5 * w, cert.D_rep, w, color="#e3b89a", label=r"raw breakdown, flipped ($\lceil D/2\rceil$)")
    ax.bar(x + 1.5 * w, cert.B_rep, w, color="#b5541c", label="certified, flipped")
    ax.set_xticks(x); ax.set_xticklabels([f"{short[r.i]}\n> {short[r.j]}" for r in cert.itertuples()], fontsize=6.5)
    ax.set_ylabel("items"); ax.set_title(f"{title}: certified budgets", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    for ax, attack, title in ((axes[1], "burst_random", "forged items, one burst"),
                              (axes[2], "online_replace", "contaminated items (losses flipped)")):
        sub = b[b.attack == attack].groupby(["method", "B"]).rate.mean().reset_index()
        for m, lab, col in (("wald", "Wald interval", "#8b8a85"), ("naive", "undeflated betting", "#5c5c5c"),
                            ("perstep", "per-step robust", "#7a9a01"), ("avg_ins", "ours, insertion", "#1f4e79"),
                            ("avg_rep", "ours, replacement", "#b5541c")):
            s = sub[sub.method == m].sort_values("B")
            ax.plot(s.B, s.rate, marker="o", ms=3, lw=1.2, color=col, label=lab)
        ax.axhline(ALPHA, ls=":", color="k", lw=0.8)
        ax.set_xscale("symlog", linthresh=10); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("corrupted items $B$"); ax.set_ylabel("P(false claim), mean over pairs"); ax.set_title(title, fontsize=9)
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{out}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{out}.png"), dpi=160, bbox_inches="tight")
    print("wrote", os.path.join(OUT, f"{out}.pdf"), "and .png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fetch", "a", "b"]
    tab = None
    if "fetch" in which:
        tab = fetch()
    elif "a" in which or "b" in which:
        tab = load_items()
    if "a" in which:
        e7a(tab)
    if "b" in which:
        e7b(tab)
    if "fig" in which:
        figure()
