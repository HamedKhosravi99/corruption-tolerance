"""E8: frontier models on HELM Capabilities (release v1.15.0): MMLU-Pro, GPQA, IFEval, Omni-MATH.
The SWE-bench Verified part of E8 is in e8_swebench.py.

Data. HELM Capabilities release v1.15.0 (2025-11-24) evaluates 68 models. We use fifteen frontier models of
2025 from seven developers, chosen by developer and release from HELM's model list (not by any score):
    Claude Opus 4, Claude Sonnet 4, Claude Sonnet 4.5 (without extended thinking), o3, GPT-4.1, GPT-5,
    GPT-5.1, gpt-oss-120b, Gemini 2.5 Pro (03-25 preview), Gemini 3 Pro (preview), Grok 4, DeepSeek-R1-0528, Kimi K2
    (instruct), Qwen3-235B (FP8) and Qwen3-235B-2507 (instruct, FP8); HELM identifiers in MODELS.
HELM ran GPT-5.1 with the provider's default reasoning effort, which is none (no reasoning).
Per-item scores come from display_predictions.json:
    MMLU-Pro (1,000 sampled questions), GPQA (gpqa_main, 446 questions): 'chain_of_thought_correctness', 0 or 1;
    IFEval (541 prompts): 'ifeval_strict_accuracy', scored at the prompt level (1 only when every
        instruction is followed);
    Omni-MATH (1,000 sampled problems): 'omni_math_accuracy', a judged score of 0, 1/3, 1/2, 2/3 or 1
        (4.2% of the scores are fractional). The record is W = (1 + s_i - s_j) / 2
        (pair_outcomes_graded in e7_benchmark.py), which equals the E7 coding for 0/1 scores.
Mode `fetch` reads the release index, downloads each display_predictions.json once, keeps only the
per-item score, and caches it as data/helm_cap/<model>__<scenario>.json.gz (no model text). It writes
results/raw/e8_<scenario>_items.csv.gz.
History: until 2026-10-03 E8 used release v1.9.0 with the first eight models and three benchmarks.
Their items and scores are identical in v1.15.0 (checked on 2026-10-03 against the earlier tables).

Modes (all reuse the E7 code):
  a          clean budgets, raw breakdown and time to certify for all 210 ordered pairs, per scenario
             (results/e8a_<scen>_pairs.csv, _perm.csv; paths in results/raw/e8a_<scen>_paths.npz);
  c          simultaneous certificates over all 210 ordered pairs (closed testing, family fixed by the
             model set), with the Bonferroni split and the single-pair values (results/e8c_matrices.npz);
  b[:scen]   contamination replay as E7b on all unordered pairs (attacked model = the less accurate one;
             exact ties are skipped), on one scenario or all four (results/e8b_<scen>_contamination.csv);
  preview    a preview of the replays of E8 on all five benchmarks (results/e8_preview.pdf/.png).
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
import e7_benchmark as E7  # noqa: E402
import e4_arena as E4  # noqa: E402
import ccb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "data", "helm_cap")
BUCKET = "https://storage.googleapis.com/crfm-helm-public/"
RELEASE = "v1.15.0"
MODELS = {"claude-opus-4": "anthropic_claude-opus-4-20250514",
          "claude-sonnet-4": "anthropic_claude-sonnet-4-20250514",
          "o3": "openai_o3-2025-04-16",
          "gpt-4.1": "openai_gpt-4.1-2025-04-14",
          "gemini-2.5-pro": "google_gemini-2.5-pro-preview-03-25",
          "grok-4": "xai_grok-4-0709",
          "deepseek-r1": "deepseek-ai_deepseek-r1-0528",
          "qwen3-235b": "qwen_qwen3-235b-a22b-fp8-tput",
          "gpt-5.1": "openai_gpt-5.1-2025-11-13",
          "gpt-5": "openai_gpt-5-2025-08-07",
          "gemini-3-pro": "google_gemini-3-pro-preview",
          "claude-sonnet-4.5": "anthropic_claude-sonnet-4-5-20250929",
          "kimi-k2": "moonshotai_kimi-k2-instruct",
          "gpt-oss-120b": "openai_gpt-oss-120b",
          "qwen3-235b-2507": "qwen_qwen3-235b-a22b-instruct-2507-fp8"}
SHORT = {"claude-opus-4": "Claude Opus 4", "claude-sonnet-4": "Claude Sonnet 4", "o3": "o3", "gpt-4.1": "GPT-4.1",
         "gemini-2.5-pro": "Gemini 2.5 Pro", "grok-4": "Grok 4", "deepseek-r1": "DeepSeek-R1", "qwen3-235b": "Qwen3-235B",
         "gpt-5.1": "GPT-5.1", "gpt-5": "GPT-5", "gemini-3-pro": "Gemini 3 Pro", "claude-sonnet-4.5": "Claude Sonnet 4.5",
         "kimi-k2": "Kimi K2", "gpt-oss-120b": "gpt-oss-120b", "qwen3-235b-2507": "Qwen3-235B-2507"}
SCENARIOS = {"mmlu_pro": "chain_of_thought_correctness", "gpqa": "chain_of_thought_correctness",
             "ifeval": "ifeval_strict_accuracy", "omni_math": "omni_math_accuracy"}
OUTCOMES = {"mmlu_pro": E7.pair_outcomes, "gpqa": E7.pair_outcomes, "ifeval": E7.pair_outcomes,
            "omni_math": E7.pair_outcomes_graded}


def _get(path):
    req = urllib.request.Request(BUCKET + urllib.parse.quote(path), headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return json.loads(data)


def items_path(scen):
    return os.path.join(RAW, f"e8_{scen}_items.csv.gz")


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    idx = _get(f"capabilities/benchmark_output/releases/{RELEASE}/runs_to_run_suites.json")
    jobs = []
    for short, helm in MODELS.items():
        for scen in SCENARIOS:
            runs = [r for r in idx if r.startswith(scen + ":") and r.split("model=")[1].split(",")[0] == helm]
            assert len(runs) == 1, (helm, scen, runs)
            jobs.append((short, scen, runs[0], f"capabilities/benchmark_output/runs/{idx[runs[0]]}/{runs[0]}/display_predictions.json"))

    def one(job):
        short, scen, run, path = job
        local = os.path.join(CACHE, f"{short}__{scen}.json.gz")
        if not os.path.exists(local):
            keep = [{"instance_id": e["instance_id"], "train_trial_index": e.get("train_trial_index", 0),
                     "score": e["stats"][SCENARIOS[scen]]} for e in _get(path)]
            with gzip.open(local, "wt") as f:
                json.dump({"source": BUCKET + path, "release": RELEASE, "run": run, "stat": SCENARIOS[scen], "items": keep}, f)
        return job, local

    rows = []
    with ThreadPoolExecutor(8) as ex:
        for (short, scen, _, _), local in ex.map(one, jobs):
            for e in json.load(gzip.open(local, "rt"))["items"]:
                if e["train_trial_index"] != 0:
                    continue
                v = float(e["score"])
                if scen == "ifeval":
                    v = float(v == 1.0)   # prompt-level strict accuracy, as in E8
                rows.append((scen, e["instance_id"], short, v))
    df = pd.DataFrame(rows, columns=["scenario", "instance_id", "model", "correct"])
    tabs = {}
    for scen in SCENARIOS:
        t = df[df.scenario == scen].pivot_table(index="instance_id", columns="model", values="correct")
        assert t.notna().all().all(), f"{scen}: items missing for some model"
        t = t[list(MODELS)].reset_index()
        t.to_csv(items_path(scen), index=False, compression="gzip")
        tabs[scen] = t
        frac = int((~t[list(MODELS)].isin([0.0, 1.0])).sum().sum())
        print(f"{scen}: {len(t)} items; fractional scores: {frac}; accuracy {t[list(MODELS)].mean().round(3).to_dict()}")
    return tabs


def load_items():
    return {scen: pd.read_csv(items_path(scen)) for scen in SCENARIOS}


def closed_matrices(tabs, names, outcomes, out_file, short):
    """Simultaneous certificates over all ordered pairs of `names` (closed testing over all sets of
    pairs, the family fixed by the model set), per scenario, with the Bonferroni split and the
    single-pair values; as e8c in e8_frontier.py."""
    lams, wts = ccb.default_grid()
    li = np.array([ccb.costs(l)[0] for l in lams]); lr = np.array([ccb.costs(l)[1] for l in lams])
    out = {}
    M = len(names)
    pairs = np.array([(a, b) for a in range(M) for b in range(M) if a != b])
    for scen, tab in tabs.items():
        n = len(tab)
        order0 = np.random.default_rng(E7.SEED_ORDER).permutation(n)
        LK = np.zeros((len(pairs), len(lams)))
        for e, (a, b) in enumerate(pairs):
            w = outcomes[scen](tab, names[a], names[b], order0)
            for q, lam in enumerate(lams):
                LK[e, q] = ccb.log_wealth(w, lam)[-1]
        LW = np.tile(np.log(wts), (len(pairs), 1))
        for kind, cvec in (("ins", li), ("rep", lr)):
            C = np.tile(cvec, (len(pairs), 1))
            closed, bonf, single = E4.simultaneous_budgets_all(LW, LK, C)
            for name, vec in (("closed", closed), ("bonf", bonf), ("single", single)):
                mat = np.full((M, M), np.nan)
                mat[pairs[:, 0], pairs[:, 1]] = vec
                out[f"{scen}_B_{name}_{kind}"] = mat
            cl, si = out[f"{scen}_B_closed_{kind}"], out[f"{scen}_B_single_{kind}"]
            print(f"{scen} ({n} items), {kind}: certified {int(np.nansum(si >= 0))} single-pair, "
                  f"{int(np.nansum(cl >= 0))} simultaneously, of {len(pairs)} ordered pairs; "
                  f"largest simultaneous {int(np.nanmax(cl))}; closed minus Bonferroni at most "
                  f"{int(np.nanmax(cl - out[f'{scen}_B_bonf_{kind}']))}")
        pd.set_option("display.width", 300)
        print(pd.DataFrame(out[f"{scen}_B_closed_ins"], index=[short[m] for m in names],
                           columns=[short[m] for m in names]).to_string())
    np.savez_compressed(os.path.join(OUT, out_file), models=np.array(names), **out)
    return out


def preview():
    """Preview figure, not a paper figure: the E8 replays on all five benchmarks, probability of a
    false statement (mean over pairs) under forged items in one burst (top) and contaminated items
    (bottom). Reads only the saved replay files; writes results/e8_preview.pdf/.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("e8b_mmlu_pro", "MMLU-Pro (1,000)"), ("e8b_gpqa", "GPQA (446)"), ("e8b_ifeval", "IFEval (541)"),
              ("e8b_omni_math", "Omni-MATH (1,000)"), ("e8b_swebench", "SWE-bench Verified (500)")]
    styles = (("wald", "Wald interval", "#d55181"), ("naive", "undeflated betting, grid", "#eb6834"),
              ("perstep", "per-step robust", "#c98500"), ("avg_ins", "ours, insertion", "#2a78d6"),
              ("avg_rep", "ours, replacement", "#008300"))
    fig, axes = plt.subplots(2, len(panels), figsize=(13, 4.6), sharey=True)
    for c, (tag, title) in enumerate(panels):
        d = pd.read_csv(os.path.join(OUT, f"{tag}_contamination.csv"))
        npairs = d.groupby(["attacked", "opponent"]).ngroups
        for r, (attack, lab) in enumerate((("burst_random", "forged items, one burst"),
                                           ("online_replace", "contaminated items"))):
            ax = axes[r, c]
            sub = d[d.attack == attack].groupby(["method", "B"]).rate.mean().reset_index()
            for m, name, col in styles:
                s = sub[sub.method == m].sort_values("B")
                ax.plot(s.B, s.rate, marker="o", ms=2.5, lw=1.2, color=col, label=name)
            ax.axhline(ccb.ALPHA, ls=":", color="k", lw=0.8)
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(-1, None)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{title}, {npairs} pairs\n{lab}" if r == 0 else lab, fontsize=8)
            if r == 1:
                ax.set_xlabel("corrupted items $B$", fontsize=8)
            if c == 0:
                ax.set_ylabel("P(some false statement)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"e8_preview.{ext}"), bbox_inches="tight", dpi=150)
    print("wrote results/e8_preview.pdf and .png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fetch", "a", "c"]
    if which == ["preview"]:
        preview()
        sys.exit()
    tabs = fetch() if "fetch" in which else load_items()
    if "a" in which:
        for scen, tab in tabs.items():
            print(f"\n===== E8a {scen}: {len(tab)} items =====")
            E7.e7a(tab, models=MODELS, tag=f"e8a_{scen}", outcomes=OUTCOMES[scen], combined_raw=True)
    if "c" in which:
        closed_matrices(tabs, list(MODELS), OUTCOMES, "e8c_matrices.npz", SHORT)
    for arg in which:
        if arg == "b" or arg.startswith("b:"):
            for scen in ([arg[2:]] if arg.startswith("b:") else list(SCENARIOS)):
                print(f"\n===== E8b {scen} =====")
                E7.e7b(tabs[scen], models=MODELS, tag=f"e8b_{scen}", outcomes=OUTCOMES[scen])
