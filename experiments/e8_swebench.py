"""E8, SWE-bench Verified part: certified budgets and contamination replay for frontier models of
2025 and 2026 under one fixed agent (the HELM part of E8 is in e8_frontier.py). Run 2026-10-03.

Data. The SWE-bench maintainers publish the runs of the bash-only agent mini-SWE-agent v2 in the
repository SWE-bench/experiments (evaluation/verified/<run>/), one run per model, on the same
500 tasks of SWE-bench Verified. Each run has metadata.yaml (model name, release date, agent
version, reported % resolved) and per_instance_details.json (resolved: true/false per task).
No license is stated for that repository. We keep only the resolved flags.
Mode `fetch` lists the mini-v2 runs, downloads both files, caches them gzipped under
data/swebench_verified/, and keeps a run only if its per-task file has all 500 tasks and its
resolved share equals the reported % resolved (to 0.1 percentage point). On 2026-10-03 this
drops three of the 14 runs: GPT-5.2-codex (no per-task file) and Gemini 3 Pro (the per-task file
marks all 500 tasks unresolved, while metadata.yaml reports 69.6%), and Gemini 3.5 Flash (its
per-task file has 441 of the 500 tasks, and it used agent version 2.4.2, not 2.0.0). It writes
results/raw/e8_swebench_items.csv.gz (tasks x models, 1 = resolved) and results/e8_swebench_runs.csv.

Record. One task. For the ordered pair (i, j): W = 1 if i resolves it and j does not, 0 for the
reverse, 1/2 otherwise; the order is a seed-0 permutation, as in E7.
Modes: a (clean budgets, all ordered pairs), c (simultaneous certificates over all ordered pairs,
closed testing), b (contamination replay as E7b on all unordered pairs; the attacked model is
the one with the lower resolve rate, exact ties skipped). The preview figure is made by
`python e8_frontier.py preview`.
"""
import gzip
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_benchmark as E7  # noqa: E402
from e8_frontier import closed_matrices  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "data", "swebench_verified")
ITEMS = os.path.join(RAW, "e8_swebench_items.csv.gz")
API = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
RAWGH = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def _meta(text):
    """The few fields of metadata.yaml that we use (flat 'key: value' lines under info:)."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        for key in ("name", "resolved", "model_release_date", "mini-swe-agent_version"):
            if s.startswith(key + ":"):
                out[key] = s.split(":", 1)[1].strip().strip("'\"")
    return out


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    runs = sorted(x["name"] for x in json.loads(_get(API)) if "_mini-v2" in x["name"])
    rows, flags = [], {}
    for run in runs:
        meta_local = os.path.join(CACHE, f"{run}__metadata.yaml")
        if not os.path.exists(meta_local):
            open(meta_local, "wb").write(_get(f"{RAWGH}/{run}/metadata.yaml"))
        meta = _meta(open(meta_local).read())
        local = os.path.join(CACHE, f"{run}__per_instance_details.json.gz")
        if not os.path.exists(local):
            files = [x["name"] for x in json.loads(_get(f"{API}/{run}"))]
            if "per_instance_details.json" in files:
                det = json.loads(_get(f"{RAWGH}/{run}/per_instance_details.json"))
                with gzip.open(local, "wt") as f:
                    json.dump({k: bool(v.get("resolved")) for k, v in det.items()}, f)
        res = json.load(gzip.open(local, "rt")) if os.path.exists(local) else None
        reported = float(meta.get("resolved", "nan"))
        share = 100.0 * sum(res.values()) / len(res) if res else float("nan")
        keep = bool(res) and len(res) == 500 and abs(share - reported) <= 0.1
        if keep:
            reason = ""
        elif not res:
            reason = "no per-task file"
        elif len(res) != 500:
            reason = f"per-task file has {len(res)} of 500 tasks"
        else:
            reason = f"per-task share {share:.1f}% differs from reported {reported}%"
        rows.append(dict(run=run, model=meta.get("name"), release=meta.get("model_release_date"),
                         agent=meta.get("mini-swe-agent_version"), reported=reported, per_task_share=share,
                         kept=keep, reason=reason))
        if keep:
            flags[meta["name"]] = res
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(os.path.join(OUT, "e8_swebench_runs.csv"), index=False)
    print(runs_df.to_string(index=False))
    tasks = sorted(next(iter(flags.values())))
    assert all(sorted(v) == tasks for v in flags.values()), "runs cover different tasks"
    tab = pd.DataFrame({m: [float(v[t]) for t in tasks] for m, v in flags.items()}, index=tasks)
    tab.index.name = "instance_id"
    tab = tab.reset_index()
    tab.to_csv(ITEMS, index=False, compression="gzip")
    print(f"{len(tab)} tasks x {len(flags)} models; resolve rate:",
          tab[list(flags)].mean().round(3).sort_values(ascending=False).to_dict())
    return tab


def load_items():
    return pd.read_csv(ITEMS)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fetch", "a", "c"]
    tab = fetch() if "fetch" in which else load_items()
    models = [c for c in tab.columns if c != "instance_id"]
    short = {m: m for m in models}
    if "a" in which:
        E7.e7a(tab, models=models, tag="e8a_swebench", combined_raw=True)
    if "c" in which:
        closed_matrices({"swebench": tab}, models, {"swebench": E7.pair_outcomes}, "e8c_swebench_matrices.npz", short)
    if "b" in which:
        E7.e7b(tab, models=models, tag="e8b_swebench")
