# Corruption Tolerance for Online Ranking of Large Language Models: code and results

This repository contains the code, the input data and the saved results for the paper
*Corruption Tolerance for Online Ranking of Large Language Models* (anonymous submission).
Every figure and table of the paper is rebuilt from the saved results in `results/` by two
scripts, and every experiment can be rerun from `experiments/` with fixed seeds.

## Contents

1. [Overview](#1-overview)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick start: rebuild the figures and tables](#4-quick-start-rebuild-the-figures-and-tables)
5. [Data](#5-data)
6. [Reproducing the experiments](#6-reproducing-the-experiments)
7. [Map from paper results to code and files](#7-map-from-paper-results-to-code-and-files)
8. [The core library](#8-the-core-library)
9. [Naming conventions](#9-naming-conventions)
10. [Computing environment and determinism](#10-computing-environment-and-determinism)

## 1. Overview

A *corruption tolerance* is a number $\hat B_t$ published with each pairwise claim about two models.
With probability at least $1-\alpha$, at all times $t$, the claim is correct or more than
$\hat B_t$ of the first $t$ records were corrupted. The code implements

* the **edge tolerance** (paper Sections 4.1 and 4.2, equation (9)): deflated capped betting over a
  grid of 14 constant bets $\bar\lambda_k = 2^{-(k-1)/2}$ with equal weights, with the insertion cost
  $c_{\rm ins}(\bar\lambda)=\log(1+\bar\lambda/2)$ or the replacement cost
  $c_{\rm rep}(\bar\lambda)=\log\frac{2+\bar\lambda}{2-\bar\lambda}$;
* the **lead tolerance** (Section 4.3, Theorem 5.7): the same construction on the exponential wealth
  $Z^{(\gamma)}_t=\exp\big(\gamma\sum_{s\le t}(W_s-\tfrac12)-\gamma^2t/8\big)$, with the costs
  $\gamma/2-\gamma^2/8$ (insertion) and $\gamma$ (replacement);
* the closed-testing and Bonferroni versions for whole leaderboards (Proposition 5.6, Corollary B.6);
* the attacks, the baselines (undeflated betting, a per-step robust test, a repeated Wald interval)
  and the experiments E1 to E10 of Section 6 and Appendix C.

## 2. Repository layout

```
.
├── README.md
├── requirements.txt          pinned Python dependencies
├── Makefile                  shortcuts for the commands of Sections 4 and 6
├── src/                      core library (imported by every script)
│   ├── ccb.py                wealths, costs, edge and lead tolerances, E1 attackers
│   ├── names.py              display names of Chatbot Arena models
│   └── arena_extract.py      streams the public Arena battle file to a compact CSV
├── experiments/              one script per experiment (E1 to E10)
├── scripts/
│   ├── make_figures.py       the paper's figures, from saved results only
│   └── make_tables.py        the paper's tables (LaTeX), from saved results only
├── data/
│   ├── helm_mmlu/            HELM MMLU per-instance results (228 gzipped JSON files)
│   ├── helm_cap/             HELM Capabilities per-item scores (60 gzipped JSON files)
│   ├── swebench_verified/    SWE-bench Verified run metadata (YAML) and per-instance results (gzipped JSON)
│   └── arena/                Chatbot Arena votes: not redistributed, see data/arena/README.md
├── results/
│   ├── *.csv, *.npz          summary results of each experiment
│   ├── *_preview.pdf/.png    quick-look plots written by some experiments (not paper figures)
│   ├── raw/                  per-repetition and per-item outputs (NumPy archives, gzipped CSV)
│   └── logs/                 standard output of the runs
├── figures/                  the paper's figures (PDF), written by scripts/make_figures.py
├── tables/                   the paper's tables (LaTeX), written by scripts/make_tables.py
└── cluster/                  optional HTCondor job files
```

## 3. Installation

Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root. They locate `src/`, `data/` and `results/`
relative to their own location, so no installation step beyond the dependencies is needed.

## 4. Quick start: rebuild the figures and tables

```bash
make paper            # = python scripts/make_figures.py && python scripts/make_tables.py
```

This rebuilds `figures/*.pdf` and `tables/*.tex` from `results/` in about a minute, without
rerunning any experiment. `make_tables.py` reads `results/e4b_table.csv`, which
`make_figures.py` writes, so the figures come first. Rebuilding from the committed results gives
tables identical to the committed ones and figures that differ only in their creation date.

## 5. Data

| Source | Used by | In this repository | Provenance |
|---|---|---|---|
| Chatbot Arena public battle file `clean_battle_20240814_public.json` (1,799,991 votes, 129 models, 2023-04-24 to 2024-08-14) | E4, E6 | no (no license stated; see [data/arena/README.md](data/arena/README.md)) | `https://storage.googleapis.com/arena_external_data/public/` |
| HELM MMLU per-instance results, four open-weight models, 57 subjects | E7 | `data/helm_mmlu/` (228 gzipped `per_instance_stats.json`) | HELM MMLU leaderboard, release index v1.13.0; `python experiments/e7_benchmark.py fetch` downloads them again |
| HELM Capabilities per-item scores, 15 models, MMLU-Pro, GPQA, IFEval, Omni-MATH | E8 | `data/helm_cap/` (60 gzipped JSON, per-item scores only) | HELM Capabilities, release index v1.15.0; `python experiments/e8_frontier.py fetch` |
| SWE-bench Verified runs of one agent with 11 models | E8 | `data/swebench_verified/` (`metadata.yaml` and gzipped `per_instance_details.json` per run) | `SWE-bench/experiments`, `evaluation/verified/`; `python experiments/e8_swebench.py fetch` |

The per-item tables derived from these files are saved as `results/raw/e7_mmlu_items.csv.gz`
and `results/raw/e8_<benchmark>_items.csv.gz`.

## 6. Reproducing the experiments

Each command writes its summary results to `results/` and its per-repetition outputs to
`results/raw/`. The run times are those of the original runs (Section 10). Redirect standard
output to `results/logs/` to keep a log, for example
`python experiments/e1_validity.py | tee results/logs/e1_log.txt`.

| Exp. | What it does | Command | Run time | Outputs |
|---|---|---|---|---|
| E1 | validity of each method under four attacks on null data | `python experiments/e1_validity.py` | ~45 min (8 processes) | `e1_validity.csv`, `raw/e1_<attack>_B<B>.npz` |
| E2 | growth of the tolerance and time to certify on i.i.d. records | `python experiments/e2_rate.py` | ~15 min | `e2_rate.csv`, `e2_tau.csv`, `raw/e2_delta<δ>.npz` |
| E3 | publishing the best of $V$ variants, with and without the $V/\alpha$ correction | `python experiments/e3_selection.py` | ~6 min | `e3_selection.csv`, `raw/e3_selection.npz` |
| E4 | counts quoted in the paper (votes, models, dates, ties) | `python experiments/e4_arena.py stats` | ~5 s | standard output (`logs/e4_data_log.txt`) |
| E4a | edge and lead tolerances on the real votes of the 20 most compared pairs | `python experiments/e4_arena.py a` | ~15 s | `e4a_pairs.csv`, `raw/e4a_paths.npz` |
| E4b | rigging replay on three near-boundary pairs | `python experiments/e4_arena.py b` | ~20 min | `e4b_rigging.csv`, `raw/e4b_pair<k>_B<B>.npz` |
| E4c | simultaneous tolerances over all 16,512 ordered pairs (closed testing and Bonferroni) | `python experiments/e4_arena.py call` | ~30 s | `e4c_matrix_all.npz` |
| E4c | the same over the 90 pairs of the ten displayed models only (comparison) | `python experiments/e4_arena.py c` | ~30 s | `e4c_matrix.npz` |
| E4c | lead tolerance with Bonferroni over all pairs (computed, not reported in the paper) | `python experiments/e4_arena.py lead` | ~20 s | `e4c_lead_matrix_all.npz` |
| E5 | regime shift (win probability 0.6, then 0.4), edge and lead tolerances | `python experiments/e5_regime.py --figure` | ~1 min | `e5_regime.csv`, `raw/e5_regime.npz`, `e5_regime_preview.*` |
| E6 | attacks on correct claims on the real votes | `python experiments/e6_suppress.py` | ~10 s | `e6_suppress.csv`, `e6_decert.csv`, `raw/e6_paths.npz` |
| E7 | HELM MMLU: tolerances, time to certify, rigging replay | `python experiments/e7_benchmark.py fetch a b fig` | a: ~40 s, b: ~36 min | `raw/e7_mmlu_items.csv.gz`, `e7a_pairs.csv`, `e7a_perm.csv`, `raw/e7a_<pair>.npz`, `e7b_contamination.csv`, `e7_preview.*` |
| E8 | HELM Capabilities: tolerances, simultaneous matrices, replays | `python experiments/e8_frontier.py fetch a c b` | a, c: ~1 min; b: ~1 h (or `b:<benchmark>` per process) | `raw/e8_<benchmark>_items.csv.gz`, `e8a_<benchmark>_pairs.csv`, `e8a_<benchmark>_perm.csv`, `raw/e8a_<benchmark>_paths.npz`, `e8c_matrices.npz`, `e8b_<benchmark>_contamination.csv` |
| E8 | SWE-bench Verified: the same for 11 models under one agent | `python experiments/e8_swebench.py fetch a c b` | b: ~15 min | `e8_swebench_runs.csv`, `raw/e8_swebench_items.csv.gz`, `e8a_swebench_*.csv`, `raw/e8a_swebench_paths.npz`, `e8c_swebench_matrices.npz`, `e8b_swebench_contamination.csv` |
| E8 | quick-look plot of the five replays (not a paper figure) | `python experiments/e8_frontier.py preview` | ~5 s | `e8_preview.*` |
| E9 | lead tolerance against edge tolerance on i.i.d. records, with and without ties | `python experiments/e9_average_rate.py` | ~2 min | `e9_average_rate.csv`, `raw/e9_average.npz` |
| E10 | fixed-time Wald and Hoeffding breakdown counts beside the certificates | `python experiments/e10_breakdown.py` | ~1 s | `e10_breakdown.csv` |

E4 and E6 need the Arena extract (Section 5). E10 reads `e4a_pairs.csv` and `e7a_pairs.csv`.
`make experiments` runs everything that does not need the Arena data or the network.

## 7. Map from paper results to code and files

All paths are relative to the repository root. "Script" is the experiment that produces the
saved results; "Built by" is the function that turns them into the paper's figure or table.

### Main text

| Paper item | Content | Script | Saved results | Built by → file |
|---|---|---|---|---|
| Figure 1 | probability of a false statement under three attacks, six methods | `experiments/e1_validity.py` | `results/e1_validity.csv` | `make_figures.py: fig_validity` → `figures/fig_validity.pdf` |
| Figure 2 (a, b) | tolerance per record and records needed to certify, with the bounds of Theorem 5.3 and Corollary B.4 | `experiments/e2_rate.py` | `results/e2_rate.csv`, `results/e2_tau.csv` | `make_figures.py: fig_rates_arena` → `figures/fig_rates_arena.pdf` |
| Figure 2 (c) | tolerance of four claims on real Arena votes over time | `experiments/e4_arena.py a` | `results/raw/e4a_paths.npz` | `make_figures.py: fig_rates_arena` → `figures/fig_rates_arena.pdf` |
| Table 1 | Arena rigging replay, mean over three pairs | `experiments/e4_arena.py b` | `results/e4b_rigging.csv` → `results/e4b_table.csv` | `make_tables.py: rigging_main_table` → `tables/tab_rigging.tex` |
| Section 6.1 numbers | false-statement probabilities quoted in the text | `experiments/e1_validity.py` | `results/e1_validity.csv` | — |
| Section 6.2 numbers | rates at $t=8{,}000$ and gaps to the bounds | `experiments/e2_rate.py` | `results/e2_rate.csv`, `results/e2_tau.csv` | — |
| Section 6.3, Arena | Llama-3-70B over Llama-3-8B: edge 2,056 / 1,009, lead 1,940 / 957, completed-set count 2,286, Wald 2,139; GPT-4o (2024-05-13) not certified above Claude-3.5-Sonnet simultaneously | `experiments/e4_arena.py stats a call`, `experiments/e10_breakdown.py` | `results/e4a_pairs.csv` (`B_ins`, `B_rep`, `B_lead_ins`, `B_lead_rep`, `point_estimate_flip`), `results/e10_breakdown.csv`, `results/e4c_matrix_all.npz` | — |
| Section 6.3, rigging | per-pair maxima 2.2% and 2.1% | `experiments/e4_arena.py b` | `results/e4b_rigging.csv` | — |
| Section 6.3, MMLU | 2,264 / 1,102 (lead 2,081 / 1,026), completed-set counts 2,414 / 1,207 | `experiments/e7_benchmark.py a` | `results/e7a_pairs.csv` (`D`, `B_ins`, `B_rep`, `B_lead_ins`, `B_lead_rep`) | — |
| Section 6.3, frontier benchmarks | no top-two ordering certified simultaneously; top-five maxima 17 forged / 7 contaminated | `experiments/e8_frontier.py`, `experiments/e8_swebench.py` | `results/e8c_matrices.npz`, `results/e8c_swebench_matrices.npz` | see Table 10 |

### Appendix C

| Paper item | Content | Script | Saved results | Built by → file |
|---|---|---|---|---|
| Table 2 | map of the experiments (text only) | — | — | — |
| Table 3 | E1, all attacks and budgets | `experiments/e1_validity.py` | `results/e1_validity.csv`, `results/raw/e1_*.npz` | `make_tables.py: e1_table` → `tables/tab_e1.tex` |
| Table 4 | E2, mean and median records to certify, union variant, upper and lower bounds | `experiments/e2_rate.py` | `results/e2_tau.csv` | `make_tables.py: tau_table` → `tables/tab_tau.tex` |
| Table 5 | E3, best of $V$ variants | `experiments/e3_selection.py` | `results/e3_selection.csv` | `make_tables.py: selection_table` → `tables/tab_selection.tex` |
| Table 6 | E4a, edge, lead and Wald counts for 20 Arena pairs | `experiments/e4_arena.py a`, `experiments/e10_breakdown.py` | `results/e4a_pairs.csv`, `results/e10_breakdown.csv` | `make_tables.py: arena_pairs_table` → `tables/tab_arena_pairs.tex` |
| Figure 3 | E4c, simultaneous tolerances of the ten displayed models (family of all 16,512 ordered pairs) | `experiments/e4_arena.py call` | `results/e4c_matrix_all.npz` | `make_figures.py: fig_matrix` → `figures/fig_arena_matrix.pdf` |
| Table 7 | E4b, rigging replay per pair | `experiments/e4_arena.py b` | `results/e4b_rigging.csv` → `results/e4b_table.csv` | `make_tables.py: rigging_table` → `tables/tab_rigging_full.tex` |
| Figure 4 | E5, regime shift: edge and lead tolerances, empirical win rates | `experiments/e5_regime.py` | `results/raw/e5_regime.npz`, `results/e5_regime.csv` | `make_figures.py: fig_regime` → `figures/fig_regime.pdf` |
| Table 8 | E6, attacks on correct claims | `experiments/e6_suppress.py` | `results/e6_suppress.csv`, `results/e6_decert.csv` | `make_tables.py: suppress_table` → `tables/tab_suppress.tex` |
| Table 9 | E7, MMLU: edge, lead and Wald counts, time to certify | `experiments/e7_benchmark.py a`, `experiments/e10_breakdown.py` | `results/e7a_pairs.csv`, `results/e7a_perm.csv`, `results/e10_breakdown.csv`; input `data/helm_mmlu/*.json.gz` | `make_tables.py: mmlu_table` → `tables/tab_mmlu.tex` |
| Figure 5 | E7 and E8, replays on resampled benchmark items | `experiments/e7_benchmark.py b`, `experiments/e8_frontier.py b`, `experiments/e8_swebench.py b` | `results/e7b_contamination.csv`, `results/e8b_<benchmark>_contamination.csv` | `make_figures.py: fig_benchmark_replay` → `figures/fig_benchmark_replay.pdf` |
| Table 10 | E8, summary per frontier benchmark | `experiments/e8_frontier.py a c`, `experiments/e8_swebench.py a c` | `results/e8a_<benchmark>_pairs.csv`, `results/e8c_matrices.npz`, `results/e8c_swebench_matrices.npz`, `results/raw/e8_<benchmark>_items.csv.gz`; inputs `data/helm_cap/*.json.gz`, `data/swebench_verified/` | `make_tables.py: frontier_table` → `tables/tab_frontier.tex` |
| Figure 6 | E8, top-six certificate grids with certified rank bounds | as Table 10 | as Table 10 | `make_figures.py: fig_frontier` → `figures/fig_frontier.pdf` |
| Table 11 | E9, lead against edge tolerance per record, limit rates | `experiments/e9_average_rate.py` | `results/e9_average_rate.csv`, `results/raw/e9_average.npz` | `make_tables.py: average_table` → `tables/tab_average.tex` |
| E5 text (lead tolerance) | status agreement, medians at the switch, runs certified after $t=2{,}000$ | `experiments/e5_regime.py` | `results/e5_regime.csv` (rows `exp_ins`, `exp_rep`) | — |
| E10 text | edge and lead tolerances against the Wald count | `experiments/e10_breakdown.py` | `results/e10_breakdown.csv` | Wald columns of Tables 6 and 9 |

Tables and figures are numbered as in the submitted PDF.

## 8. The core library

`src/ccb.py` holds everything that the experiments share.

| Function | Purpose |
|---|---|
| `default_grid()` | the 14 constant bets $\bar\lambda_k=2^{-(k-1)/2}$ and equal weights |
| `costs(lam)` | insertion and replacement costs of the betting wealth |
| `log_wealth(W, lam)` | log of the betting wealth $K_t$ for a stream of outcomes |
| `exp_costs(gam)`, `exp_log_wealth(W, gam)` | costs and log wealth of the exponential wealth $Z_t$ (lead tolerance) |
| `avg_certified_budget(W, kind, family=...)` | the tolerance (9): each bet deflated by its own cost, then averaged; `kind` is `"insertion"` or `"replacement"`; `family="bet"` gives the edge tolerance, `family="exp"` the lead tolerance; `logk` adds a selection correction |
| `avg_violation(W, flag, kind, ...)` | largest margin $\log\Phi_t(N_t)-\log(1/\alpha)$ over time; a false statement occurs if it is nonnegative |
| `grid_certified_budget`, `grid_violation` | the union-over-bets variant used for comparison in E2 |
| `single_violation` | single-bet versions, used by E1 |
| `attack_*`, `optimal_offline_replace_violation` | the E1 attackers |

## 9. Naming conventions

The code predates some of the paper's terms.

| In the code | In the paper |
|---|---|
| certified budget, `B`, `B_ins`, `B_rep` | corruption tolerance $\hat B_t$ (edge tolerance) with the insertion or the replacement cost |
| `B_lead_ins`, `B_lead_rep`, `family="exp"`, `exp_*` | lead tolerance (Theorem 5.7) |
| `avg_*`, `avg_ins`, `avg_rep` | the grid tolerance (9) ("Ours, grid") |
| `ins`, `rep` (single bet) | "Ours, one bet" with $\lambda=0.2$ |
| `naive` | undeflated betting |
| `perstep` | per-step robust test |
| `wald` | repeated Wald interval |
| `D`, `point_estimate_flip` | wins minus losses $\Delta$ |
| `tau0`, `tau<b>` | records until the tolerance first reaches $0$ or $b$ |

## 10. Computing environment and determinism

* E1 to E4 (except the all-pairs closed test of E4c) ran on a 12-core workstation (Windows, Python 3).
* The all-pairs closed test of E4c, E6 to E8, and the figures and tables ran on a laptop (macOS,
  Python 3.9.6) with the versions pinned in `requirements.txt`.
* E5, E9, E10, the lead-tolerance runs of E4a, E4c and E7, and the reruns that reproduce the earlier
  numbers ran as four-core HTCondor jobs on a Linux cluster (Python 3.14.7, NumPy 2.5.3,
  pandas 3.0.5). The job files are in `cluster/`.

Every script uses fixed seeds (`np.random.default_rng(<seed>)`). The reported numbers are
reproduced exactly across these machines: every rerun on the cluster or the laptop gave the same
summary results. Continuous per-repetition scores saved in `results/raw/` can differ in the last
digits across NumPy versions and platforms, which never changes a reported number.

## License and citation

To be added after the review period.
