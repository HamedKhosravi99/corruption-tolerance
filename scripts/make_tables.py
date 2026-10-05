"""Write LaTeX tables for the appendix from the saved results."""
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
sys.path.insert(0, os.path.join(ROOT, "src"))
import names as model_names  # noqa: E402

RES = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)


def esc(s):
    return s.replace("_", r"\_")


def num(x):
    """Integer with thousands separators and a true minus sign (-1 marks 'not certified')."""
    v = int(round(float(x)))
    return f"{v:,}" if v >= 0 else "$-" + f"{-v:,}".replace(",", "{,}") + "$"


def dec(x, nd):
    """x rounded half up to nd decimals, as a string. The saved rates are exact decimals (counts over
    1,000 or 10,000 replays), so ties such as 0.0435 are rounded from their decimal value, not from
    the binary float (which would print 0.043)."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP))


def rate(x):
    """Probability with two decimals and no leading zero, as in the main-text table."""
    s = dec(x, 2)
    return "1.0" if Decimal(s) >= 1 else s.lstrip("0")


def tau_table():
    t = pd.read_csv(os.path.join(RES, "e2_tau.csv"))
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"$\delta$ & cost & $b$ & mean $\tau_b$ & union & upper \eqref{eq:tau-upper} & lower (Cor.~\ref{cor:lower-mean}) & median \\", r"\midrule"]
    for _, r in t[t.b.isin([0, 10, 50, 100, 400])].iterrows():
        lines.append(f"{r.delta:.2f} & {r.kind[:3]}. & {int(r.b)} & {r.mean_tau:,.0f} & {r.mean_tau_union:,.0f} & "
                     f"{r.upper:,.0f} & {r.lower:,.0f} & {r.median_tau:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_tau.tex"), "w").write("\n".join(lines))


def selection_table():
    s = pd.read_csv(os.path.join(RES, "e3_selection.csv"))
    p = s.pivot(index="k", columns="method", values="rate")
    se = s.pivot(index="k", columns="method", values="se")
    lines = [r"\begin{tabular}{rcc}", r"\toprule", r"$V$ & uncorrected & corrected \\", r"\midrule"]
    for k in p.index:
        lines.append(f"{k} & {dec(p.loc[k, 'naive'], 3)} ({dec(se.loc[k, 'naive'], 3)}) & "
                     f"{dec(p.loc[k, 'corrected'], 3)} ({dec(se.loc[k, 'corrected'], 3)}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_selection.tex"), "w").write("\n".join(lines))


def wald_counts(dataset):
    """E10: fixed-time Wald breakdown counts keyed by pair name 'x > y'; empty if E10 has not run."""
    path = os.path.join(RES, "e10_breakdown.csv")
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    d = d[d.dataset == dataset]
    return {p: (wf, wr) for p, wf, wr in zip(d.pair, d.wald_forged, d.wald_flipped)}


def wnum(v):
    return "--" if v != v else num(v)


def arena_pairs_table():
    a = pd.read_csv(os.path.join(RES, "e4a_pairs.csv"))
    wald = wald_counts("arena")
    lines = [r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
             r" & & & & & \multicolumn{2}{c}{edge} & \multicolumn{2}{c}{lead} & \multicolumn{2}{c}{Wald} & \\",
             r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"leader & other & votes & $\bar W$ & ties & forged & flipped & forged & flipped & forged & flipped & $\Delta$ \\",
             r"\midrule"]
    for _, r in a.iterrows():
        wf, wr = wald.get(f"{r.leader} > {r.other}", (np.nan, np.nan))
        lines.append(f"{esc(r.leader)} & {esc(r.other)} & {r.n:,} & {r.win_rate:.3f} & {r.tie_rate:.2f} & "
                     f"{num(r.B_ins)} & {num(r.B_rep)} & {num(r.B_lead_ins)} & {num(r.B_lead_rep)} & {wnum(wf)} & {wnum(wr)} & {num(r.point_estimate_flip)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_arena_pairs.tex"), "w").write("\n".join(lines))


def e1_table():
    d = pd.read_csv(os.path.join(RES, "e1_validity.csv"))
    names = {"naive": "Undefl.", "perstep": "Per-step", "ins": "Ins.\\ (1 bet)", "avg_ins": "Ins.\\ (grid)",
             "rep": "Rep.\\ (1 bet)", "avg_rep": "Rep.\\ (grid)"}
    att = {"burst_start": "burst at start", "burst_trigger": "burst at trigger", "spread": "spread",
           "online_replace": "flip losses"}
    methods = list(names)
    lines = [r"\begin{tabular}{lr" + "r" * len(methods) + "}", r"\toprule",
             "attack & $B$ & " + " & ".join(names[m] for m in methods) + r" \\", r"\midrule"]
    for a in att:
        for B in [0, 10, 20, 40, 100]:
            s = d[(d.attack == a) & (d.B == B)].set_index("method").rate
            lines.append(f"{att[a]} & {B} & " + " & ".join(dec(s[m], 3) for m in methods) + r" \\")
        lines.append(r"\midrule" if a != "online_replace" else "")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_e1.tex"), "w").write("\n".join(lines))


def rigging_table():
    t = pd.read_csv(os.path.join(RES, "e4b_table.csv"))
    names = {"wald": "Wald", "naive": "Undefl.", "perstep": "Per-step", "avg_ins": "Ins.", "avg_rep": "Rep."}
    att = {"burst_random": "burst", "spread": "spread", "online_replace": "flip"}
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             "promoted vs opponent & attack & " + " & ".join(names.values()) + r" \\", r"\midrule"]
    for _, r in t.iterrows():
        cells = [" / ".join(rate(r[f"{m}_{B}"]) for B in (0, 100, 400)) for m in names]   # as in Table 1
        x, y = r.pair.split(" vs ")
        pair = f"{model_names.arena(x)} vs {model_names.arena(y)}"
        lines.append(f"{pair if r.attack == 'burst_random' else ''} & {att[r.attack]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_rigging_full.tex"), "w").write("\n".join(lines))


def rigging_main_table():
    """Rigging summary table (Appendix C, with the per-pair table): mean over the three E4b pairs at B = 0 / 100 / 400, burst and flip."""
    t = pd.read_csv(os.path.join(RES, "e4b_table.csv"))
    mean = t.groupby("attack", sort=False).mean(numeric_only=True)
    names = {"wald": "Wald interval", "naive": "Undeflated betting", "perstep": "Per-step robust",
             "avg_ins": "Ours, insertion", "avg_rep": "Ours, replacement"}
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Method & Burst & Flip \\", r"\midrule"]
    for m, lab in names.items():
        cells = [" / ".join(rate(mean.loc[a, f"{m}_{B}"]) for B in (0, 100, 400)) for a in ("burst_random", "online_replace")]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_rigging.tex"), "w").write("\n".join(lines))


ARENA_SHORT = model_names.ARENA
MMLU_SHORT = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "qwen2.5-7b-instruct": "Qwen2.5-7B",
              "gemma-2-9b": "Gemma-2-9B", "mistral-7b-instruct-v0.3": "Mistral-7B"}


def suppress_table():
    """E6: de-certification budgets and damage per corrupted record (attacks on true claims)."""
    dec = pd.read_csv(os.path.join(RES, "e6_decert.csv"))
    sup = pd.read_csv(os.path.join(RES, "e6_suppress.csv"))
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"pair & $n$ & $\Bhat_n$ forged & $\Bhat_n$ flipped & de-certify by & de-certify by & loss of tolerance & loss of tolerance \\",
             r" & & & & inserted losses & replaced wins & per inserted loss & per replaced win \\",
             r"\midrule"]
    for (x, y), g in dec.groupby(["leader", "other"], sort=False):
        s = sup[(sup.leader == x) & (sup.other == y) & (sup.B == 100) & (sup.cost == "ins")]
        d_ins = float(s[s.attack == "insert_losses"].damage_per_record.iloc[0])
        d_flip = float(s[s.attack == "flip_wins"].damage_per_record.iloc[0])
        bi = int(g[(g.cost == "ins")].B_clean.iloc[0]); br = int(g[(g.cost == "rep")].B_clean.iloc[0])
        di = int(g[(g.attack == "insert_losses") & (g.cost == "ins")].decert_budget.iloc[0])
        df_ = int(g[(g.attack == "flip_wins") & (g.cost == "ins")].decert_budget.iloc[0])
        lines.append(f"{ARENA_SHORT[x]} $>$ {ARENA_SHORT[y]} & {int(g.n.iloc[0]):,} & {bi:,} & {br:,} & {di:,} & {df_:,} & {d_ins:.2f} & {d_flip:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_suppress.tex"), "w").write("\n".join(lines))


def mmlu_table():
    """E7: corruption tolerances on HELM MMLU for the six pairs, oriented by accuracy."""
    a = pd.read_csv(os.path.join(RES, "e7a_pairs.csv"))
    pp = pd.read_csv(os.path.join(RES, "e7a_perm.csv"))
    acc_order = ["qwen2.5-7b-instruct", "gemma-2-9b", "mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct"]
    wald = wald_counts("mmlu")
    lines = [r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
             r" & & & & & \multicolumn{2}{c}{edge} & \multicolumn{2}{c}{lead} & \multicolumn{2}{c}{Wald} & \multicolumn{2}{c}{$\tau_0$} \\",
             r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
             r"pair & wins & losses & ties & $\Delta$ & forged & contam. & forged & contam. & forged & contam. & seed 0 & median \\",
             r"\midrule"]
    for ai, i in enumerate(acc_order):
        for j in acc_order[ai + 1:]:
            r = a[(a.i == i) & (a.j == j)].iloc[0]
            q = pp[(pp.i == i) & (pp.j == j) & (pp.cost == "ins")]
            med = q.tau0.fillna(np.inf).median() if len(q) else np.inf   # never certified counts as infinite
            tau_med = "--" if not np.isfinite(med) else (f"{med:,.1f}" if med % 1 else f"{int(med):,}")
            cell = lambda v: f"{int(v):,}" if v >= 0 else "--"
            tau0 = f"{int(r.tau0_ins):,}" if r.B_ins >= 0 and not np.isnan(r.tau0_ins) else "--"
            lines.append(f"{MMLU_SHORT[i]} $>$ {MMLU_SHORT[j]} & {int(r.wins):,} & {int(r.losses):,} & {int(r.ties):,} & {int(r.D):,} & "
                         f"{cell(r.B_ins)} & {cell(r.B_rep)} & {cell(r.B_lead_ins)} & {cell(r.B_lead_rep)} & {wnum(wald.get(f'{i} > {j}', (np.nan, np.nan))[0])} & {wnum(wald.get(f'{i} > {j}', (np.nan, np.nan))[1])} & {tau0} & {tau_med} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_mmlu.tex"), "w").write("\n".join(lines))


def frontier_table():
    """E8: certificates on the five frontier benchmarks (four from HELM v1.15.0, SWE-bench Verified).
    Counts of certified ordered pairs, pair by pair and simultaneously (closed testing over all
    ordered pairs of the benchmark's models), and simultaneous certificates (forged / contaminated)."""
    rows = [("MMLU-Pro", "mmlu_pro", "e8c_matrices.npz"), ("GPQA", "gpqa", "e8c_matrices.npz"),
            ("IFEval", "ifeval", "e8c_matrices.npz"), ("Omni-MATH", "omni_math", "e8c_matrices.npz"),
            ("SWE-bench Verified", "swebench", "e8c_swebench_matrices.npz")]
    cell = lambda a, b: "--" if a < 0 else f"{int(a):,} / {int(b):,}"
    lines = [r"\begin{tabular}{lrrrrccc}", r"\toprule",
             r"benchmark & items & models & certified pairs & certified pairs & top two & largest tolerance & largest tolerance \\",
             r" & & & (pair by pair) & (simultaneously) & & in the top five & overall \\", r"\midrule"]
    for label, scen, mat in rows:
        z = np.load(os.path.join(RES, mat), allow_pickle=True)
        names = list(z["models"])
        tab = pd.read_csv(os.path.join(RES, "raw", f"e8_{scen}_items.csv.gz"))
        acc = tab[names].mean().sort_values(ascending=False, kind="stable")
        ix = {m: k for k, m in enumerate(names)}
        ci, cr, si = z[f"{scen}_B_closed_ins"], z[f"{scen}_B_closed_rep"], z[f"{scen}_B_single_ins"]
        top2 = (ix[acc.index[0]], ix[acc.index[1]])
        top5 = [ix[m] for m in acc.index[:5]]
        best5 = max(((ci[a, b], cr[a, b]) for a in top5 for b in top5 if a != b), key=lambda x: x[0])
        k = np.nanargmax(ci)
        a, b = np.unravel_index(k, ci.shape)
        lines.append(f"{label} & {len(tab):,} & {len(names)} & {int(np.nansum(si >= 0))} of {len(names) * (len(names) - 1)} & "
                     f"{int(np.nansum(ci >= 0))} & {cell(ci[top2], cr[top2])} & {cell(*best5)} & {cell(ci[a, b], cr[a, b])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_frontier.tex"), "w").write("\n".join(lines))



def average_table():
    """E9: Bhat_t / t at the last checkpoint for the betting and the exponential wealth (median and
    10-90% range over the repetitions) and the limit rates of Theorems 5.3 and 5.7."""
    path = os.path.join(RES, "e9_average_rate.csv")
    if not os.path.exists(path):
        print("e9_average_rate.csv not found; tab_average.tex not written")
        return
    d = pd.read_csv(path)
    d = d[d.t == d.t.max()]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"$\delta$ & ties & cost & edge \eqref{eq:bhat-grid} & lead \eqref{eq:zwealth} & limit rates (edge, lead) \\",
             r"\midrule"]
    for (delta, pi), g in d.groupby(["delta", "pi"], sort=True):
        for kind in ("insertion", "replacement"):
            b = g[(g.kind == kind) & (g.family == "bet")].iloc[0]
            e = g[(g.kind == kind) & (g.family == "exp")].iloc[0]
            lines.append(f"{delta:.2f} & {int(round((1 - pi) * 100))}\\% & {kind[:3]}. & "
                         f"{b['median']:.3f} ({b.q10:.3f}--{b.q90:.3f}) & {e['median']:.3f} ({e.q10:.3f}--{e.q90:.3f}) & "
                         f"{b.limit_rate:.3f}, {e.limit_rate:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_average.tex"), "w").write("\n".join(lines))

if __name__ == "__main__":
    tau_table()
    average_table()
    selection_table()
    arena_pairs_table()
    e1_table()
    rigging_table()
    rigging_main_table()
    suppress_table()
    mmlu_table()
    frontier_table()
    print("tables written")
