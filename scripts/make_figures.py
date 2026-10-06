"""Make all paper figures and tables from the saved results (no simulation here)."""
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
sys.path.insert(0, os.path.join(ROOT, "src"))
import names as model_names  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "lines.linewidth": 1.4, "pdf.fonttype": 42,
})

# Fixed entity -> style (colour follows the method, never its rank). Every method also has its own
# marker shape, and the single-bet variants are dashed with hollow markers, so that each curve can be
# told apart in black-and-white print.
STYLE = {
    "naive": dict(color="#eb6834", marker="o", ls="-", label="Undeflated betting"),
    "perstep": dict(color="#c98500", marker="s", ls="-", label=r"Per-step robust ($\varepsilon=B/n$)"),
    "wald": dict(color="#d55181", marker="v", ls="-", label="Wald interval, repeated looks"),
    "ins": dict(color="#2a78d6", marker="^", ls="--", hollow=True, label=r"Ours, insertion, one bet"),
    "avg_ins": dict(color="#2a78d6", marker="X", ls="-", label=r"Ours, insertion, grid"),
    "rep": dict(color="#008300", marker="D", ls="--", hollow=True, label=r"Ours, replacement, one bet"),
    "avg_rep": dict(color="#008300", marker="*", ls="-", label=r"Ours, replacement, grid"),
    # the average tolerance (exponential wealth, Theorem 5.7): dotted, hollow markers
    "exp_ins": dict(color="#2a78d6", marker="P", ls=":", hollow=True, label=r"Lead, insertion, grid"),
    "exp_rep": dict(color="#008300", marker="h", ls=":", hollow=True, label=r"Lead, replacement, grid"),
}
ALPHA = 0.05
MS = 4.1    # marker size (points); stars and crosses are drawn larger so that all shapes look alike


def mk(marker, hollow=False, ms=MS):
    """Marker keywords: a filled marker with a thin white edge, or a hollow one (white face)."""
    kw = dict(marker=marker, ms=ms * {"*": 1.8, "X": 1.15, "P": 1.15, "D": 0.85}.get(marker, 1.0))
    kw.update(dict(mfc="white", mew=0.8) if hollow else dict(mec="white", mew=0.35))
    return kw


def smk(st, ms=MS):
    """Marker keywords of a method style."""
    return mk(st["marker"], st.get("hollow", False), ms)


def marks(x, n, shift=0.0, log=False):
    """Indices of about n points of the increasing array x, evenly spaced in x (or log x). The shift,
    a fraction of the spacing, staggers the markers of lines that run close together."""
    v = np.log(np.asarray(x, dtype=float)) if log else np.asarray(x, dtype=float)
    grid = v[0] + (np.arange(n) + 0.5 + shift) * (v[-1] - v[0]) / n
    return sorted(set(np.abs(v[:, None] - grid[None, :]).argmin(axis=0).tolist()))


def plot_rate(ax, df, attack, methods, title):
    sub = df[df.attack == attack]
    for m in methods:
        s = sub[sub.method == m].sort_values("B")
        st = STYLE[m]
        # undeflated betting is drawn above the per-step test: the two nearly coincide under flips, and a
        # circle of the same size sits inside the square, so both markers stay visible
        ax.errorbar(s.B, s.rate, yerr=1.96 * s.se, color=st["color"], ls=st["ls"], capsize=0,
                    elinewidth=0.6, label=st["label"], zorder=2.5 if m == "naive" else 2, **smk(st))
    ax.axhline(ALPHA, color="#0b0b0b", lw=0.7, ls=":")
    ax.text(ax.get_xlim()[1] if False else 101, ALPHA, r" $\alpha$", va="center", fontsize=6.5)
    ax.set_title(title)
    ax.set_xlabel(r"attacker's budget $B$")
    ax.set_yscale("symlog", linthresh=0.05, linscale=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
    ax.set_yticklabels(["0", ".025", ".05", ".1", ".25", ".5", "1"])
    ax.grid(True, axis="y")


def fig_validity():
    df = pd.read_csv(os.path.join(RES, "e1_validity.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.75), sharey=True)
    methods = ["naive", "perstep", "ins", "avg_ins", "rep", "avg_rep"]   # all six methods in every panel
    plot_rate(axes[0], df, "burst_trigger", methods, "(a) Burst of forged wins")
    plot_rate(axes[1], df, "online_replace", methods, "(b) Losses flipped")
    plot_rate(axes[2], df, "spread", methods, "(c) Forged wins at random times")
    axes[0].set_ylabel(r"$\mathbb{P}$(false statement)")
    h, l = axes[1].get_legend_handles_labels()
    h = [x[0] for x in h]   # the data line of each errorbar container: no error-bar tick in the key
    l = [x + ", one bet" if x == STYLE["naive"]["label"] else x for x in l]   # E1 uses the single bet 0.2
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_validity.pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def fig_rates_arena():
    rate = pd.read_csv(os.path.join(RES, "e2_rate.csv"))
    tau = pd.read_csv(os.path.join(RES, "e2_tau.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.95))
    # (a) corruption tolerance per record, delta = 0.1
    ax = axes[0]
    delta = 0.10
    # E2 certifies with the average over the grid, so the markers are those of the grid certificates
    for kind, st, opt, lab, shift in (("insertion", STYLE["avg_ins"], 2 * delta, r"$2\delta$", 0.0),
                                      ("replacement", STYLE["avg_rep"], delta, r"$\delta$", 0.25)):
        col = st["color"]
        s = rate[(np.isclose(rate.delta, delta)) & (rate.kind == kind) & (rate.t >= 300)].sort_values("t")
        ax.plot(s.t, s.q50.clip(lower=0), color=col, label=kind, markevery=marks(s.t, 6, shift, log=True), **smk(st))
        ax.fill_between(s.t, s.q10.clip(lower=0), s.q90.clip(lower=0), color=col, alpha=0.12, lw=0)
        ax.axhline(opt, color=col, lw=0.7, ls=":")
        # the 2delta label at the left; the delta label at the right, clear of the insertion curve
        x, ha = (330, "left") if kind == "insertion" else (s.t.max(), "right")
        ax.text(x, opt + 0.004, "optimal rate " + lab, color=col, fontsize=5.8, va="bottom", ha=ha)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.235)
    ax.set_xlabel(r"records $t$")
    ax.set_ylabel(r"$\widehat B_t/t$")
    ax.set_title(r"(a) Corruption tolerance per record, $\delta=0.1$")
    ax.legend(loc="lower right", frameon=False, borderaxespad=0.1, handlelength=1.8, labelspacing=0.25)
    ax.grid(True, axis="y")
    # (b) time to certify
    ax = axes[1]
    for kind, st in (("insertion", STYLE["avg_ins"]), ("replacement", STYLE["avg_rep"])):
        col = st["color"]
        s = tau[(np.isclose(tau.delta, 0.05)) & (tau.kind == kind)].sort_values("b")
        ax.plot(s.b, s.mean_tau, color=col, label=kind, **smk(st))
        ax.plot(s.b, s.upper, color=col, ls="--", lw=0.9)
        # hollow markers where the two lower bounds separate, so that each is identified in grey print
        ax.plot(s.b, s.lower, color=col, ls=":", lw=0.9, markevery=list(np.flatnonzero(s.b.to_numpy() >= 100)),
                **mk(st["marker"], hollow=True, ms=0.9 * MS))
    ax.plot([], [], color="#52514e", ls="--", lw=0.9, label="upper bound")
    ax.plot([], [], color="#52514e", ls=":", lw=0.9, label="lower bound")
    ax.set_xlabel(r"tolerance $b$")
    ax.set_ylabel(r"$\mathbb{E}\,\tau_b$ (records)")
    ax.set_title(r"(b) Records to certify, $\delta=0.05$")
    ax.set_ylim(0, 16500)   # room for the key above the replacement curves (11,239 at b = 400)
    ax.legend(loc="upper left", frameon=False, handlelength=1.8, labelspacing=0.25)
    ax.grid(True, axis="y")
    # (c) Arena paths
    ax = axes[2]
    paths = np.load(os.path.join(RES, "raw", "e4a_paths.npz"))
    # one marker shape per claim; filled on the forged-vote (solid) line, hollow on the flipped-vote (dashed) one
    show = [("llama-3-70b-instruct__llama-3-8b-instruct", "Llama-3-70B over Llama-3-8B", "#2a78d6", "o"),
            ("gpt-4o-2024-05-13__llama-3-70b-instruct", "GPT-4o over Llama-3-70B", "#1baf7a", "s"),
            ("claude-3-opus-20240229__claude-3-sonnet-20240229", "Claude-3-Opus over Claude-3-Sonnet", "#4a3aa7", "^"),
            ("claude-3-sonnet-20240229__claude-3-haiku-20240307", "Claude-3-Sonnet over Claude-3-Haiku", "#eb6834", "D")]
    for c, (key, lab, col, marker) in enumerate(show):
        bi, br, ts = paths[key]
        dates = pd.to_datetime(ts, unit="s")
        every = marks(ts, 5, shift=0.2 * c - 0.3)
        ax.plot(dates, np.maximum(bi, 0), color=col, lw=1.2, label=lab, markevery=every, **mk(marker))
        ax.plot(dates, np.maximum(br, 0), color=col, lw=0.8, ls="--", markevery=every,
                **mk(marker, hollow=True, ms=0.9 * MS))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(4, 6, 8)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylabel(r"tolerance $\widehat B_t$ (votes)")
    ax.set_title("(c) Chatbot Arena, real votes")
    ax.plot([], [], color="#52514e", lw=1.2, label="forged votes (insertion)")
    ax.plot([], [], color="#52514e", lw=0.8, ls="--", label="flipped votes (replacement)")
    ax.legend(loc="upper left", frameon=False, fontsize=5.4, handlelength=2.2, labelspacing=0.25)
    ax.set_ylim(0, 4700)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rates_arena.pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def fig_matrix():
    # family of all ordered pairs of all models (e4_arena.py call); the displayed ten are a subset
    z = np.load(os.path.join(RES, "e4c_matrix_all.npz"))
    models = [model_names.arena(m) for m in z["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.2))
    for ax, key, title in ((axes[0], "B_closed_ins", "forged votes (insertion)"),
                           (axes[1], "B_closed_rep", "flipped votes (replacement)")):
        B = z[key].copy()
        n = z["n"]
        k = len(models)
        M = np.where(np.isnan(B), -1, B)
        im = ax.imshow(np.where(M >= 0, M, np.nan), cmap="Blues", vmin=0, vmax=np.nanmax(np.where(M >= 0, M, np.nan)))
        for a in range(k):
            for b in range(k):
                if a == b:   # same model: grey cell, as in fig_frontier
                    ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, color="#e4e3df"))
                elif M[a, b] >= 0:
                    ax.text(b, a, f"{int(M[a, b])}", ha="center", va="center", fontsize=6.5,
                            color="white" if M[a, b] > 0.6 * np.nanmax(M) else "#0b0b0b")
                elif n[a, b] == 0:   # no battles in the window: a visible dot
                    ax.plot(b, a, marker="o", ms=2.2, color="#52514e", mec="none")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(models, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticklabels(models, fontsize=6.5)
        ax.set_title(f"Corruption tolerance, {title}")
        ax.set_xlabel("loses to row model")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_arena_matrix.pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def table_rigging():
    df = pd.read_csv(os.path.join(RES, "e4b_rigging.csv"))
    df = df[df.B.isin([0, 50, 100, 200, 400, 800])]
    rows = []
    for (x, y), g in df.groupby(["attacked", "opponent"], sort=False):
        for attack in ("burst_random", "spread", "online_replace"):
            r = {"pair": f"{x} vs {y}", "attack": attack}
            for m in ("wald", "naive", "perstep", "avg_ins", "avg_rep"):
                s = g[(g.attack == attack) & (g.method == m)].set_index("B").rate
                for B in (0, 100, 400):
                    r[f"{m}_{B}"] = s.get(B, np.nan)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "e4b_table.csv"), index=False)
    # mean over the three pairs, for the main-text table
    mean = out.groupby("attack", sort=False).mean(numeric_only=True)
    mean.to_csv(os.path.join(RES, "e4b_table_mean.csv"))
    print(mean.round(3).T.to_string())


def fig_regime():
    """E5: certificate and empirical win rates under a regime shift (median, 10-90% band)."""
    z = np.load(os.path.join(RES, "raw", "e5_regime.npz"))
    t = np.arange(1, int(z["N"]) + 1)
    T1 = int(z["T1"])
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3))
    ax = axes[0]
    for key, st, shift in (("q_ins", STYLE["avg_ins"], -0.2), ("q_rep", STYLE["avg_rep"], 0.2),
                           ("q_exp_ins", STYLE["exp_ins"], 0.3), ("q_exp_rep", STYLE["exp_rep"], -0.3)):
        q = z[key]
        ax.plot(t, q[1], color=st["color"], ls=st["ls"], label=st["label"], markevery=marks(t, 10, shift), **smk(st))
        if not key.startswith("q_exp"):     # the band of the average tolerance would coincide with it
            ax.fill_between(t, q[0], q[2], color=st["color"], alpha=0.18, lw=0)
    ax.axvline(T1, color="#52514e", ls=":", lw=0.8)
    ax.axhline(0, color="#52514e", lw=0.5)
    ax.set_xlabel("record $t$"); ax.set_ylabel(r"tolerance $\widehat B_t$")
    t1 = f"{T1:,}".replace(",", "{,}")   # thousands separator inside math, as in the text ($1{,}000$)
    ax.set_title(f"(a) win probability {float(z['P1'])} for $t \\leq {t1}$, then {float(z['P2'])}")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(t, z["q_cum"][1], color="#52514e", label="cumulative win rate", markevery=marks(t, 10, -0.2), **mk("o"))
    ax.fill_between(t, z["q_cum"][0], z["q_cum"][2], color="#52514e", alpha=0.15, lw=0)
    ax.plot(t, z["q_trail"][1], color="#eb6834", ls="--", label=f"trailing {int(z['trail'])}-record win rate",
            markevery=marks(t, 10, 0.2), **mk("s"))
    ax.fill_between(t, z["q_trail"][0], z["q_trail"][2], color="#eb6834", alpha=0.15, lw=0)
    ax.axhline(0.5, color="#52514e", lw=0.5); ax.axvline(T1, color="#52514e", ls=":", lw=0.8)
    ax.set_xlabel("record $t$"); ax.set_ylabel("empirical win rate"); ax.set_title("(b) win rates")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_regime.pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def fig_benchmark_replay():
    """E7 and E8: probability of a false statement on resampled benchmark items. (a, b) HELM MMLU,
    four 7-9B models, mean over the six pairs (as Table 1). (c) Contaminated items on the five frontier
    benchmarks of E8: per benchmark the mean over its pairs; line = average over the benchmarks,
    band = range over the benchmarks."""
    e7 = pd.read_csv(os.path.join(RES, "e7b_contamination.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.0), sharey=True)
    methods = ("wald", "naive", "perstep", "avg_ins", "avg_rep")
    for ax, (attack, title) in zip(axes[:2], (("burst_random", "(a) MMLU, forged"),
                                             ("online_replace", "(b) MMLU, flipped"))):
        sub = e7[e7.attack == attack].groupby(["method", "B"]).rate.mean().reset_index()
        for m in methods:
            s = sub[sub.method == m].sort_values("B")
            st = STYLE[m]
            ax.plot(s.B, s.rate, color=st["color"], ls=st["ls"], label=st["label"], **smk(st))
        ax.set_title(title)
    ax = axes[2]
    per = {m: [] for m in methods}
    for scen, _, _, _ in FRONTIER_BENCH:
        d = pd.read_csv(os.path.join(RES, f"e8b_{scen}_contamination.csv"))
        for m in methods:
            per[m].append(d[(d.attack == "online_replace") & (d.method == m)].groupby("B").rate.mean())
    for m in methods:
        c = pd.concat(per[m], axis=1)
        st = STYLE[m]
        ax.fill_between(c.index, c.min(axis=1), c.max(axis=1), color=st["color"], alpha=0.15, lw=0)
        ax.plot(c.index, c.mean(axis=1), color=st["color"], ls=st["ls"], **smk(st))
    ax.set_title("(c) Five frontier benchmarks, flipped")
    ax.title.set_fontsize(7.2)
    for ax in axes:
        ax.axhline(ALPHA, ls=":", color="#52514e", lw=0.8)
        ax.set_xscale("symlog", linthresh=10, linscale=0.5)
        ax.set_xlim(-1, None)
        ax.set_xlabel("corrupted items $B$")
        ax.grid(True, axis="y")
    axes[0].set_ylabel(r"$\mathbb{P}$(false statement)")
    h, l = axes[0].get_legend_handles_labels()
    l = [x + ", grid" if x == STYLE["naive"]["label"] else x for x in l]   # the replays average over the grid
    axes[0].legend(h, l, frameon=False, loc="upper left", fontsize=5.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_benchmark_replay.pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


FRONTIER_BENCH = (("mmlu_pro", "MMLU-Pro (1,000)", "e8c_matrices.npz", "#2a78d6"),
                  ("gpqa", "GPQA (446)", "e8c_matrices.npz", "#4a3aa7"),
                  ("ifeval", "IFEval (541)", "e8c_matrices.npz", "#1baf7a"),
                  ("omni_math", "Omni-MATH (1,000)", "e8c_matrices.npz", "#c98500"),
                  ("swebench", "SWE-bench Verified (500)", "e8c_swebench_matrices.npz", "#d55181"))
FRONTIER_NAMES = {"claude-opus-4": "Claude Opus 4", "claude-sonnet-4": "Claude Sonnet 4", "o3": "o3", "gpt-4.1": "GPT-4.1",
                  "gemini-2.5-pro": "Gemini 2.5 Pro", "grok-4": "Grok 4", "deepseek-r1": "DeepSeek-R1", "qwen3-235b": "Qwen3-235B",
                  "gpt-5.1": "GPT-5.1", "gpt-5": "GPT-5", "gemini-3-pro": "Gemini 3 Pro", "claude-sonnet-4.5": "Claude Sonnet 4.5",
                  "kimi-k2": "Kimi K2", "gpt-oss-120b": "gpt-oss-120b", "qwen3-235b-2507": "Qwen3-235B-2507",
                  # SWE-bench run names -> the names used in the text
                  "Claude 4.5 Opus (high)": "Claude Opus 4.5", "Claude 4.6 Opus": "Claude Opus 4.6",
                  "Claude 4.5 Sonnet (high)": "Claude Sonnet 4.5", "Claude 4.5 Haiku (high)": "Claude Haiku 4.5",
                  "GPT 5.2 (high)": "GPT-5.2", "GPT 5 mini": "GPT-5 mini", "GLM 5 (high)": "GLM-5"}


def fig_frontier():
    """E8: the six highest-scoring models of each frontier benchmark. Cell (row, column): simultaneous
    certificate (contaminated items; closed testing over all ordered pairs of the benchmark's models)
    of "row has ever had an edge over column", or empty if not certified. Last column: the certified rank bound, one plus
    the number of models certified above the row model (among all models of the benchmark), the
    analogue of Arena's approximate ranking."""
    K = 6
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.3))
    for ax, (scen, title, mat, _) in zip(axes.flat, FRONTIER_BENCH):
        z = np.load(os.path.join(RES, mat), allow_pickle=True)
        names = list(z["models"])
        tab = pd.read_csv(os.path.join(RES, "raw", f"e8_{scen}_items.csv.gz"))
        acc = tab[names].mean().sort_values(ascending=False, kind="stable")
        top = list(acc.index[:K])
        ix = {m: k for k, m in enumerate(names)}
        ci, cr = z[f"{scen}_B_closed_ins"], z[f"{scen}_B_closed_rep"]
        M = np.full((K, K + 1), np.nan)
        for a, i in enumerate(top):
            for b, j in enumerate(top):
                if a != b and ci[ix[i], ix[j]] >= 0:
                    M[a, b] = cr[ix[i], ix[j]]
        ax.imshow(M, cmap="Greens", vmin=-4, vmax=30, aspect="equal")
        for a in range(K):
            for b in range(K):
                if a == b:
                    ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, color="#e4e3df"))
                elif not np.isnan(M[a, b]):   # an empty cell means not certified, as in fig_matrix
                    ax.text(b, a, f"{int(M[a, b])}", ha="center", va="center", fontsize=6.8,
                            color="white" if M[a, b] > 18 else "#0b0b0b")
            rank_bound = 1 + int(sum(ci[ix[m], ix[top[a]]] >= 0 for m in names if m != top[a]))
            ax.text(K, a, f"{rank_bound}", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#0b0b0b")
        ax.axvline(K - 0.5, color="#52514e", lw=0.6)
        lab = lambda m: FRONTIER_NAMES.get(m, m).replace(" (high)", "")
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"{lab(m)}  {100 * acc[m]:.1f}" for m in top], fontsize=6.3)
        ax.set_xticks(range(K + 1))
        ax.set_xticklabels([lab(m) for m in top] + ["rank bound"], rotation=50, ha="right", fontsize=6.0)
        ax.set_title(title, fontsize=7.9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0, pad=1.5)
    key = axes.flat[-1]
    key.axis("off")
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.6)
    # the key sits in the sixth slot, starting where the row labels of that column start, so that it
    # stays inside the figure (placed after the layout, which it must not influence)
    fig.canvas.draw()
    x0 = (axes.flat[2].get_tightbbox(fig.canvas.get_renderer()).x0 + 0.08 * fig.dpi) / fig.bbox.width
    kp = key.get_position()
    fig.text(x0, kp.y0 + 0.62 * kp.height, "Row: model and score (%).\n"
             "Cell: tolerance in contaminated items\nfor \"row has ever had an edge over\ncolumn\", simultaneous over all pairs.\n"
             "Empty cell: not certified.\n"
             "Rank bound: 1 + number of\nmodels certified above the row.",
             fontsize=6.8, va="center", ha="left", color="#0b0b0b", linespacing=1.35)
    fig.savefig(os.path.join(FIG, "fig_frontier.pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    fig_validity()
    fig_rates_arena()
    fig_matrix()
    table_rigging()
    fig_regime()
    fig_benchmark_replay()
    fig_frontier()
