"""E4: Chatbot Arena replay.

Data: public Chatbot Arena battles, clean_battle_20240814_public.json (1,799,991 anonymous
battles, 2023-04-24 to 2024-08-14), extracted by src/arena_extract.py to a gzipped CSV.
Outcome for an ordered pair (i, j): W = 1 if i wins, 0 if j wins, 1/2 for either tie
label. A tie contributes the factor 1 to every betting wealth, so it is equivalent to
dropping it; the null P(i wins) <= P(j wins) is unchanged.

E4a  certified budgets on the real, time-ordered votes for the most-played pairs.
E4b  rigging replay: for near-boundary pairs, null streams are drawn by resampling the
     pair's own votes (the attacked model's empirical win rate is <= 1/2, so the claim
     'attacked model is better' is false). The attacker adds or flips B votes.
E4c  pairwise certified-budget matrix (last 120 days), simultaneous over pairs via closed
     testing (a Bonferroni split is computed for comparison). Mode `call` uses the family
     of all ordered pairs of all models, which is fixed in advance; this is the version in
     the paper (e4c_matrix_all.npz). Mode `c` uses only the 90 pairs among the ten
     displayed models, a family chosen from the data, and is kept for comparison
     (e4c_matrix.npz).
stats  counts quoted in the paper (votes, models, dates, share of ties), written to
     results/e4_data_log.txt.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import ccb  # noqa: E402

ALPHA = ccb.ALPHA
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")
# Arena extract: data/arena/arena_battles.csv.gz, or the file named by the ARENA_DATA variable
DATA = os.environ.get("ARENA_DATA", os.path.join(ROOT, "data", "arena", "arena_battles.csv.gz"))


def load():
    d = pd.read_csv(DATA).sort_values("tstamp", kind="stable").reset_index(drop=True)
    d["tie"] = d.winner.str.startswith("tie")
    return d


def data_stats(d):
    """Counts quoted in the paper, saved so that the text can be checked against them."""
    models = pd.unique(pd.concat([d.model_a, d.model_b]))
    day = lambda x: pd.to_datetime(x, unit="s").strftime("%Y-%m-%d")
    lines = [f"votes: {len(d):,}",
             f"models: {len(models)}",
             f"first vote: {day(d.tstamp.min())}, last vote: {day(d.tstamp.max())}",
             f"anonymous battles: {int(d.anony.sum()):,}",
             f"ties (tie or tie (bothbad)): {int(d.tie.sum()):,} = {d.tie.mean():.4f} of all votes",
             f"  of which tie (bothbad): {int((d.winner == 'tie (bothbad)').sum()):,}"]
    open(os.path.join(OUT, "e4_data_log.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def pair_stream(d, i, j):
    """Time-ordered outcomes of i against j (1 = i wins, 0 = j wins, 1/2 = tie)."""
    m = ((d.model_a == i) & (d.model_b == j)) | ((d.model_a == j) & (d.model_b == i))
    s = d[m]
    iwin = ((s.winner == "model_a") & (s.model_a == i)) | ((s.winner == "model_b") & (s.model_b == i))
    w = np.where(s.tie, 0.5, np.where(iwin, 1.0, 0.0))
    return w, s.tstamp.to_numpy()


# ------------------------------------------------------------------ E4a
def e4a(d, top=20):
    a = np.minimum(d.model_a, d.model_b)
    b = np.maximum(d.model_a, d.model_b)
    counts = pd.Series(list(zip(a, b))).value_counts().head(top)
    rows, paths = [], {}
    for (x, y), n in counts.items():
        w, ts = pair_stream(d, x, y)
        if w.mean() < 0.5:  # orient so that the leader comes first
            x, y, w = y, x, 1 - w
        bi = ccb.avg_certified_budget(w[None, :], "insertion")[0]
        br = ccb.avg_certified_budget(w[None, :], "replacement")[0]
        li = ccb.avg_certified_budget(w[None, :], "insertion", family="exp")[0]      # lead certificate (Theorem 5.7)
        lr = ccb.avg_certified_budget(w[None, :], "replacement", family="exp")[0]
        naive_flip = len(w) * (2 * w.mean() - 1)  # forged votes that move the point estimate to 1/2 (= wins - losses)
        rows.append(dict(leader=x, other=y, n=len(w), win_rate=w.mean(),
                         tie_rate=float(np.mean(w == 0.5)),
                         B_ins=max(bi[-1], -1), B_rep=max(br[-1], -1),
                         B_lead_ins=max(li[-1], -1), B_lead_rep=max(lr[-1], -1),
                         point_estimate_flip=naive_flip,
                         first_certified_t=int(np.argmax(bi >= 0)) + 1 if (bi >= 0).any() else -1))
        paths[f"{x}__{y}"] = np.stack([bi, br, ts])
        paths[f"{x}__{y}__lead"] = np.stack([li, lr])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e4a_pairs.csv"), index=False)
    np.savez_compressed(os.path.join(RAW, "e4a_paths.npz"), **paths)
    print(df.to_string(), flush=True)


# ------------------------------------------------------------------ E4b
def wald_violation(W, F, every=50):
    """'Practice' baseline: 95% Wald interval for the mean outcome, recomputed every
    `every` records; a false claim is made if the lower end ever exceeds 1/2."""
    valid = ~np.isnan(W)
    x = np.where(valid, W, 0.0)
    n = np.cumsum(valid, axis=1)
    s = np.cumsum(x, axis=1)
    s2 = np.cumsum(x * x, axis=1)
    idx = np.arange(every - 1, W.shape[1], every)
    n, s, s2 = n[:, idx], s[:, idx], s2[:, idx]
    mean = s / n
    var = np.maximum(s2 / n - mean ** 2, 1e-12)
    lo = mean - 1.96 * np.sqrt(var / n)
    return (lo - 0.5).max(axis=1)


def attack_burst_random(H, B, rng):
    R, T = H.shape
    rows, flags = [], []
    for r in range(R):
        pos = rng.integers(0, T + 1)
        rows.append(np.concatenate([H[r, :pos], np.ones(B), H[r, pos:]]))
        flags.append(np.concatenate([np.zeros(pos), np.ones(B), np.zeros(T - pos)]))
    return ccb._pad(rows, flags, T + B)


def e4b(d, reps=1000):
    pairs = [("claude-3-opus-20240229", "gpt-4-turbo-2024-04-09"),
             ("claude-3-opus-20240229", "gemini-1.5-pro-api-0409-preview"),
             ("claude-3-sonnet-20240229", "llama-3-70b-instruct")]
    BS = [0, 25, 50, 100, 200, 400, 800]
    rows = []
    for pi, (x, y) in enumerate(pairs):
        w, _ = pair_stream(d, x, y)  # x = attacked model, its empirical win rate < 1/2
        assert w.mean() < 0.5, (x, y, w.mean())
        n = len(w)
        for bi, B in enumerate(BS):
            rng = np.random.default_rng(40_000 + 100 * pi + bi)
            res = {}
            for attack in ("burst_random", "spread", "online_replace"):
                H = rng.choice(w, size=(reps, n), replace=True)
                if attack == "burst_random":
                    W, F = attack_burst_random(H, B, rng)
                elif attack == "spread":
                    W, F = ccb.attack_spread(H, B, rng)
                else:
                    W, F = ccb.attack_online_replace(H, B)
                v = dict(wald=wald_violation(W, F),
                         naive=ccb.avg_violation(W, np.zeros_like(F)),  # same grid, no deflation
                         perstep=ccb.single_violation(W, F, 0.2, "none", eps=B / n),
                         avg_ins=ccb.avg_violation(W, F, "insertion"),
                         avg_rep=ccb.avg_violation(W, F, "replacement"))
                res.update({f"{attack}__{k}": val for k, val in v.items()})
                for k, val in v.items():
                    p = float(np.mean(val >= -1e-12))
                    rows.append(dict(attacked=x, opponent=y, n=n, win_rate=w.mean(), attack=attack,
                                     B=B, method=k, rate=p, se=np.sqrt(p * (1 - p) / reps)))
            np.savez_compressed(os.path.join(RAW, f"e4b_pair{pi}_B{B}.npz"), **res)
            print(x, y, B, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e4b_rigging.csv"), index=False)
    print(df.pivot_table(index=["attacked", "attack", "B"], columns="method", values="rate").to_string())


# ------------------------------------------------------------------ E4c
def final_log_parts(w, kind):
    """(log w_k, log K^(k)_T, c_k) at the final time for one pair's outcome stream."""
    lams, wts = ccb.default_grid()
    out = []
    for lam, wk in zip(lams, wts):
        li, lr = ccb.costs(lam)
        L = float(ccb.log_wealth(w, lam)[-1]) if len(w) else 0.0
        out.append((np.log(wk), L, li if kind == "insertion" else lr))
    return out


def log_E(parts, b):
    """log of the deflated e-value sum_k w_k K^(k) exp(-b c_k)."""
    return np.logaddexp.reduce([lw + L - b * c for lw, L, c in parts])


def closed_budget(all_parts, e, alpha=ALPHA, bmax=100000):
    """Closed testing over all sets of true nulls: the claim for pair e is certified at
    budget b iff every set S containing e has average deflated e-value >= 1/alpha.
    The smallest average over S containing e adds the other pairs in increasing order."""
    def ok(b):
        le = np.array([log_E(p, b) for p in all_parts])
        E = np.exp(le - le.max())  # common scaling
        scale = le.max()
        others = np.sort(np.delete(E, e))
        csum = np.concatenate([[0.0], np.cumsum(others)])
        avg = (E[e] + csum) / np.arange(1, len(others) + 2)
        return np.log(avg.min()) + scale >= np.log(1 / alpha)
    if not ok(0):
        return -1
    lo, hi = 0, 1
    while ok(hi) and hi < bmax:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def bonferroni_budget(parts, m, alpha=ALPHA):
    b = -1
    while log_E(parts, b + 1) >= np.log(m / alpha):
        b += 1
    return b


def e4c(d, days=120, k=10):
    end = d.tstamp.max()
    s = d[d.tstamp >= end - days * 86400]
    games = pd.concat([s.model_a, s.model_b]).value_counts()
    net = {}
    for m in games.index[:25]:
        t = s[(s.model_a == m) | (s.model_b == m)]
        won = ((t.winner == "model_a") & (t.model_a == m)) | ((t.winner == "model_b") & (t.model_b == m))
        lost = ((t.winner == "model_a") & (t.model_b == m)) | ((t.winner == "model_b") & (t.model_a == m))
        net[m] = (won.sum() - lost.sum()) / len(t)
    models = sorted(net, key=net.get, reverse=True)[:k]
    pairs = [(a_, b_) for a_ in range(k) for b_ in range(k) if a_ != b_]
    out = {}
    for kind in ("insertion", "replacement"):
        parts, ncount = [], np.zeros((k, k), dtype=int)
        for a_, b_ in pairs:
            w, _ = pair_stream(s, models[a_], models[b_])
            ncount[a_, b_] = len(w)
            parts.append(final_log_parts(w, kind))
        Bc = np.full((k, k), np.nan)
        Bb = np.full((k, k), np.nan)
        Bs = np.full((k, k), np.nan)  # single-pair certificate (no correction over pairs)
        for e, (a_, b_) in enumerate(pairs):
            Bc[a_, b_] = closed_budget(parts, e)
            Bb[a_, b_] = bonferroni_budget(parts[e], len(pairs))
            Bs[a_, b_] = bonferroni_budget(parts[e], 1)
        out[f"B_closed_{kind[:3]}"] = Bc
        out[f"B_bonf_{kind[:3]}"] = Bb
        out[f"B_single_{kind[:3]}"] = Bs
        out["n"] = ncount
        print(kind)
        print(pd.DataFrame(Bc, index=models, columns=models).to_string())
        print(pd.DataFrame(Bb, index=models, columns=models).to_string())
        print(pd.DataFrame(Bs, index=models, columns=models).to_string())
    np.savez_compressed(os.path.join(OUT, "e4c_matrix.npz"), models=np.array(models), days=days, **out)
    print(pd.DataFrame(out["n"], index=models, columns=models).to_string())


# ------------------------------------------------------------------ E4c, all pairs
def simultaneous_budgets_all(LW, LK, C, alpha=ALPHA):
    """Closed, Bonferroni and single-pair budgets for every pair at once.

    LW, LK, C are (P, m) arrays with log w_k, log K^(k)_T and c_k for P pairs and m bets.
    Closed testing over all sets of pairs certifies pair e at budget b iff every set S
    containing e has an average deflated e-value of at least 1/alpha (closed_budget does
    this for one pair). All pairs are handled together as follows. Sort the e-values,
    x_1 <= ... <= x_P, and let C_s be the partial sums and A_s = C_s / s. For the pair of
    rank r, the smallest average over sets S of size s containing it is (x_r + C_{s-1}) / s
    for s <= r and A_s for s > r. The first expression is unimodal in s: it decreases while
    s x_s - C_{s-1} <= x_r, and that quantity is nondecreasing in s, so the minimizer is
    found by searchsorted. The budgets are found by a sweep over b, which is valid since
    every deflated e-value decreases in b. Returns (closed, bonferroni, single), each an
    integer array of length P with -1 for 'not certified'."""
    P = LW.shape[0]
    closed = np.full(P, -1)
    bonf = np.full(P, -1)
    single = np.full(P, -1)
    thr1, thrP = np.log(1 / alpha), np.log(P / alpha)
    b = 0
    while True:
        logE = np.logaddexp.reduce(LW + LK - b * C, axis=1)
        ps = logE >= thr1
        if not ps.any():
            break
        single[ps] = b
        bonf[logE >= thrP] = b
        mx = logE.max()
        E = np.exp(logE - mx)
        order = np.argsort(E, kind="stable")
        x = E[order]
        csum = np.cumsum(x)
        s = np.arange(1, P + 1)
        cm1 = np.concatenate([[0.0], csum[:-1]])          # C_{s-1}
        A = csum / s
        sufmin = np.minimum.accumulate(A[::-1])[::-1]      # min_{s' >= s} A_{s'}
        after = np.concatenate([sufmin[1:], [np.inf]])     # min_{s' > r} A_{s'} at rank r
        h = s * x - cm1                                    # nondecreasing
        cnt = np.searchsorted(h, x, side="right")          # number of s with h(s) <= x_r
        smin = np.minimum(cnt + 1, s)
        g = (x + cm1[smin - 1]) / smin
        ok_sorted = np.log(np.minimum(g, after)) + mx >= thr1
        ok = np.empty(P, dtype=bool)
        ok[order] = ok_sorted
        closed[ok] = b
        b += 1
    return closed, bonf, single


def e4c_all(d, days=120, k=10):
    """E4c with the family of pairs fixed in advance: all ordered pairs of all models in the
    data, so that Prop. 'simultaneous certificates' applies as stated. The k displayed models
    are chosen as in e4c (net win rate in the window); under simultaneous control over all
    pairs the choice of what to display may depend on the data. Writes e4c_matrix_all.npz."""
    end = d.tstamp.max()
    s = d[d.tstamp >= end - days * 86400]
    catalog = sorted(set(d.model_a) | set(d.model_b))
    M = len(catalog)
    idx = {m_: i for i, m_ in enumerate(catalog)}
    lams, wts = ccb.default_grid()
    m = len(lams)
    li = np.array([ccb.costs(l)[0] for l in lams])
    lr = np.array([ccb.costs(l)[1] for l in lams])
    # log wealth at the end of the window for every ordered pair (0 without battles)
    LK = np.zeros((M, M, m))
    ncount = np.zeros((M, M), dtype=int)
    lo = np.minimum(s.model_a, s.model_b)
    hi = np.maximum(s.model_a, s.model_b)
    for (x_, y_), g in s.groupby([lo, hi], sort=False):
        w, _ = pair_stream(g, x_, y_)
        i, j = idx[x_], idx[y_]
        ncount[i, j] = ncount[j, i] = len(w)
        for q, lam in enumerate(lams):
            LK[i, j, q] = ccb.log_wealth(w, lam)[-1]
            LK[j, i, q] = ccb.log_wealth(1 - w, lam)[-1]
    pairs = np.array([(a_, b_) for a_ in range(M) for b_ in range(M) if a_ != b_])
    P = len(pairs)
    LKp = LK[pairs[:, 0], pairs[:, 1]]
    LW = np.tile(np.log(wts), (P, 1))
    print(f"{M} models in the catalog, {P} ordered pairs, "
          f"{int((ncount > 0).sum())} with battles in the last {days} days, {len(s)} battles")
    # displayed models, chosen as in e4c
    games = pd.concat([s.model_a, s.model_b]).value_counts()
    net = {}
    for m_ in games.index[:25]:
        t = s[(s.model_a == m_) | (s.model_b == m_)]
        won = ((t.winner == "model_a") & (t.model_a == m_)) | ((t.winner == "model_b") & (t.model_b == m_))
        lost = ((t.winner == "model_a") & (t.model_b == m_)) | ((t.winner == "model_b") & (t.model_a == m_))
        net[m_] = (won.sum() - lost.sum()) / len(t)
    models = sorted(net, key=net.get, reverse=True)[:k]
    sel = np.array([idx[m_] for m_ in models])
    out = {"catalog": np.array(catalog), "pairs": pairs, "n_all": ncount, "n": ncount[np.ix_(sel, sel)],
           "LK_all": LK}
    for kind, cvec in (("insertion", li), ("replacement", lr)):
        C = np.tile(cvec, (P, 1))
        closed, bonf, single = simultaneous_budgets_all(LW, LKp, C)
        full = {}
        for name, vec in (("closed", closed), ("bonf", bonf), ("single", single)):
            Bfull = np.full((M, M), np.nan)
            Bfull[pairs[:, 0], pairs[:, 1]] = vec
            full[name] = Bfull
            out[f"B_{name}_{kind[:3]}_all"] = Bfull
            out[f"B_{name}_{kind[:3]}"] = Bfull[np.ix_(sel, sel)]
        print(kind)
        for name in ("closed", "bonf", "single"):
            print(name)
            print(pd.DataFrame(out[f"B_{name}_{kind[:3]}"], index=models, columns=models).to_string())
    np.savez_compressed(os.path.join(OUT, "e4c_matrix_all.npz"), models=np.array(models), days=days, P=P, **out)
    print(pd.DataFrame(out["n"], index=models, columns=models).to_string())


def e4c_lead(d, days=120, k=10):
    """E4c for the lead certificate (exponential wealth, Theorem 5.7) made simultaneous by the
    Bonferroni correction of Corollary B.6 over all ordered pairs of all models in the data
    (closed testing does not transfer to the lead claim). Same window, catalog and displayed
    models as e4c_all. Writes e4c_lead_matrix_all.npz and prints the matrices beside the edge
    certificate's closed-testing and Bonferroni matrices from e4c_matrix_all.npz."""
    end = d.tstamp.max()
    s = d[d.tstamp >= end - days * 86400]
    catalog = sorted(set(d.model_a) | set(d.model_b))
    M = len(catalog)
    idx = {m_: i for i, m_ in enumerate(catalog)}
    gams, wts = ccb.default_grid()
    m = len(gams)
    li, lr = ccb.exp_costs(gams)
    LK = np.zeros((M, M, m))
    ncount = np.zeros((M, M), dtype=int)
    lo = np.minimum(s.model_a, s.model_b)
    hi = np.maximum(s.model_a, s.model_b)
    for (x_, y_), g in s.groupby([lo, hi], sort=False):
        w, _ = pair_stream(g, x_, y_)
        i, j = idx[x_], idx[y_]
        ncount[i, j] = ncount[j, i] = len(w)
        for q, gam in enumerate(gams):
            LK[i, j, q] = ccb.exp_log_wealth(w, gam)[-1]
            LK[j, i, q] = ccb.exp_log_wealth(1 - w, gam)[-1]
    pairs = np.array([(a_, b_) for a_ in range(M) for b_ in range(M) if a_ != b_])
    P = len(pairs)
    LKp = LK[pairs[:, 0], pairs[:, 1]]
    LW = np.tile(np.log(wts), (P, 1))
    print(f"lead certificate: {M} models, {P} ordered pairs, {int((ncount > 0).sum())} with battles in the last {days} days")
    edge = np.load(os.path.join(OUT, "e4c_matrix_all.npz"))
    models = list(edge["models"])
    sel = np.array([idx[m_] for m_ in models])
    out = {"catalog": np.array(catalog), "pairs": pairs, "n_all": ncount, "n": ncount[np.ix_(sel, sel)], "LK_all": LK}
    for kind, cvec in (("insertion", li), ("replacement", lr)):
        C = np.tile(cvec, (P, 1))
        _, bonf, single = simultaneous_budgets_all(LW, LKp, C)
        for name, vec in (("bonf", bonf), ("single", single)):
            Bfull = np.full((M, M), np.nan)
            Bfull[pairs[:, 0], pairs[:, 1]] = vec
            out[f"B_{name}_{kind[:3]}_all"] = Bfull
            out[f"B_{name}_{kind[:3]}"] = Bfull[np.ix_(sel, sel)]
        print(kind)
        for name in ("bonf", "single"):
            print("lead", name)
            print(pd.DataFrame(out[f"B_{name}_{kind[:3]}"], index=models, columns=models).to_string())
        for name in ("closed", "bonf"):
            print("edge", name)
            print(pd.DataFrame(edge[f"B_{name}_{kind[:3]}"], index=models, columns=models).to_string())
        eb, ec = edge[f"B_bonf_{kind[:3]}_all"], edge[f"B_closed_{kind[:3]}_all"]
        lb = out[f"B_bonf_{kind[:3]}_all"]
        cert = lambda X: int(np.nansum(X >= 0))
        print(f"certified ordered pairs among {P}: edge closed {cert(ec)}, edge Bonferroni {cert(eb)}, lead Bonferroni {cert(lb)}")
        both = (lb >= 0) & (ec >= 0)
        print(f"  lead Bonferroni / edge closed on pairs certified by both: median {np.nanmedian((lb[both] + 1) / (ec[both] + 1)):.3f}, "
              f"10-90% {np.nanpercentile((lb[both] + 1) / (ec[both] + 1), 10):.3f}-{np.nanpercentile((lb[both] + 1) / (ec[both] + 1), 90):.3f}")
    np.savez_compressed(os.path.join(OUT, "e4c_lead_matrix_all.npz"), models=np.array(models), days=days, P=P, **out)


if __name__ == "__main__":
    d = load()
    which = sys.argv[1:] or ["a", "b", "c"]
    if "lead" in which:
        e4c_lead(d)
    if "stats" in which:
        data_stats(d)
    if "a" in which:
        e4a(d)
    if "c" in which:
        e4c(d)
    if "call" in which:
        e4c_all(d)
    if "b" in which:
        e4b(d)
