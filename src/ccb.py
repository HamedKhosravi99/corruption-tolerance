"""Certified corruption budgets: core computations shared by all experiments.

Conventions
-----------
A record stream for one ordered pair (i, j) is an array W with entries in {0, 1/2, 1}
(1 = i beats j, 1/2 = tie, 0 = j beats i). Padding entries are NaN and contribute a
factor 1. `flag` marks corrupted records (inserted or replaced); N_t = cumsum(flag).

For a constant bet lam in (0, 2) the per-record wealth factor is 1 + lam (W - 1/2),
which lies in [m_-, M_+] = [1 - lam/2, 1 + lam/2].

A certificate with cost c (c = M_+ for insertion, c = M_+/m_- for replacement) reports
    Bhat_t = floor( log(alpha w K_t) / log c ),
where w is the weight of the component (w = 1 for a single bet). It makes a false
statement at time t iff Bhat_t >= N_t, i.e. iff
    S_t := (log K_t + log(alpha w)) / log c - N_t >= 0.
For methods without a budget (c = 1) we use S_t := log K_t + log(alpha) instead.
"""
import numpy as np

ALPHA = 0.05


def log_factors(W, lam, eps=0.0):
    """log(1 + lam (W - 1/2 - eps)); NaN padding -> 0."""
    f = np.log1p(lam * (W - 0.5 - eps))
    return np.where(np.isnan(W), 0.0, f)


def log_wealth(W, lam, eps=0.0):
    return np.cumsum(log_factors(W, lam, eps), axis=-1)


def costs(lam):
    Mp, mm = 1 + lam / 2, 1 - lam / 2
    return np.log(Mp), np.log(Mp / mm)


# ------------------------------------------- exponential wealth (average ordering, Theorem 5.7)
def exp_log_factors(W, gam):
    """gam (W - 1/2) - gam^2 / 8, the log factor of the exponential wealth Z; NaN padding -> 0."""
    f = gam * (W - 0.5) - gam ** 2 / 8
    return np.where(np.isnan(W), 0.0, f)


def exp_log_wealth(W, gam):
    return np.cumsum(exp_log_factors(W, gam), axis=-1)


def exp_costs(gam):
    """Insertion and replacement costs of the exponential wealth: gam/2 - gam^2/8 (the exact log-wealth
    gain of one forged win, so this cost is sharp) and gam (the exact gain of one flipped loss)."""
    return gam / 2 - gam ** 2 / 8, gam


# ---------------------------------------------------------------- bet grid
def default_grid(n=14):
    """Constant bets lam_k = 2^{-k/2}, k = 0..n-1 (1 down to ~0.011), uniform weights."""
    lams = 2.0 ** (-np.arange(n) / 2.0)
    w = np.full(n, 1.0 / n)
    return lams, w


def _grid_logs(W, kind, grid, family="bet"):
    """(log w_k, log wealth of bet k, cost of bet k) for the betting wealth K (family 'bet', the
    default) or the exponential wealth Z (family 'exp'); the same grid serves as caps or as gammas."""
    lams, w = default_grid() if grid is None else grid
    out = []
    for lam, wk in zip(lams, w):
        if family == "exp":
            li, lr = exp_costs(lam)
            L = exp_log_wealth(W, lam)
        else:
            li, lr = costs(lam)
            L = log_wealth(W, lam)
        out.append((np.log(wk), L, li if kind == "insertion" else lr))
    return out


def avg_log_F(parts, b):
    """log F_t(b) = log sum_k w_k K^(k)_t exp(-b c_k); b broadcastable to the wealth arrays."""
    acc = None
    for lw, L, c in parts:
        x = lw + L - b * c
        acc = x if acc is None else np.logaddexp(acc, x)
    return acc


def avg_certified_budget(W, kind="insertion", alpha=ALPHA, grid=None, logk=0.0, family="bet"):
    """Main certificate (deflate each bet by its own cost, then average):
        Bhat_t = max{ b in {0,1,...} : sum_k w_k K^(k)_t exp(-b c_k) >= k_sel / alpha },
    and -1 if no such b. `logk` = log of the selection correction k_sel (0: none).
    `family='exp'` uses the exponential wealth and its costs (the average tolerance).
    Integer bisection between the union-bound value (a lower bound) and
    floor(max_k (log K^(k) + log alpha)/c_k) (an upper bound)."""
    parts = _grid_logs(W, kind, grid, family)
    thr = np.log(1 / alpha) + logk
    lo = None
    hi = None
    for lw, L, c in parts:
        a = np.floor((L + lw - thr) / c)
        h = np.floor((L - thr) / c)
        lo = a if lo is None else np.maximum(lo, a)
        hi = h if hi is None else np.maximum(hi, h)
    lo = np.maximum(lo, -1.0)
    hi = np.maximum(hi, lo)
    while True:
        open_ = hi > lo
        if not open_.any():
            break
        mid = np.floor((lo + hi + 1) / 2)
        ok = avg_log_F(parts, mid) >= thr
        lo = np.where(open_ & ok, mid, lo)
        hi = np.where(open_ & ~ok, mid - 1, hi)
    return lo


def avg_violation(W, flag, kind="insertion", alpha=ALPHA, grid=None, logk=0.0, family="bet"):
    """max_t [log F_t(N_t) - log(k_sel/alpha)]; a false statement is made iff >= 0."""
    parts = _grid_logs(W, kind, grid, family)
    N = np.cumsum(flag, axis=-1)
    return (avg_log_F(parts, N) - np.log(1 / alpha) - logk).max(axis=-1)


def grid_certified_budget(W, kind="insertion", alpha=ALPHA, grid=None):
    """Union variant: Bhat_t = max_k floor(log(alpha w_k K^(k)_t) / log c_k). Returns float array (-inf allowed)."""
    lams, w = default_grid() if grid is None else grid
    out = None
    for lam, wk in zip(lams, w):
        li, lr = costs(lam)
        c = li if kind == "insertion" else lr
        b = np.floor((log_wealth(W, lam) + np.log(alpha * wk)) / c)
        out = b if out is None else np.maximum(out, b)
    return out


def grid_violation(W, flag, kind="insertion", alpha=ALPHA, grid=None):
    """max_t max_k [(log K^(k)_t + log(alpha w_k))/log c_k - N_t]; violation iff >= 0."""
    lams, w = default_grid() if grid is None else grid
    N = np.cumsum(flag, axis=-1)
    best = None
    for lam, wk in zip(lams, w):
        li, lr = costs(lam)
        c = li if kind == "insertion" else lr
        s = ((log_wealth(W, lam) + np.log(alpha * wk)) / c - N).max(axis=-1)
        best = s if best is None else np.maximum(best, s)
    return best


def single_violation(W, flag, lam, kind, alpha=ALPHA, eps=0.0):
    """max_t S_t for a single constant bet. kind in {'none','insertion','replacement'}."""
    L = log_wealth(W, lam, eps)
    if kind == "none":
        return (L + np.log(alpha)).max(axis=-1)
    li, lr = costs(lam)
    c = li if kind == "insertion" else lr
    N = np.cumsum(flag, axis=-1)
    return ((L + np.log(alpha)) / c - N).max(axis=-1)


# ---------------------------------------------------------------- attackers
def _pad(rows, flags, width):
    R = len(rows)
    W = np.full((R, width), np.nan)
    F = np.zeros((R, width))
    for r, (x, f) in enumerate(zip(rows, flags)):
        W[r, : len(x)] = x
        F[r, : len(f)] = f
    return W, F


def attack_burst_start(H, B):
    """Insert B wins before the first honest record."""
    R, T = H.shape
    W = np.concatenate([np.ones((R, B)), H], axis=1)
    F = np.concatenate([np.ones((R, B)), np.zeros((R, T))], axis=1)
    return W, F


def attack_burst_trigger(H, B, lam, trigger):
    """Honest records until the observed log-wealth (bet lam) first reaches `trigger`,
    then insert B wins. If the trigger is never reached no budget is spent."""
    R, T = H.shape
    L = log_wealth(H, lam)
    hit = L >= trigger
    rows, flags = [], []
    for r in range(R):
        idx = np.flatnonzero(hit[r])
        if len(idx) == 0 or B == 0:
            rows.append(H[r]); flags.append(np.zeros(T)); continue
        tau = idx[0] + 1  # insert after record tau
        rows.append(np.concatenate([H[r, :tau], np.ones(B), H[r, tau:]]))
        flags.append(np.concatenate([np.zeros(tau), np.ones(B), np.zeros(T - tau)]))
    return _pad(rows, flags, T + B)


def attack_spread(H, B, rng):
    """Before each honest record, insert a forged win with probability q = B/T, until
    the budget is spent (a per-step, 'Huber-like' attacker)."""
    R, T = H.shape
    q = B / T
    rows, flags = [], []
    for r in range(R):
        ins = rng.random(T) < q
        ins &= np.cumsum(ins) <= B
        k = int(ins.sum())
        pos = np.flatnonzero(ins)
        x = np.insert(H[r], pos, 1.0)
        f = np.insert(np.zeros(T), pos, 1.0)
        rows.append(x); flags.append(f)
    return _pad(rows, flags, T + B)


def attack_online_replace(H, B):
    """Value-dependent, online: overwrite each honest loss (W=0) by a win as soon as it
    is seen, until B records have been replaced. Needs no look-ahead."""
    loss = H == 0
    sel = loss & (np.cumsum(loss, axis=1) <= B)
    W = np.where(sel, 1.0, H)
    return W, sel.astype(float)


def optimal_offline_replace_violation(H, B, lam, c_log, alpha=ALPHA, eps=0.0):
    """Exact optimum of an offline attacker that replaces up to B losses by wins,
    against a constant-bet test with cost log c = c_log (c_log = 0: no budget).

    Replacing a loss at position s <= t raises log K_t by d = log(M_+/m_-) (with the
    per-step shift eps, d = log((1+lam(1/2-eps))/(1-lam(1/2+eps)))). The attacker's best
    value of S_t therefore uses N = min(B, L_t) replacements before t, where L_t is the
    number of honest losses up to t. Returns max_t S_t.
    """
    Lw = log_wealth(H, lam, eps)
    d = np.log1p(lam * (0.5 - eps)) - np.log1p(-lam * (0.5 + eps))
    N = np.minimum(B, np.cumsum(H == 0, axis=1))
    if c_log == 0:
        return (Lw + N * d + np.log(alpha)).max(axis=1)
    return ((Lw + N * d + np.log(alpha)) / c_log - N).max(axis=1)
