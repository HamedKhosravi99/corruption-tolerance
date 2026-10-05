"""E6: attacks on true claims (certificate suppression) on real Chatbot Arena votes.

For the three most strongly certified pairs of E4a (leader i, other j; real, time-ordered
votes), an attacker works AGAINST the leader. Two attacks, B corrupted records each:
  insertion denial   insert B losses (W = 0) for i in one burst;
  replacement denial flip B observed wins of i (W = 1 -> 0), the first B wins after the burst time.
Validity is not at stake (the claim is true, so no statement can be false); the question is
how much certified budget the attacker destroys and how fast the certificate recovers once
honest votes resume. Because the final grid wealth depends on the record only through the
numbers of wins and losses (App. D.1), the final certificate under either attack is the same
wherever the burst is placed; the recovery path is computed for a burst at the midpoint.

Metrics per pair, attack and B:
  B_clean, B_attacked           final certificates (insertion and replacement cost)
  damage, damage_per_record     B_clean - B_attacked and its ratio to B
  decert_budget                 smallest B at which the final certificate is -1 (bisection)
  T_recover0, T_recover50       honest records after the burst until Bhat_t >= 0 and until
                                Bhat_t >= 0.5 * Bhat_clean at the burst time (NaN: not within data)
Outputs: results/e6_suppress.csv, results/e6_decert.csv, results/raw/e6_paths.npz.
Needs the Arena extract (see data/arena/README.md).
"""
import os
import sys

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

PAIRS = [("llama-3-70b-instruct", "llama-3-8b-instruct"),
         ("llama-3-70b-instruct", "snowflake-arctic-instruct"),
         ("gemini-1.5-pro-api-0409-preview", "llama-3-8b-instruct")]
BS = [50, 100, 200, 400, 800]


def attacked_stream(w, B, attack, t0):
    """Return (W, flag) for a burst of B denial records at position t0 (0-based insert point)."""
    if attack == "insert_losses":
        W = np.concatenate([w[:t0], np.zeros(B), w[t0:]])
        f = np.concatenate([np.zeros(t0), np.ones(B), np.zeros(len(w) - t0)])
        return W, f
    # flip the first B wins of the leader at or after t0
    W = w.copy()
    f = np.zeros(len(w))
    wins = np.flatnonzero(W[t0:] == 1.0)[:B] + t0
    W[wins] = 0.0
    f[wins] = 1.0
    return W, f


def final_budget(W, kind):
    return int(ccb.avg_certified_budget(W[None, :], kind)[0, -1])


def decert_budget(w, attack, kind):
    """Smallest B with final certificate -1 (the final value is monotone in B and does not
    depend on where the records are placed, so the burst is put at position 0)."""
    nwin = int((w == 1.0).sum())
    if final_budget(w, kind) < 0:
        return 0
    hi = len(w) if attack == "insert_losses" else nwin   # certainly -1: as many losses as records, or no wins left
    if final_budget(attacked_stream(w, hi, attack, 0)[0], kind) >= 0:
        return np.nan
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if final_budget(attacked_stream(w, mid, attack, 0)[0], kind) >= 0:
            lo = mid
        else:
            hi = mid
    return hi


def run(d):
    rows, dec, paths = [], [], {}
    for x, y in PAIRS:
        w, ts = E4.pair_stream(d, x, y)
        n = len(w)
        assert w.mean() > 0.5, (x, y, w.mean())
        nwin = int((w == 1).sum())
        t0 = n // 2
        clean = {k: ccb.avg_certified_budget(w[None, :], kind)[0] for k, kind in (("ins", "insertion"), ("rep", "replacement"))}
        paths[f"{x}__{y}__clean_ins"] = clean["ins"].astype(np.int32)
        paths[f"{x}__{y}__clean_rep"] = clean["rep"].astype(np.int32)
        for attack in ("insert_losses", "flip_wins"):
            for k, kind in (("ins", "insertion"), ("rep", "replacement")):
                dec.append(dict(leader=x, other=y, n=n, wins=nwin, attack=attack, cost=k,
                                B_clean=int(clean[k][-1]), decert_budget=decert_budget(w, attack, kind)))
            for B in BS:
                if attack == "flip_wins" and B > nwin:
                    continue
                W, f = attacked_stream(w, B, attack, t0)
                for k, kind in (("ins", "insertion"), ("rep", "replacement")):
                    path = ccb.avg_certified_budget(W[None, :], kind)[0]
                    paths[f"{x}__{y}__{attack}_B{B}_{k}"] = path.astype(np.int32)
                    end = len(W) - 1
                    pre = clean[k][t0 - 1]              # certificate just before the burst
                    after = path[t0 + (B if attack == "insert_losses" else 0):]  # honest records after the burst
                    r0 = np.flatnonzero(after >= 0)
                    r50 = np.flatnonzero(after >= 0.5 * pre)
                    rows.append(dict(leader=x, other=y, n=n, attack=attack, B=B, cost=k,
                                     B_clean=int(clean[k][-1]), B_attacked=int(path[end]),
                                     damage=int(clean[k][-1] - path[end]),
                                     damage_per_record=float((clean[k][-1] - path[end]) / B),
                                     B_pre_burst=int(pre), B_right_after_burst=int(path[t0 + (B if attack == "insert_losses" else 0) - 1]) if attack == "insert_losses" else int(path[t0 - 1]),
                                     T_recover0=float(r0[0] + 1) if len(r0) else np.nan,
                                     T_recover50=float(r50[0] + 1) if len(r50) else np.nan))
        print(x, y, n, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e6_suppress.csv"), index=False)
    dd = pd.DataFrame(dec)
    dd.to_csv(os.path.join(OUT, "e6_decert.csv"), index=False)
    np.savez_compressed(os.path.join(RAW, "e6_paths.npz"), **paths)
    pd.set_option("display.width", 250)
    print(dd.to_string(index=False))
    print(df.to_string(index=False))
    return df, dd


if __name__ == "__main__":
    run(E4.load())
