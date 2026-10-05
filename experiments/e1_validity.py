"""E1: validity of certified budgets under four attackers (null p = 1/2).

For each attacker and budget B we estimate the probability that a method makes a
false statement at some time t <= horizon:
  * naive      : undeflated constant-bet test, false iff K_t >= 1/alpha
  * perstep    : per-step robust test (null mean 1/2 + eps, eps = B/T), false iff K_t >= 1/alpha
  * ins, rep   : constant-bet certified budget with cost M_+ / (M_+/m_-); false iff Bhat_t >= N_t
  * avg_ins, avg_rep : grid certificate (deflate each bet by its cost, then average), same criterion
The optimal offline (look-ahead) replacement attacker coincides with online_replace for
constant bets (Sharpness theorem, part (d)), so it is not run separately.
Raw per-repetition values of max_t S_t are saved to results/raw/e1_*.npz.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import ccb  # noqa: E402

ALPHA = ccb.ALPHA
LAM = 0.2
T = 5000
REPS = 10000
CHUNK = 2000
BS = [0, 5, 10, 20, 30, 40, 60, 100]
ATTACKS = ["burst_start", "burst_trigger", "spread", "online_replace"]
METHODS = ["naive", "perstep", "ins", "rep", "avg_ins", "avg_rep"]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")


def evaluate(attack, B, rng):
    H = (rng.random((CHUNK, T)) < 0.5).astype(float)
    eps = B / T
    li, lr = ccb.costs(LAM)
    res = {}
    if attack == "burst_start":
        W, F = ccb.attack_burst_start(H, B)
    elif attack == "burst_trigger":
        W, F = ccb.attack_burst_trigger(H, B, LAM, np.log(1 / ALPHA) - B * li)
    elif attack == "spread":
        W, F = ccb.attack_spread(H, B, rng)
    elif attack == "online_replace":
        W, F = ccb.attack_online_replace(H, B)
    res["naive"] = ccb.single_violation(W, F, LAM, "none")
    res["perstep"] = ccb.single_violation(W, F, LAM, "none", eps=eps)
    res["ins"] = ccb.single_violation(W, F, LAM, "insertion")
    res["rep"] = ccb.single_violation(W, F, LAM, "replacement")
    res["avg_ins"] = ccb.avg_violation(W, F, "insertion")
    res["avg_rep"] = ccb.avg_violation(W, F, "replacement")
    return res


def run_config(args):
    ai, attack, bi, B = args
    t0 = time.time()
    rng = np.random.default_rng(10_000 + 100 * ai + bi)
    acc = {m: [] for m in METHODS}
    for _ in range(REPS // CHUNK):
        r = evaluate(attack, B, rng)
        for m in METHODS:
            acc[m].append(r[m])
    acc = {m: np.concatenate(v) for m, v in acc.items()}
    np.savez_compressed(os.path.join(RAW, f"e1_{attack}_B{B}.npz"), **acc)
    rows = []
    for m in METHODS:
        p = float(np.mean(acc[m] >= -1e-12))
        rows.append(dict(attack=attack, B=B, method=m, rate=p,
                         se=np.sqrt(p * (1 - p) / REPS), reps=REPS))
    print(attack, B, {m: round(float(np.mean(acc[m] >= -1e-12)), 4) for m in METHODS},
          f"{time.time() - t0:.1f}s", flush=True)
    return rows


def main():
    from concurrent.futures import ProcessPoolExecutor
    configs = [(ai, a, bi, B) for ai, a in enumerate(ATTACKS) for bi, B in enumerate(BS)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = [r for rs in ex.map(run_config, configs) for r in rs]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e1_validity.csv"), index=False)


if __name__ == "__main__":
    main()
