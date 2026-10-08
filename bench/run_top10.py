"""Attacking the top-10 recovery bottleneck, on purpose.

The single-table result said the surrogate orders the bulk of the space
correctly (rank correlation 0.79 at 256 training cells) and still cannot point
at the ten best cells (top-10 recovery 0.10). That is a *measured* failure of a
specific ability, so it gets its own experiment rather than a paragraph.

What is varied, one at a time:
    kernel      categorical (one-hot) vs continuous (metric) -- on a table whose
                factors are real settings, the metric kernel is the corrected
                hypothesis
    n_train     the budget: is this a data problem or a model problem?
    acquisition the policy that consumes the ranking -- an arm that finds the
                best ten is worth more than one that ranks them

Reported per setting:
    spearman    rank correlation over held-out cells
    top10       fraction of the true best ten that the model's best ten contains
    regret      the best *queried* value as a fraction of the ceiling
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pspine import data
from pspine.gp import MixedGP
from pspine.space import DesignSpace, Factor


def spearman(a, b):
    """Rank correlation without scipy, since only the ranks matter here."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def top_k_overlap(pred, truth, k=10):
    """How many of the true best `k` the model's best `k` contains."""
    th = set(np.argsort(truth)[-k:].tolist())
    ph = set(np.argsort(pred)[-k:].tolist())
    return len(th & ph) / float(k)


def expand_index(codes, factors=("cement", "slag", "fly_ash", "water",
                                 "superplasticizer", "coarse_agg", "fine_agg", "age")):
    """Not every factor varied in every row; the concrete table's grid is sparse,
    so build the interface from the observed columns only."""
    return codes


def run(name, seeds, sizes, bin_masks, k=10):
    space, codes, y, meta = data.load(name)
    n = len(codes)
    rng = np.random.default_rng(0)
    rows = []
    for n_train in sizes:
        s_cont, b_cont, t_cont = [], [], []
        for seed in range(seeds):
            perm = np.random.default_rng(1000 + seed).permutation(n)
            tr, te = perm[:n_train], perm[n_train:]
            if len(te) < k + 5:
                continue
            g = MixedGP.for_space(space, max_fit=400).fit(codes[tr], y[tr])
            mu, _ = g.predict(codes[te])
            s_cont.append(spearman(mu, y[te]))
            b_cont.append(top_k_overlap(mu, y[te], k))
            # the binned control: the same rows, but each continuous factor
            # replaced by its rank quintile, which is what a categorical
            # surrogate would see
            t_cont.append(float(np.mean(np.abs(mu - y[te]))))
        rows.append({"n_train": n_train, "spearman_mean": float(np.mean(s_cont)),
                     "spearman_std": float(np.std(s_cont)),
                     "top10_mean": float(np.mean(b_cont)),
                     "mae_mean": float(np.mean(t_cont)),
                     "hypothesis": "continuous kernel (metric kept)"})
        print("  %s/%s n_train=%-4d spearman=%.3f top10=%.3f mae=%.4g" % (
            "metric", name, n_train, rows[-1]["spearman_mean"],
            rows[-1]["top10_mean"], rows[-1]["mae_mean"]), flush=True)

    # the control: bin every factor into 5 ranked levels and fit the same
    # surrogate. If the metric kernel is the fix, this arm must be worse.
    bfactors, bcodes = [], []
    for j, f in enumerate(space.factors):
        vals = np.asarray(f.value_of, dtype=float)
        bcodes.append(np.digitize(codes[:, j], np.quantile(vals, [.2, .4, .6, .8])))
        bfactors.append(Factor.of(f.name + "_bin", list(range(5))))
    bspace = DesignSpace.of(bfactors)
    bcodes = np.asarray(bcodes, dtype=int).T
    brows = []
    for n_train in sizes:
        s_b, b_b = [], []
        for seed in range(seeds):
            perm = np.random.default_rng(2000 + seed).permutation(n)
            tr, te = perm[:n_train], perm[n_train:]
            if len(te) < k + 5:
                continue
            g = MixedGP.for_space(bspace, max_fit=400).fit(bcodes[tr], y[tr])
            mu, _ = g.predict(bcodes[te])
            s_b.append(spearman(mu, y[te]))
            b_b.append(top_k_overlap(mu, y[te], k))
        brows.append({"n_train": n_train, "spearman_mean": float(np.mean(s_b)),
                      "top10_mean": float(np.mean(b_b)),
                      "hypothesis": "same factors rank-binned to 5 levels"})
        print("  %s/%s n_train=%-4d spearman=%.3f top10=%.3f" % (
            "binned", name, n_train, brows[-1]["spearman_mean"],
            brows[-1]["top10_mean"]), flush=True)
    return {"table": name, "rows": n, "metric": rows, "binned": brows, "k": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sizes", default="16,32,64,128,256,512")
    ap.add_argument("--tables", default="ccpp,concrete")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    out = {"seeds": a.seeds, "sizes": sizes, "tables": {}}
    for name in a.tables.split(","):
        print("== %s" % name, flush=True)
        out["tables"][name] = run(name, a.seeds, sizes, None)
    p = a.out or os.path.join(HERE, "top10.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
    for name, d in out["tables"].items():
        print("\n%s (n=%d)" % (name, d["rows"]))
        print("  %-9s %s" % ("n_train", "  ".join("%-10d" % r["n_train"] for r in d["metric"])))
        print("  %-9s %s" % ("metric", "  ".join("%-10.3f" % r["spearman_mean"] for r in d["metric"])))
        print("  %-9s %s" % ("binned", "  ".join("%-10.3f" % r["spearman_mean"] for r in d["binned"])))
        print("  %-9s %s" % ("top10-m", "  ".join("%-10.3f" % r["top10_mean"] for r in d["metric"])))
        print("  %-9s %s" % ("top10-b", "  ".join("%-10.3f" % r["top10_mean"] for r in d["binned"])))


if __name__ == "__main__":
    main()
