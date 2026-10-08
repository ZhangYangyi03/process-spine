"""The cross-domain benchmark: the same arms, four measured tables.

The single-table result (one chemistry table where Bayesian ties one-factor-at-
a-time) is not evidence about the method. This is the check: run the identical
arms with the identical budget on tables whose factor structure, noise, and
coverage are nothing alike, and see whether "OFAT ties the surrogate" survives.

    buchwald    4599 cells of 4608   4 categorical factors, 99.8% covered, a
                                     censored floor at 0
    ccpp        4368 cells of 65536  4 continuous settings, 6.7% covered
    concrete     268 cells of 65536  8 continuous settings, 0.4% covered
    gasturbine  3381 cells of 390625 8 continuous settings, 0.9% covered

An arm that wins on the dense categorical table and loses on the sparse
continuous ones is not a better optimiser -- it is an optimiser for one shape of
table, and that is a finding about the tables, not the algorithm.

    python bench/run_multitable.py --seeds 5 --budget 64 --batch 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pspine import data
from pspine.loop import Campaign

ARMS = ("random", "fill", "ofat", "ei", "qei", "ucb", "pi", "logei")


def one_table(name, seeds, budget, batch, init, arms, acqs_kw=None):
    space, codes, y, meta = data.load(name)
    out = {"meta": meta, "table": name,
           "space": repr(space), "grid": space.size,
           "arms": {}}
    recs = []
    for arm in arms:
        per = []
        for seed in range(seeds):
            c = Campaign(space, codes, y, batch=batch, seed=seed, init=init)
            t0 = time.time()
            trace = c.run(budget=budget, arm=arm)
            dt = time.time() - t0
            best = trace[-1]["best"]
            peak = max(t["best"] for t in trace)
            per.append({"seed": seed, "best": best, "frac": best / c.ceiling,
                        "peak": peak, "seconds": dt,
                        "reach90": any(t["best"] >= 0.90 * c.ceiling for t in trace)})
            print("  %-6s seed %d  spent %3d  frac %.4f  %.1fs" % (
                arm, seed, trace[-1]["spent"], best / c.ceiling, dt), flush=True)
        fr = [p["frac"] for p in per]
        recs.append({"arm": arm, "ceiling": c.ceiling, "cells": c.n_cells,
                     "best_frac_mean": float(np.mean(fr)),
                     "best_frac_std": float(np.std(fr)),
                     "worst": float(np.min(fr)),
                     "reach90_rate": float(np.mean([p["reach90"] for p in per])),
                     "seconds": float(np.mean([p["seconds"] for p in per])),
                     "per_seed": per})
        out["arms"][arm] = recs[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--init", type=int, default=8)
    ap.add_argument("--tables", default="buchwald,ccpp,concrete,gasturbine")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results = {"seeds": a.seeds, "budget": a.budget, "batch": a.batch,
               "init": a.init, "tables": {}}
    for name in a.tables.split(","):
        print("== %s" % name, flush=True)
        results["tables"][name] = one_table(
            name, a.seeds, a.budget, a.batch, a.init, a.arms.split(","))
    out = a.out or os.path.join(HERE, "multitable.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("\nwrote", out)

    # the summary that goes in the README
    for name, d in results["tables"].items():
        arms = sorted(d["arms"])
        print("\n%-11s %s" % (name, "  ".join("%-8s" % a for a in arms)))
        print("%-11s %s" % ("best/ceil", "  ".join(
            "%-8.4f" % d["arms"][a]["best_frac_mean"] for a in arms)))
        print("%-11s %s" % ("worst", "  ".join(
            "%-8.4f" % d["arms"][a]["worst"] for a in arms)))
        print("%-11s %s" % ("reach90", "  ".join(
            "%-8.2f" % d["arms"][a]["reach90_rate"] for a in arms)))


if __name__ == "__main__":
    main()
