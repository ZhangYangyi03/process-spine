"""Run the campaigns and write bench/results.json.

This is launched as an *external* child process (wmic process call create),
because the process that requests these numbers runs under a hard heap ceiling
of a couple of MB and cannot parse the 1.2 MB table at all, let alone hold a
300x300 kernel. The child has the host's real memory. It writes JSON and a
progress file; neither is returned through stdout, because the launch mechanism
does not give the child a usable stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pspine import baselines as B
from pspine import data as pdata
from pspine.loop import Campaign

HERE = os.path.dirname(os.path.abspath(__file__))


def PROG(msg):
    with open(os.path.join(HERE, "progress.txt"), "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def run_one(space, codes, y, arm, seed, budget, batch, init):
    c = Campaign(space, codes, y, batch=batch, acq=arm, seed=seed, init=init)
    t0 = time.time()
    tr = c.run(budget=budget, arm=arm)
    ceil = float(np.nanmax(c.y_grid))
    return {
        "arm": arm, "seed": seed, "budget": budget,
        "seconds": round(time.time() - t0, 2),
        "best": tr[-1]["best"], "best_frac": tr[-1]["best"] / ceil,
        "q90": next((t["spent"] for t in tr if t["best"] >= 0.90 * ceil), None),
        "q95": next((t["spent"] for t in tr if t["best"] >= 0.95 * ceil), None),
        "trace": tr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--init", type=int, default=8)
    ap.add_argument("--acqs", default="ei,logei,ucb,pi")
    args = ap.parse_args()

    PROG("start")
    space, codes, y, meta = pdata.buchwald()
    c0 = Campaign(space, codes, y)
    PROG("loaded %s | cells %d/%d (%.1f%%) | max %.1f mean %.2f | zeros %d" % (
        space, c0.n_cells, space.size, 100 * c0.coverage,
        np.nanmax(c0.y_grid), np.nanmean(c0.y_grid), int((y == 0).sum())))

    arms = ["random", "ofat", "fill"] + [a.strip() for a in args.acqs.split(",") if a.strip()]
    out = {
        "dataset": meta, "space": str(space), "grid_size": space.size,
        "cells_present": c0.n_cells, "coverage": c0.coverage,
        "ceiling": float(np.nanmax(c0.y_grid)), "ceiling_note":
            "max cell mean in the finished table; the line no campaign can cross",
        "budget": args.budget, "batch": args.batch, "init": args.init,
        "seeds": args.seeds, "runs": [],
    }
    for arm in arms:
        for seed in range(args.seeds):
            rec = run_one(space, codes, y, arm, seed, args.budget, args.batch, args.init)
            out["runs"].append(rec)
            PROG("%-6s seed %d best %6.2f (%.3f) q90=%-4s %.0fs" % (
                arm, seed, rec["best"], rec["best_frac"], rec["q90"], rec["seconds"]))
    agg = {}
    for arm in arms:
        rs = [r for r in out["runs"] if r["arm"] == arm]
        if not rs:
            continue
        q90s = [r["q90"] for r in rs]
        hit = [q for q in q90s if q is not None]
        agg[arm] = {
            "n_seeds": len(rs),
            "best_frac_mean": float(np.mean([r["best_frac"] for r in rs])),
            "best_frac_std": float(np.std([r["best_frac"] for r in rs])),
            "best_worst": float(np.min([r["best"] for r in rs])),
            "q90_hit_rate": len(hit) / len(rs),
            "q90_mean": float(np.mean(hit)) if hit else None,
            "seconds_mean": float(np.mean([r["seconds"] for r in rs])),
        }
    out["aggregate"] = agg
    json.dump(out, open(args.out, "w"), indent=1)
    PROG("wrote " + args.out)

    # human-readable summary
    lines = ["arm     best_frac  std    worst   q90_hit  q90_mean  s/run"]
    for arm, a in agg.items():
        lines.append("%-8s %8.3f %6.3f %7.2f %8.2f %9s %6.1f" % (
            arm, a["best_frac_mean"], a["best_frac_std"], a["best_worst"],
            a["q90_hit_rate"], ("%.1f" % a["q90_mean"]) if a["q90_mean"] else "-",
            a["seconds_mean"]))
    open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    PROG("SUMMARY\n" + "\n".join(lines))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        PROG("TRACEBACK:" + traceback.format_exc())
        raise
