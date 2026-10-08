"""How good is the surrogate, before blaming the acquisition function.

A campaign result is a joint statement about (a) how well the surrogate predicts
unqueried cells and (b) how the acquisition trades exploration against it. If
(a) is bad, a Bayesian arm losing to space-filling is a fact about the surrogate,
not about Bayesian optimisation, and the honest report has to say which.

The split here is deliberately hostile to the model: train on a random subset of
the cells, predict the rest, and report rank correlation on the *held-out* cells.
Rank correlation rather than RMSE because the acquisition only uses the surrogate
to order candidates -- a model that gets the order right and the scale wrong is
fine, and one that does the reverse is not.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pspine import data as pdata
from pspine.gp import MixedGP

HERE = os.path.dirname(os.path.abspath(__file__))


def PROG(msg):
    with open(os.path.join(HERE, "gp_fidelity.log"), "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    space, codes, y, meta = pdata.buchwald()
    t0 = time.time()
    flat = np.ravel_multi_index(tuple(np.asarray(codes, int).T), tuple(space.cardinality))
    yg = np.full(space.size, np.nan)
    yg[flat] = y
    present = ~np.isnan(yg)
    cells = np.flatnonzero(present)
    grid = np.asarray(space.all_codes(), dtype=int)
    out = {"space": str(space), "n_cells": int(present.sum()), "sizes": []}
    rng = np.random.default_rng(0)
    for n_train in (16, 32, 64, 128, 256):
        rows = []
        for rep in range(3):
            tr = rng.choice(cells, size=n_train, replace=False)
            te = np.setdiff1d(cells, tr)
            gp = MixedGP(cardinality=space.cardinality).fit(grid[tr], yg[tr])
            mu, sd = gp.predict(grid[te])
            rows.append({
                "spearman": spearman(mu, yg[te]),
                "pearson": float(np.corrcoef(mu, yg[te])[0, 1]),
                "rmse": float(np.sqrt(np.mean((mu - yg[te]) ** 2))),
                "top10_hit": float(np.mean(np.isin(np.argsort(-mu)[:10],
                                                   np.argsort(-yg[te])[:10]))),
                "ls": [float(v) for v in gp.ls],
            })
            PROG("n_train=%3d rep=%d spearman=%.3f top10=%.2f ls=%s" % (
                n_train, rep, rows[-1]["spearman"], rows[-1]["top10_hit"],
                np.round(gp.ls, 2).tolist()))
        out["sizes"].append({
            "n_train": n_train,
            "spearman_mean": float(np.mean([r["spearman"] for r in rows])),
            "pearson_mean": float(np.mean([r["pearson"] for r in rows])),
            "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
            "top10_hit_mean": float(np.mean([r["top10_hit"] for r in rows])),
            "ls_last": rows[-1]["ls"],
        })
    # the honest reference for top10_hit: random ranking
    out["random_top10_hit"] = 10.0 / max(1, (len(cells) - 256))
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "gp_fidelity.json"), "w"), indent=1)
    PROG("wrote gp_fidelity.json in %.1fs" % out["seconds"])


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        PROG("TRACEBACK:" + traceback.format_exc())
        raise
