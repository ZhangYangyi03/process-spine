"""Loaders for the tables this package is benchmarked on.

Two rules, both learned the hard way:

  * a real table is unevenly sampled -- some factor combinations were run many
    times and others never. The grid is the *product* of the factor levels, so
    the table is a sparse observation of it, and anything that assumes a full
    factorial will silently look at the wrong thing.
  * a zero is not a measurement of zero. In the Buchwald table 0% yield means
    "no product detected", which is a censored reading. It is kept as a row
    (it is a real experiment that cost real time) but it must not be treated as
    a precise low value.

Everything is returned as (space, codes, y, meta) so a loader cannot smuggle a
filtered view of the table into a campaign.
"""
from __future__ import annotations

import csv
import hashlib
import os
from collections import OrderedDict

import numpy as np

from .space import DesignSpace, Factor

HERE = os.path.dirname(os.path.abspath(__file__))
# the tables ship inside the package, so a wheel is self-contained and a pip
# install has the same fuel as a checkout
DATADIR = os.path.join(HERE, "_datasets")

BUCHWALD_SHA256 = "fe310b50897e97578078558909efc2edaa7b98ad8939b1caad740a718398ed62"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(1 << 13)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def buchwald(path=None, verify=True):
    """The Doyle-group HTE amination set: 4 factors -> yield (%).

    base (3) x ligand (4) x aryl_halide (16) x additive (24) = 4608 possible
    cells, 4599 of them run. This is the standard real-data benchmark for
    categorical Bayesian optimisation in chemistry, which is why it is here:
    the objective it feeds is not a function of my own design.
    """
    path = path or os.path.join(DATADIR, "buchwald", "data_table.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if verify:
        got = _sha256(path)
        if got != BUCHWALD_SHA256:
            raise ValueError(f"data_table.csv hash mismatch: {got} != {BUCHWALD_SHA256}")
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("yield") in (None, ""):
                continue
            rows.append(r)
    factor_cols = ["base", "ligand", "aryl_halide", "additive"]
    levels = OrderedDict()
    for c in factor_cols:
        seen = []
        for r in rows:
            v = r[c]
            if v not in seen:
                seen.append(v)
        levels[c] = seen
    space = DesignSpace.of([Factor.of(c, levels[c]) for c in factor_cols])
    codes = np.asarray([space.encode([r[c] for c in factor_cols]) for r in rows], dtype=int)
    y = np.asarray([float(r["yield"]) for r in rows], dtype=float)
    meta = {
        "source": "doylelab/rxnpredict data_table.csv (MIT, (c) 2017 Ahneman et al.)",
        "sha256": _sha256(path),
        "rows": len(rows),
        "grid": space.size,
        "cells_observed": len({tuple(c) for c in codes}),
        "zeros": int((y == 0).sum()),
        "censored_note": "yield == 0 means 'no product detected', not a measured 0.0",
    }
    return space, codes, y, meta


def uci_csv(path, factor_cols, target_col, rename=None):
    """Generic loader for a plain numeric CSV/XLS-derived table.

    Cuts each numeric factor into equal-frequency bins so that the same
    categorical machinery applies. Binning is a real simplification and is
    declared as such in the README: for a continuous factor, a campaign that
    wants the exact optimum needs a continuous kernel, and this package does
    not pretend to have one.
    """
    import csv as _csv
    rename = rename or {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rdr = _csv.DictReader(f)
        rows = [r for r in rdr if all(r.get(c, "") != "" for c in factor_cols + [target_col])]
    Xraw = np.asarray([[float(r[c]) for c in factor_cols] for r in rows], dtype=float)
    y = np.asarray([float(r[target_col]) for r in rows], dtype=float)
    factors, codes = [], []
    for j, c in enumerate(factor_cols):
        col = Xraw[:, j]
        qs = np.quantile(col, np.linspace(0, 1, 6)[1:-1])
        bins = np.digitize(col, qs)
        name = rename.get(c, c)
        labels = [f"{name}_q{k}" for k in range(len(qs) + 1)]
        factors.append(Factor.of(name, labels))
        codes.append(bins)
    space = DesignSpace.of(factors)
    codes = np.asarray(codes, dtype=int).T
    meta = {"source": os.path.basename(path), "rows": len(rows),
            "binning": "5 equal-frequency bins per factor (declared simplification)",
            "target": target_col}
    return space, codes, y, meta
