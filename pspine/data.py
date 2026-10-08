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
        verify_pinned(path)
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
        "verified": verify_pinned(path) is not None,
        "zeros": int((y == 0).sum()),
        "censored_note": "yield == 0 means 'no product detected', not a measured 0.0",
    }
    return space, codes, y, meta


def uci_csv(path, factor_cols, target_col, rename=None):
    """Deprecated in favour of continuous_table: binning a continuous factor
    throws away the metric, and the measured tables are continuous."""
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


# ---------------------------------------------------------------------------
# Continuous process tables
#
# These are the cross-domain fuel: a power plant's hour-by-hour log, a
# turbine's emission record, a concrete mix design set. Their factors are
# *settings*, not named levels, so binning them into categories would throw
# away the metric the surrogate needs -- which is the whole point of having a
# continuous kernel.
#
# The quantisation is still real and is declared rather than hidden: a table of
# measured settings is quantised onto an n-level grid across each factor's
# [q_lo, q_hi] range, a row snaps to its nearest grid point, and a grid cell is
# answerable if at least one real row landed in it. The spacing is reported so
# the campaign's resolution is a number rather than an impression.
# ---------------------------------------------------------------------------

SHA_PINNED = {
    "buchwald/data_table.csv": "fe310b50897e97578078558909efc2edaa7b98ad8939b1caad740a718398ed62",
    "ccpp/ccpp.csv": "79e1c4524022de9468fc84289ab2c8f9ca5030790fa25e79e13443ddce4a24f3",
    "gasturbine/gasturbine.csv": "8d7e812a5e964bba5e09d3e2de9baab0a494e7b60bd11ef3b64ddf9e39874203",
    "concrete/concrete.csv": "98072ff035fc27116079be5c0075c3144677c66877fb5d95a026f1404af80aa5",
}


def verify_pinned(path):
    """Check a bundled table against its pinned hash, and raise if it moved.

    Every table is checked, not just the first: a fuel that is only sometimes
    verified is a fuel that is not verified. A mismatch means the bytes are not
    the bytes the published numbers were computed from, so the honest move is to
    stop rather than to produce a plausible-looking result from unknown data.
    """
    key = "/".join((os.path.basename(os.path.dirname(path)), os.path.basename(path)))
    want = SHA_PINNED.get(key)
    if want is None:
        return None
    got = _sha256(path)
    if got != want:
        raise ValueError(f"{key} hash mismatch: {got} != {want}")
    return got


def continuous_table(path, factor_cols, target_col, n_levels=12,
                     q=(0.01, 0.99), sha=None, source="", rename=None,
                     positive=False):
    """Load a table of measured settings as a continuous design space.

    Returns (space, codes, y, meta). `codes` are grid indices, `meta["snap"]`
    is the worst-case quantisation error as a fraction of each factor's range,
    and `meta["coverage"]` is the fraction of grid cells the table contains.
    """
    import csv as _csv
    rename = rename or {}
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = []
        for r in _csv.DictReader(f):
            if any(r.get(c, "") in ("", None) for c in list(factor_cols) + [target_col]):
                continue
            try:
                rows.append(([float(r[c]) for c in factor_cols], float(r[target_col])))
            except (TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no usable rows in {path}")
    verified = verify_pinned(path)
    Xraw = np.asarray([r[0] for r in rows], dtype=float)
    y = np.asarray([r[1] for r in rows], dtype=float)
    if positive:
        y = np.where(y <= 0, np.nan, y)
        keep = np.isfinite(y)
        Xraw, y = Xraw[keep], y[keep]

    factors, cols = [], []
    snap = {}
    for j, c in enumerate(factor_cols):
        col = Xraw[:, j]
        lo, hi = (float(np.quantile(col, q[0])), float(np.quantile(col, q[1])))
        if not (hi > lo):
            lo, hi = float(col.min()), float(col.max())
        name = rename.get(c, c)
        f = Factor.continuous_of(name, lo, hi, n=n_levels)
        factors.append(f)
        # snap each row to the nearest grid point, then to its index
        vals = f.value_of.astype(float)
        idx = np.abs(col[:, None] - vals[None, :]).argmin(axis=1)
        cols.append(idx)
        step = (hi - lo) / (n_levels - 1)
        snap[name] = step / 2.0 / (hi - lo)          # half a step, in spans
    space = DesignSpace.of(factors)
    codes = np.asarray(cols, dtype=int).T
    flat = np.ravel_multi_index(tuple(codes.T), tuple(space.cardinality))
    cells = len(set(flat.tolist()))
    meta = {
        "source": source or os.path.basename(path),
        "sha256": sha or _sha256(path),
        "verified": bool(verified),
        "rows": len(rows),
        "factors": len(factor_cols),
        "levels_per_factor": n_levels,
        "grid": space.size,
        "cells_observed": cells,
        "coverage": cells / space.size,
        "target": target_col,
        "quantisation": "half a grid step per factor, worst case",
        "snap_max_frac_of_range": snap,
    }
    return space, codes, y, meta


def ccpp(path=None):
    """Combined-cycle power plant: ambient conditions -> net hourly output."""
    path = path or os.path.join(DATADIR, "ccpp", "ccpp.csv")
    return continuous_table(
        path, ["AT", "V", "AP", "RH"], "PE", n_levels=16, q=(0.01, 0.99),
        source="UCI 294 Combined Cycle Power Plant (Kaya/Tufekci, 2012); 9568 hourly records, 2006-2011",
        rename={"AT": "ambient_temp", "V": "exhaust_vacuum", "AP": "ambient_pressure",
                "RH": "rel_humidity"})


def gasturbine(path=None):
    """Gas turbine: eight measured operating variables -> NOx emission."""
    path = path or os.path.join(DATADIR, "gasturbine", "gasturbine.csv")
    return continuous_table(
        path, ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"], "NOX",
        n_levels=5, q=(0.01, 0.99), positive=True,
        source="UCI 551 Gas Turbine CO and NOx Emission (2015); 36733 records, 2011-2015",
        rename={"AT": "ambient_temp", "AP": "ambient_pressure", "AH": "ambient_humidity",
                "AFDP": "air_filter_dp", "GTEP": "gt_exhaust_pressure",
                "TIT": "turbine_inlet_temp", "TAT": "turbine_after_temp",
                "CDP": "compressor_discharge_pressure"})


def concrete(path=None):
    """Concrete mix design: eight mix variables and curing age -> strength."""
    path = path or os.path.join(DATADIR, "concrete", "concrete.csv")
    return continuous_table(
        path, ["cement", "slag", "fly_ash", "water", "superplasticizer",
               "coarse_agg", "fine_agg", "age"], "strength",
        n_levels=4, q=(0.0, 1.0),
        source="UCI 165 Concrete Compressive Strength (Yeh, 2007); 1030 mix designs")


TABLES = {
    "buchwald": buchwald,
    "ccpp": ccpp,
    "gasturbine": gasturbine,
    "concrete": concrete,
}


def load(name):
    """Load one of the bundled tables by name."""
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; have {sorted(TABLES)}")
    return TABLES[name]()
