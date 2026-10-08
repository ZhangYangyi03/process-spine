"""The campaign loop: fit, propose, observe, repeat -- and what it costs.

The unit of cost is an *experiment*, not a model call. That is the entire reason
this package exists: in a simulation the objective can be called a million
times, in a process it cannot, so every arm is scored in queries against a fixed
budget.

A table is a *sparse* observation of the design grid. The Buchwald set covers
4599 of the 4608 cells its four factors define, and other tables are far thinner.
Nothing here may assume a full factorial: the grid is the product of the levels,
a cell is queryable only if the table contains it, and sampling a cell the lab
never ran is exactly the mistake this module is built not to make.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import baselines as B
from .gp import MixedGP


def grid_from_table(space, codes, y):
    """Fold a table of rows into the design grid.

    Returns (y_grid, present) where y_grid[cell] is the mean response of the rows
    that landed in that cell and present[cell] says whether the lab ever ran it.
    Replicates are averaged rather than dropped: a cell run twice is one design
    at two measurements, and the campaign's budget counts the design once.
    """
    codes = np.asarray(codes, dtype=int)
    y = np.asarray(y, dtype=float)
    flat = np.ravel_multi_index(tuple(codes.T), tuple(space.cardinality))
    n = space.size
    yg = np.full(n, np.nan)
    present = np.zeros(n, bool)
    sums = np.zeros(n); counts = np.zeros(n)
    np.add.at(sums, flat, y)
    np.add.at(counts, flat, 1.0)
    seen = counts > 0
    yg[seen] = sums[seen] / counts[seen]
    present[seen] = True
    return yg, present, counts


@dataclass
class Campaign:
    space: object
    codes: object                 # (n_rows, dim) integer codes from the table
    y: object                     # (n_rows,) response
    batch: int = 8
    acq: str = "logei"
    seed: int = 0
    init: int = 8
    acq_kw: dict = field(default_factory=dict)

    def __post_init__(self):
        self.codes = np.asarray(self.codes, dtype=int)
        self.y = np.asarray(self.y, dtype=float)
        self.y_grid, self.present, self.replicates = grid_from_table(self.space, self.codes, self.y)
        self.rng = np.random.default_rng(self.seed)
        self.ceiling = float(np.nanmax(self.y_grid))

    # ---- public numbers -------------------------------------------------
    @property
    def n_cells(self):
        return int(self.present.sum())

    @property
    def coverage(self):
        return self.present.sum() / self.space.size

    def query(self, cell):
        """What an experiment at this cell would return. Only defined for a cell
        the table actually contains -- an unrun cell is not a cheap observation,
        it is an experiment that has not happened yet."""
        if not self.present[cell]:
            raise KeyError(f"cell {cell} was never run; it is not in the table")
        return float(self.y_grid[cell])

    # ---- the loop -------------------------------------------------------
    def run(self, budget, arm="logei"):
        mask = self.present.copy()            # queryable cells
        observed = []
        trace = []
        for i in self._initial(mask):
            mask[i] = False
            observed.append((i, self.query(i)))
        trace.append(self._snap(observed, mask))
        while len(observed) < budget:
            nb = min(self.batch, budget - len(observed))
            picks = self._pick(arm, mask, nb, observed)
            if not picks:
                break
            for i in picks:
                if not mask[i]:
                    continue
                mask[i] = False
                observed.append((i, self.query(i)))
            trace.append(self._snap(observed, mask))
        return trace

    def _initial(self, mask):
        idx = np.flatnonzero(mask)
        k = min(self.init, len(idx))
        return list(self.rng.choice(idx, size=k, replace=False))

    def _pick(self, arm, mask, nb, observed):
        codes_grid = np.asarray(self.space.all_codes(), dtype=int)
        if arm == "random":
            return B.random_arm(mask, nb, self.rng)
        if arm == "fill":
            return B.fill_arm(codes_grid, mask, nb)
        if arm == "ofat":
            return B.ofat_arm(self.space, codes_grid, mask, observed, nb, self.rng)
        return self._propose(arm, mask, nb, observed, codes_grid)

    def _propose(self, arm, mask, nb, observed, codes_grid):
        X = np.asarray([o[0] for o in observed], dtype=int)
        Xc = codes_grid[X]
        y = np.asarray([o[1] for o in observed], dtype=float)
        gp = MixedGP(cardinality=self.space.cardinality).fit(Xc, y)
        cand = np.flatnonzero(mask)
        mu, sd = gp.predict(codes_grid[cand])
        best = float(y.max())
        from . import acq
        picks = acq.greedy_batch(arm, mu, sd, np.ones(len(cand), bool), nb, best, **self.acq_kw)
        return [int(cand[p]) for p in picks]

    def _snap(self, observed, mask):
        vals = [v for _, v in observed]
        return {"spent": len(observed), "best": max(vals) if vals else float("nan"),
                "n_observed": len(observed), "remaining": int(mask.sum())}

    def last_gp(self, observed):
        codes_grid = np.asarray(self.space.all_codes(), dtype=int)
        Xc = codes_grid[[o[0] for o in observed]]
        y = np.asarray([o[1] for o in observed], dtype=float)
        return MixedGP(cardinality=self.space.cardinality).fit(Xc, y)

    def regret_curve(self, trace):
        return [(t["spent"], t["best"] / self.ceiling if self.ceiling else np.nan) for t in trace]
