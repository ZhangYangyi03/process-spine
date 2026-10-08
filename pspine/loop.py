"""The campaign loop: fit, propose, observe, repeat -- and what it costs.

The unit of cost is an *experiment*, not a model call. That is the entire reason
this package exists: in a simulation the objective can be called a million
times, in a process it cannot, so every arm is scored in queries against a fixed
budget.

A candidate is a row of the table. That is the same rule as the Buchwald set and
it is deliberately the same rule for the continuous tables: an eight-factor
turbine table defines a 390625-cell product grid and a 1030-row concrete table
defines 65536 cells, and in both cases the cells the lab or the plant actually
ran are a few thousand at most. Scoring the whole product would be scoring
settings nobody has ever measured, which is the mistake this module exists not
to make. So `cells` is the distinct observed cells, a cell's response is the
mean of the rows that landed in it, and the campaign can only spend a query on a
cell that is in `cells`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import baselines as B
from .gp import MixedGP


def cells_from_table(space, codes, y):
    """Fold a table of rows into its distinct cells.

    Returns (cell_codes, cell_y, cell_n, draw_multiplicity) where
    `cell_codes[k]` is the integer code vector of the k-th distinct cell the
    table contains, `cell_y[k]` is the mean response of the rows in it, and
    `cell_n[k]` is how many rows it cost. Replicates are averaged rather than
    dropped: a cell run twice is one design at two measurements, and the
    campaign's budget counts the design once.
    """
    codes = np.asarray(codes, dtype=int)
    y = np.asarray(y, dtype=float)
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)
    nf = len(space.cardinality)
    if codes.shape[1] != nf:
        raise ValueError(f"table has {codes.shape[1]} factor columns, space has {nf}")
    key = np.ravel_multi_index(tuple(codes.T), tuple(space.cardinality))
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, y)
    cell_y = sums / counts
    # one representative row per cell, in the order np.unique sorted the keys
    first = np.unique(inv, return_index=True)[1]
    cell_codes = codes[first]
    cell_y = sums / counts
    return cell_codes, cell_y, counts, inv


@dataclass
class Campaign:
    """A budgeted sequence of experiments drawn from one measured table."""
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
        if self.codes.ndim == 1:
            self.codes = self.codes.reshape(-1, 1)
        self.cells, self.cell_y, self.replicates, _ = cells_from_table(
            self.space, self.codes, self.y)
        self.n_cells = len(self.cells)
        self.rng = np.random.default_rng(self.seed)
        self.ceiling = float(self.cell_y.max())
        self.n_factors = len(self.space.cardinality)

    # ---- public numbers -------------------------------------------------
    @property
    def coverage(self):
        """Fraction of the design space the table actually covers. Reported,
        never assumed: the Buchwald set is 99.8% complete and the concrete set
        is 0.4%, and a campaign's difficulty is not the same thing in the two."""
        return self.n_cells / self.space.size

    def query(self, cell):
        """What an experiment at this cell would return.

        Only defined for a cell the table contains -- a cell nobody ran is not a
        cheap observation, it is an experiment that has not happened yet, and
        the whole point of this package is that such a query is impossible.
        """
        if not (0 <= cell < self.n_cells):
            raise KeyError(f"cell index {cell} is out of range; the table has {self.n_cells}")
        return float(self.cell_y[cell])

    # ---- the loop -------------------------------------------------------
    def run(self, budget, arm="ei"):
        available = np.ones(self.n_cells, bool)
        observed = []
        trace = []
        for i in self._initial(available):
            available[i] = False
            observed.append((i, self.query(i)))
        trace.append(self._snap(observed, available))
        while len(observed) < budget:
            nb = min(self.batch, budget - len(observed))
            picks = self._pick(arm, available, nb, observed)
            picks = [p for p in picks if available[p]]
            if not picks:
                break
            for i in picks:
                available[i] = False
                observed.append((i, self.query(i)))
            trace.append(self._snap(observed, available))
        return trace

    def _initial(self, available):
        idx = np.flatnonzero(available)
        k = min(self.init, len(idx))
        return list(self.rng.choice(idx, size=k, replace=False))

    def _pick(self, arm, available, nb, observed):
        if arm == "random":
            return B.random_arm(available, nb, self.rng)
        if arm == "fill":
            return B.fill_arm(self.cells, available, nb)
        if arm == "ofat":
            return B.ofat_arm(self.space, self.cells, available, observed, nb, self.rng)
        return self._propose(arm, available, nb, observed)

    def fit_gp(self, observed):
        Xc = self.cells[[o[0] for o in observed]]
        y = np.asarray([o[1] for o in observed], dtype=float)
        return MixedGP.for_space(self.space).fit(Xc, y)

    def _propose(self, arm, available, nb, observed):
        from . import acq
        gp = self.fit_gp(observed)
        cand = np.flatnonzero(available)
        best = max(v for _, v in observed)
        if arm in ("qei", "qei_pess"):
            # Batch EI: score the *set* by conditioning on each pick in turn,
            # instead of taking the top-k of one EI pass and hoping the points
            # differ. The pessimism variant hallucinates a low value so the
            # batch spreads further across the uncertainty.
            pess = 1.0 if arm == "qei" else 0.5
            picks = acq.q_ei_batch(gp, self.cells[cand], nb, best,
                                   pessimism=pess, chunk=self.acq_kw.get("chunk", 192))
            return [int(cand[p]) for p in picks]
        mu, sd = gp.predict(self.cells[cand])
        picks = acq.greedy_batch(arm, mu, sd, np.ones(len(cand), bool), nb, best,
                                 **self.acq_kw)
        return [int(cand[p]) for p in picks]

    def _snap(self, observed, available):
        vals = [v for _, v in observed]
        return {"spent": len(observed), "best": max(vals) if vals else float("nan"),
                "n_observed": len(observed), "remaining": int(available.sum())}

    def last_gp(self, observed):
        return self.fit_gp(observed)

    def regret_curve(self, trace):
        return [(t["spent"], t["best"] / self.ceiling if self.ceiling else np.nan)
                for t in trace]
