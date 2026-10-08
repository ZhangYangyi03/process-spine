"""The comparison arms. A campaign result means nothing without them.

    random    what most labs do without a model
    ofat      one-factor-at-a-time: the classic process move -- hold everything,
              vary one knob, keep the improvement, move to the next knob
    fill      greedy max-min distance in one-hot space: the "cover the space"
              design a DoE course would teach
    oracle    the best rows of the *whole finished table*. Not a competitor: a
              ceiling. A campaign can never beat reading the table it is drawing
              from, and printing that number is what keeps the rest honest.
"""
from __future__ import annotations

import numpy as np


def random_arm(mask, batch, rng):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    return [int(i) for i in rng.choice(idx, size=min(batch, len(idx)), replace=False)]


def fill_arm(codes_grid, mask, batch):
    """Greedy max-min: pick the point furthest (in one-hot L1) from everything
    already chosen. Deterministic, so two runs of this arm agree."""
    chosen = codes_grid[~mask]
    idx = list(np.flatnonzero(mask))
    if not idx:
        return []
    if len(chosen) == 0:
        return [idx[0]]
    picks = []
    for _ in range(min(batch, len(idx))):
        best, bestd = None, -1
        for i in idx:
            d = int((codes_grid[i][None, :] != chosen).sum())
            if d > bestd:
                best, bestd = i, d
        picks.append(best)
        chosen = np.vstack([chosen, codes_grid[best][None, :]])
        idx.remove(best)
    return picks


def ofat_arm(space, codes_grid, mask, observed, batch, rng):
    """Coordinate ascent with no surrogate, which is what a careful engineer
    does: start from the best point seen, vary one factor at a time, take the
    level with the best observed mean, keep it, move on."""
    if not observed:
        return random_arm(mask, batch, rng)
    Xo = codes_grid[[o[0] for o in observed]]
    yo = np.asarray([o[1] for o in observed], dtype=float)
    cur = Xo[int(np.argmax(yo))].copy()
    picks, used = [], set()
    for _ in range(min(batch, int(mask.sum()))):
        placed = False
        for j in range(len(space.factors)):
            scores = []
            for lv in range(len(space.factors[j].levels)):
                sel = Xo[:, j] == lv
                scores.append(float(yo[sel].mean()) if sel.any() else -np.inf)
            if not np.isfinite(max(scores)):
                continue
            cand = cur.copy(); cand[j] = int(np.argmax(scores))
            flat = int(np.ravel_multi_index(tuple(cand), tuple(space.cardinality)))
            if mask[flat] and flat not in used:
                picks.append(flat); used.add(flat); cur = cand; placed = True
                break
        if not placed:
            rest = [i for i in random_arm(mask, 32, rng) if i not in used]
            if not rest:
                break
            picks.append(rest[0]); used.add(rest[0])
            cur = codes_grid[rest[0]].copy()
    return picks


def oracle_at_budget(y_grid, mask, budget):
    """The ceiling: mean of the `budget` best cells that the table contains."""
    vals = np.sort(y_grid[mask])[::-1]
    return float(vals[:budget].mean()) if len(vals) else float("nan")
