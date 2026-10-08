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
    """Greedy max-min in code space: pick the cell furthest from everything
    chosen so far. Deterministic.

    Vectorised, because the naive form is O(n_candidates x n_chosen) *per pick*
    in pure Python and that is minutes per seed on a 4000-cell table -- the arm
    is a control, and a control that is 100x slower than the methods it controls
    is a control that quietly does not get run.
    """
    G = np.asarray(codes_grid)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    picks = []
    alive = np.ones(len(idx), bool)
    if (~mask).any():
        chosen = G[~mask]
        d = np.min(np.abs(G[idx][:, None, :] - chosen[None, :, :]).sum(axis=2), axis=1)
    else:
        # nothing chosen yet: seed with the first candidate, then the loop below
        # spreads the rest away from it. Returning early here would hand back a
        # one-point batch, and a control arm that under-fills its batch is a
        # control comparing against a smaller budget.
        d = np.zeros(len(idx))
    for _ in range(min(batch, len(idx))):
        if not alive.any():
            break
        d_masked = np.where(alive, d, -np.inf)
        j = int(np.argmax(d_masked))
        picks.append(int(idx[j]))
        alive[j] = False
        # distance from the new pick, folded into the running min
        nd = np.abs(G[idx] - G[idx[j]][None, :]).sum(axis=1)
        d = np.minimum(d, nd)
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
            n = len(space.factors[j].levels)
            neighbour = cur.copy()
            # One grid step, not "the level with the best mean anywhere in the
            # range": for a continuous factor, jumping to a distant level is not
            # one-factor-at-a-time, it is a scan. A categorical factor has no
            # neighbour relation, so there the level with the best observed mean
            # is the only sensible move.
            if getattr(space.factors[j], "continuous", False):
                for step in (1, -1):
                    cand = cur.copy()
                    lv = int(np.clip(cand[j] + step, 0, n - 1))
                    if lv == cand[j]:
                        continue
                    cand[j] = lv
                    flat = int(np.ravel_multi_index(tuple(cand), tuple(space.cardinality)))
                    sel = np.flatnonzero((codes_grid == cand).all(axis=1))
                    if len(sel) and mask[sel[0]] and sel[0] not in used:
                        picks.append(int(sel[0])); used.add(int(sel[0]))
                        cur = cand; placed = True
                        break
            else:
                scores = []
                for lv in range(n):
                    sel = Xo[:, j] == lv
                    scores.append(float(yo[sel].mean()) if sel.any() else -np.inf)
                if not np.isfinite(max(scores)):
                    continue
                cand = cur.copy(); cand[j] = int(np.argmax(scores))
                sel = np.flatnonzero((codes_grid == cand).all(axis=1))
                if len(sel) and mask[sel[0]] and sel[0] not in used:
                    picks.append(int(sel[0])); used.add(int(sel[0]))
                    cur = cand; placed = True
            if placed:
                break
        if not placed:
            rest = [i for i in random_arm(mask, 32, rng) if i not in used]
            if not rest:
                break
            picks.append(rest[0]); used.add(rest[0])
            cur = codes_grid[rest[0]].copy()
    return picks


def oracle_at_budget(cell_y, budget):
    """The ceiling: mean of the `budget` best cells the table contains. A
    campaign can never beat reading the finished table, and printing this keeps
    the rest honest."""
    vals = np.sort(np.asarray(cell_y, dtype=float))[::-1]
    return float(vals[:budget].mean()) if len(vals) else float("nan")
