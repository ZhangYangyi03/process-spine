"""Tests that would fail if the numbers in the README stopped being true.

The distinction this suite keeps: *behaviour* tests pin the mechanics (a cell
the lab never ran cannot be queried; the same seed gives the same trace), while
*reproduction* tests pin the published result (the arms still rank the way the
README says). A reproduction test is allowed to be slow and is allowed to fail
loudly -- that is its job.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pspine import acq, baselines as B, data as pdata
from pspine.gp import MixedGP
from pspine.loop import Campaign, grid_from_table
from pspine.space import DesignSpace, Factor


# ---- the design space -----------------------------------------------------
def test_space_product_and_roundtrip():
    sp = DesignSpace.of([Factor.of("a", ["x", "y"]), Factor.of("b", [1, 2, 3])])
    assert sp.size == 6 and sp.dim == 2
    for code in sp.all_codes():
        assert sp.encode(sp.point_for(code)) == code


def test_space_rejects_a_bad_level():
    sp = DesignSpace.of([Factor.of("a", ["x", "y"])])
    with pytest.raises(KeyError):
        sp.encode(["nope"])


def test_space_needs_two_levels():
    with pytest.raises(ValueError):
        Factor.of("a", ["only"])


# ---- the table is sparse, and the code must behave as if it is ------------
def test_a_cell_the_lab_never_ran_cannot_be_queried():
    sp = DesignSpace.of([Factor.of("a", ["x", "y"]), Factor.of("b", ["p", "q"])])
    codes = np.array([[0, 0], [1, 1]])            # 2 of 4 cells observed
    y = np.array([1.0, 2.0])
    c = Campaign(sp, codes, y)
    assert c.n_cells == 2 and sp.size == 4
    with pytest.raises(KeyError):
        c.query(1)                                # cell (0,1) was never run
    assert c.query(0) == 1.0


def test_replicates_are_averaged_not_dropped():
    sp = DesignSpace.of([Factor.of("a", ["x", "y"])])
    codes = np.array([[0], [0], [1]])
    y = np.array([10.0, 20.0, 5.0])
    yg, present, counts = grid_from_table(sp, codes, y)
    assert yg[0] == pytest.approx(15.0)
    assert counts[0] == 2 and present.all()


# ---- the surrogate --------------------------------------------------------
def test_gp_interpolates_its_training_points():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([1.0, 5.0])
    gp = MixedGP(cardinality=(2, 2)).fit(X, y)
    mu, sd = gp.predict(X)
    assert np.allclose(mu, y, atol=0.35)
    assert np.all(sd >= 0)


def test_gp_kernel_is_symmetric_and_bounded():
    X = np.array([[0, 0], [0, 1], [1, 1]])
    gp = MixedGP(cardinality=(2, 2))
    K = gp._K(X, X)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), gp.signal)


def test_gp_prediction_is_chunk_invariant():
    """Chunking is an allocation choice and must not move a single number."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(40, 3))
    y = rng.normal(size=40)
    gp = MixedGP(cardinality=(3, 3, 3)).fit(X, y)
    Q = rng.integers(0, 3, size=(500, 3))
    m1, s1 = gp.predict(Q, chunk=7)
    m2, s2 = gp.predict(Q, chunk=500)
    assert np.allclose(m1, m2) and np.allclose(s1, s2)


def test_gp_length_scales_are_reported_per_factor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1.0, 1.0, 5.0, 5.0])            # only factor 0 matters
    gp = MixedGP(cardinality=(2, 2)).fit(X, y)
    fs = gp.factor_scale()
    assert 0 in fs and 1 in fs


# ---- acquisition functions ------------------------------------------------
def test_ucb_is_a_stated_trade_between_mean_and_width():
    """UCB moves with kappa, and that is the whole content of the arm.

    At kappa=3 the wide-but-mediocre third point (1 + 3*5) beats the narrow
    certainty of the second (10 + 3*0.001). At kappa=0 it cannot. A test that
    pinned one winner would be pinning a convention, not a property.
    """
    mu = np.array([0.0, 10.0, 1.0])
    sd = np.array([1.0, 0.001, 5.0])
    assert acq.score("ucb", mu, sd, best=10.0, kappa=3.0).argmax() == 2
    assert acq.score("ucb", mu, sd, best=10.0, kappa=0.0).argmax() == 1


def test_ei_ranks_the_uncertain_near_winner_above_the_certain_also_ran():
    """EI is not "pick the point with the best mean", and the difference is the
    point. A candidate at the incumbent with no spread has almost nothing left
    to gain (bounded by xi); a candidate a little below it but very uncertain
    has real expected gain. If this ever stops holding, EI has become UCB."""
    mu = np.array([0.0, 10.0, 1.0])
    sd = np.array([1.0, 0.001, 5.0])
    vals = acq.score("ei", mu, sd, best=10.0)
    assert np.isfinite(vals).all() and (vals >= 0).all()
    assert vals[2] > vals[1]                      # wide-but-below beats certain-incumbent
    assert vals[1] <= 0.011                        # incumbent's remaining EI is ~xi
    assert np.isfinite(acq.score("logei", mu, sd, best=10.0)).all()


def test_unknown_acquisition_raises():
    with pytest.raises(KeyError):
        acq.score("nope", np.zeros(2), np.ones(2), best=0.0)


def test_batch_picks_do_not_repeat_a_point():
    mu = np.array([5.0, 4.0, 3.0, 2.0])
    sd = np.ones(4)
    picks = acq.greedy_batch("ei", mu, sd, np.ones(4, bool), 3, best=0.0)
    assert len(set(picks)) == 3


# ---- baselines ------------------------------------------------------------
def test_fill_arm_is_deterministic_and_spreads():
    sp = DesignSpace.of([Factor.of("a", ["x", "y"]), Factor.of("b", ["p", "q"])])
    grid = np.asarray(sp.all_codes(), int)
    mask = np.ones(4, bool); mask[0] = False     # (0,0) already done
    p1 = B.fill_arm(grid, mask, 2)
    p2 = B.fill_arm(grid, mask, 2)
    assert p1 == p2 and len(set(p1)) == 2


def test_oracle_is_the_best_rows_not_the_mean():
    ygrid = np.array([1.0, 9.0, 5.0, 3.0])
    mask = np.ones(4, bool)
    assert B.oracle_at_budget(ygrid, mask, 2) == pytest.approx(7.0)


# ---- the loop -------------------------------------------------------------
def test_same_seed_gives_the_same_trace():
    sp = DesignSpace.of([Factor.of("a", list("abc")), Factor.of("b", list("pq"))])
    grid = np.asarray(sp.all_codes(), int)
    y = (grid[:, 0] * 3 + grid[:, 1]).astype(float)
    t1 = Campaign(sp, grid, y, seed=3, init=2, batch=2).run(budget=6, arm="random")
    t2 = Campaign(sp, grid, y, seed=3, init=2, batch=2).run(budget=6, arm="random")
    assert t1 == t2


def test_campaign_never_exceeds_its_budget():
    sp = DesignSpace.of([Factor.of("a", list("abcd"))])
    grid = np.asarray(sp.all_codes(), int)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    tr = Campaign(sp, grid, y, seed=0, init=1, batch=2).run(budget=3, arm="ei")
    assert tr[-1]["spent"] <= 3
    assert all(t["spent"] <= 3 for t in tr)


def test_campaign_cannot_observe_more_than_the_table_has():
    sp = DesignSpace.of([Factor.of("a", list("ab"))])
    grid = np.asarray(sp.all_codes(), int)
    y = np.array([1.0, 2.0])
    tr = Campaign(sp, grid, y, seed=0, init=1, batch=1).run(budget=10, arm="random")
    assert tr[-1]["spent"] == 2                   # two cells exist, not ten


# ---- a GP arm on a tiny problem ------------------------------------------
# The GP arms are exercised on a small sub-grid: they are slow by construction
# (a marginal-likelihood fit per batch), and this suite has to stay runnable.
def _mini_dataset():
    sp = DesignSpace.of([Factor.of("a", list("abcd")), Factor.of("b", list("pq"))])
    grid = np.asarray(sp.all_codes(), int)
    y = (6.0 - np.abs(grid[:, 0] - 1) * 2.0 - grid[:, 1]).astype(float)
    return sp, grid, y


def test_gp_arm_runs_and_beats_nothing_in_particular():
    sp, grid, y = _mini_dataset()
    tr = Campaign(sp, grid, y, seed=0, init=4, batch=2).run(budget=8, arm="ei")
    assert tr[-1]["spent"] == 8
    assert tr[-1]["best"] >= np.sort(y)[-3]       # a sane arm on an easy surface


# ---- the published result -------------------------------------------------
BENCH = os.path.join(ROOT, "bench", "results_full.json")


@pytest.mark.skipif(not os.path.exists(BENCH), reason="bench output not present")
def test_published_arms_still_land_where_the_readme_says():
    d = json.load(open(BENCH))
    agg = d["aggregate"]
    assert set(agg) >= {"random", "ofat", "fill", "ei", "logei", "ucb", "pi"}
    # the three claims the README makes, each with room for a rerun
    assert agg["ei"]["best_frac_mean"] > agg["random"]["best_frac_mean"]
    assert agg["fill"]["best_frac_mean"] < 0.95
    assert abs(agg["ofat"]["best_frac_mean"] - agg["ei"]["best_frac_mean"]) < 0.03
    assert d["ceiling"] == pytest.approx(100.0)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "bench", "gp_fidelity.json")),
                    reason="fidelity output not present")
def test_the_surrogate_is_weak_exactly_where_the_readme_says():
    d = json.load(open(os.path.join(ROOT, "bench", "gp_fidelity.json")))
    by_n = {s["n_train"]: s for s in d["sizes"]}
    assert by_n[16]["spearman_mean"] < 0.4          # weak at the start
    assert by_n[256]["spearman_mean"] > 0.7         # better with more cells
    assert by_n[256]["top10_hit_mean"] < 0.25       # and still cannot point
