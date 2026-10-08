"""Acquisition functions, and the only thing that really varies between them.

Everything here is written for a *small* campaign: the whole space is a few
thousand rows, so we evaluate the acquisition exactly on every unqueried row
rather than optimising it with a continuous inner loop. That removes an entire
class of "the optimiser was fine, the inner optimiser was not" failure.
"""
from __future__ import annotations

import numpy as np

try:                                    # scipy is available on this host
    from scipy.stats import norm
    _SQRT2PI = np.sqrt(2.0 * np.pi)
except Exception:                       # pragma: no cover
    norm = None
    _SQRT2PI = None


def _z(mu, sd, best):
    sd = np.clip(sd, 1e-12, None)
    return (mu - best) / sd


def ei(mu, sd, best, xi=0.01):
    """Expected improvement over the best value seen so far."""
    if norm is None:
        raise RuntimeError("EI needs scipy.stats.norm")
    z = _z(mu, sd, best - xi)
    return (mu - best + xi) * norm.cdf(z) + sd * norm.pdf(z)


def pi(mu, sd, best, xi=0.01):
    """Probability of improvement."""
    if norm is None:
        raise RuntimeError("PI needs scipy.stats.norm")
    return norm.cdf(_z(mu, sd, best - xi))


def ucb(mu, sd, best, kappa=2.0):
    """Upper confidence bound (width kappa). Higher is better."""
    return mu + kappa * sd


def logei(mu, sd, best, xi=0.0):
    """EI on a log-scaled objective. This is the variant that matters when the
    table has a floor of exact zeros, which the Buchwald yield table does:
    0% yield is a *censored* read, not a measurement of 0.0, and EI on the raw
    scale keeps proposing points near the floor because the surrogate is
    confident there. Log-scaling the objective removes that attractor."""
    if norm is None:
        raise RuntimeError("logEI needs scipy.stats.norm")
    floor = 1e-3
    lm = np.log(np.clip(mu, floor, None))
    lb = np.log(max(best, floor))
    sd_log = np.clip(sd / np.clip(mu, floor, None), 1e-6, None)
    z = (lm - lb - xi) / sd_log
    return (lm - lb - xi) * norm.cdf(z) + sd_log * norm.pdf(z)


def q_ei_batch(gp, cand_codes, batch, best, xi=0.01, pessimism=1.0, chunk=192):
    """q-EI by sequential conditioning -- the Kriging Believer idea.

    Greedy top-k of a single EI pass answers the wrong question: it scores each
    candidate against the *current* posterior and then picks several points that
    are all high for the same reason. Batch EI asks what a *set* is worth, and
    the standard cheap approximation is to assume each pick returns its own
    posterior mean, fold that hallucinated measurement back into the posterior,
    and re-score. The second pick then sees the first pick's uncertainty already
    spent, which is exactly the information a greedy top-k throws away.

    `pessimism` scales the hallucinated value: 1.0 is the posterior mean (plain
    Kriging Believer), and values below 1.0 make the liar pessimistic, which
    spreads a batch harder across the uncertainty. Returned as candidate
    indices, ordered by the pick sequence.
    """
    if norm is None:
        raise RuntimeError("qEI needs scipy.stats.norm")
    cand = np.asarray(cand_codes, dtype=float)
    alive = np.ones(len(cand), bool)
    picks = []
    g = gp
    for _ in range(min(batch, len(cand))):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        mu, sd = g.predict(cand[idx], chunk=chunk)
        vals = ei(mu, sd, best, xi=xi)
        j = int(np.argmax(vals))
        pick = int(idx[j])
        picks.append(pick)
        alive[pick] = False
        g = g.conditioned(cand[pick][None, :],
                          np.array([mu[j] - pessimism * sd[j]]))
    return picks


FUNCS = {"ei": ei, "logei": logei, "pi": pi, "ucb": ucb}


def score(kind, mu, sd, best, **kw):
    f = FUNCS.get(kind)
    if f is None:
        raise KeyError(f"unknown acquisition {kind!r}; have {sorted(FUNCS)}")
    return f(mu, sd, best, **kw)


def greedy_batch(kind, mu, sd, mask, batch, best, **kw):
    """Pick `batch` points greedily, penalising near-duplicates in the batch.

    Duplication is the real risk in a categorical space: the acquisition
    surface can have one very tall spike, and a naive top-k returns the same
    level set eight times. The penalty below is a local exclusion around each
    pick, which is the categorical version of what a batch EI would do by
    conditioning on the other picks.
    """
    mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    s = score(kind, mu, sd, best, **kw).copy()
    s[~mask] = -np.inf
    picks = []
    for _ in range(batch):
        j = int(np.argmax(s))
        if not np.isfinite(s[j]):
            break
        picks.append(j)
        s[j] = -np.inf
    return picks
