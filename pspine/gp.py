"""A surrogate that survives a table of named levels *and* a table of settings.

Two kinds of factor, two kinds of distance, and the difference is not cosmetic:

  named level (ligand A/B/C)   a one-hot linear model cannot see that two
                              ligands are similar, and a kernel on raw integer
                              codes invents an order the labels do not have.
                              Both are wrong in a way that shows up as
                              *confident* nonsense in a small campaign. So a
                              named factor contributes 1 when the level matches
                              and exp(-1/l^2) when it does not -- one half of the
                              squared one-hot distance, which is 2 on a mismatch.

  continuous setting (24 C)    binning it into levels throws away the metric,
                              which is the wrong move on the continuous process
                              tables: a power plant's output moves smoothly in
                              ambient temperature, and a binner destroys the
                              gradient the surrogate needs. So a continuous
                              factor contributes the squared exponential on its
                              scaled coordinate.

  kernel(a,b) = signal * exp( - sum_f d2_f(a_f, b_f) / l_f^2 )

A per-factor length scale is fitted by marginal likelihood. A factor whose
length scale collapses is a factor the data says does not matter -- the same
statement contact-sid makes about a joint that cannot excite anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def weighted_sq_dist(A, B, continuous, spacing, ls):
    """sum_f d2_f / l_f^2 -- (n_A, n_B).

    A continuous factor's coordinate is first divided by its own span (spacing
    carries 1/(n-1) per grid step), so its length scale is in *fractions of the
    range* and means the same thing on a factor measured in Celsius as on one
    measured in kg/m^3. Without that, the same length scale would be
    "irrelevant" for a 1-1000 range and "very relevant" for a 0-1 range, and the
    factor ranking the model reports would be an artefact of units.
    """
    A = np.asarray(A, dtype=float).reshape(len(A), -1)
    B = np.asarray(B, dtype=float).reshape(len(B), -1)
    ls = np.asarray(ls, dtype=float)
    out = np.zeros((A.shape[0], B.shape[0]))
    for j, cont in enumerate(continuous):
        a = A[:, j][:, None]
        b = B[:, j][None, :]
        if cont:
            # half the squared gap, so the kernel is the usual exp(-d^2/l^2) with
            # the RBF's factor of 2 absorbed into the length scale
            d2 = 0.5 * ((a - b) * spacing[j]) ** 2
        else:
            # one-hot distance squared is 2 for a mismatch, so /2 leaves 1 --
            # exactly the "same -> 1, different -> exp(-1/l^2)" rule the module
            # docstring states. Using 2 here would make the docstring a lie and
            # the length-scale grid mean something else than what it says.
            d2 = (a != b).astype(float)
        out += d2 / (ls[j] ** 2)
    return out


@dataclass
class MixedGP:
    """Zero-mean GP over a finite space of named levels and/or real settings.

    Hyperparameters are fitted by maximising the marginal likelihood with a
    coarse multi-start: the candidate space is small enough that a grid over
    length scales is cheaper than gradients and cannot return a worse optimum
    than its own multi-start.
    """
    cardinality: tuple
    continuous: tuple = None
    spacing: tuple = None
    ls: np.ndarray = None
    signal: float = 1.0
    noise: float = 1e-2
    ls_grid: tuple = (0.18, 0.35, 0.7, 1.4, 3.0)
    max_fit: int = 300
    mu: float = 0.0

    def __post_init__(self):
        self.cardinality = tuple(self.cardinality)
        d = len(self.cardinality)
        self.continuous = ((False,) * d if self.continuous is None
                           else tuple(bool(x) for x in self.continuous))
        if len(self.continuous) != d:
            raise ValueError("continuous flags must match the factor count")
        if self.spacing is None:
            self.spacing = tuple(0.0 if not c else 1.0 / (n - 1)
                                 for c, n in zip(self.continuous, self.cardinality))
        else:
            self.spacing = tuple(float(x) for x in self.spacing)
            if len(self.spacing) != d:
                raise ValueError("spacing must match the factor count")
        if self.ls is None:
            self.ls = np.full(d, 0.7)
        else:
            self.ls = np.asarray(self.ls, dtype=float)
        self.X_ = None
        self.y_ = None
        self.K_inv_ = None
        self.alpha_ = None

    @classmethod
    def for_space(cls, space, **kw):
        """A surrogate whose factor kinds match a DesignSpace's."""
        return cls(cardinality=space.cardinality,
                   continuous=tuple(f.continuous for f in space.factors),
                   spacing=space.spacing, **kw)

    # ---- kernel ----------------------------------------------------------
    def _K(self, A, B, ls=None, signal=None):
        ls = self.ls if ls is None else ls
        signal = self.signal if signal is None else signal
        return signal * np.exp(-weighted_sq_dist(A, B, self.continuous, self.spacing, ls))

    # ---- marginal likelihood --------------------------------------------
    def _nll(self, X, y, ls, signal, noise):
        K = self._K(X, X, ls, signal) + (noise + 1e-9) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return float(0.5 * y @ a + np.log(np.diag(L)).sum() + 0.5 * len(X) * math.log(2 * math.pi))

    def _ls_candidates(self):
        """Every factor at one grid value, plus a per-factor move away from the
        default. A full grid over 10 factors would be 5^10; this stays linear in
        the factor count and still lets one irrelevant factor collapse alone."""
        g = self.ls_grid
        d = len(self.cardinality)
        base = np.full(d, 0.7)
        cands = [base]
        for v in g:
            cands.append(np.full(d, v))
        for j in range(d):
            for v in g:
                c = base.copy(); c[j] = v; cands.append(c)
        seen, out = set(), []
        for c in cands:
            k = tuple(np.round(c, 6))
            if k not in seen:
                seen.add(k); out.append(c)
        return out

    def fit(self, X, y, signal_range=(0.5, 1.0, 2.0), noise_range=(1e-2, 5e-2, 2e-1)):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) > self.max_fit:
            # A random subset, because the marginal likelihood of 4599 rows is a
            # 4599^2 matrix and this process lives under a hard commit ceiling.
            # Declared, not hidden: the fit sees at most `max_fit` cells.
            rng = np.random.default_rng(0)
            keep = rng.choice(len(X), self.max_fit, replace=False)
            X, y = X[keep], y[keep]
        self.mu = float(y.mean())
        yc = y - self.mu
        best = (np.inf, None)
        for ls in self._ls_candidates():
            for signal in signal_range:
                for noise in noise_range:
                    v = self._nll(X, yc, np.asarray(ls), signal, noise)
                    if v < best[0]:
                        best = (v, (np.asarray(ls, dtype=float), float(signal), float(noise)))
        if best[1] is None:
            best = (np.inf, (np.full(len(self.cardinality), 0.7), 1.0, 1e-2))
        self.ls, self.signal, self.noise = best[1]
        K = self._K(X, X) + (self.noise + 1e-9) * np.eye(len(X))
        self.K_inv_ = np.linalg.inv(K)
        self.X_, self.y_ = X, yc
        self.alpha_ = self.K_inv_ @ yc
        return self

    # ---- prediction ------------------------------------------------------
    def predict(self, Xs, return_std=True, chunk=192):
        """Prediction in row chunks, which changes no number, only the peak
        allocation: a single (n_candidates x n_fit) kernel over a few thousand
        candidates is the difference between running and not running here."""
        Xs = np.asarray(Xs, dtype=float)
        if Xs.ndim == 1:
            Xs = Xs.reshape(-1, 1)
        if self.X_ is None:
            m = np.full(len(Xs), self.mu)
            s = np.full(len(Xs), math.sqrt(self.signal))
            return (m, s) if return_std else m
        ms, vs = [], []
        for a in range(0, len(Xs), chunk):
            Xc = Xs[a:a + chunk]
            Ks = self._K(Xc, self.X_)
            ms.append(self.mu + Ks @ self.alpha_)
            if return_std:
                vs.append(self.signal - np.einsum("ij,jk,ik->i", Ks, self.K_inv_, Ks))
        m = np.concatenate(ms)
        if not return_std:
            return m
        return m, np.sqrt(np.clip(np.concatenate(vs), 1e-12, None))

    # ---- conditioning (for batch acquisition) ----------------------------
    def conditioned(self, Xnew, ynew):
        """A copy of this GP with `Xnew` added as observed, hyperparameters held.

        This is what a batch acquisition needs: q-EI scores a *set*, which means
        re-asking "what is the posterior mean and width here" after pretending
        the already-chosen points have been measured. Re-running the
        hyperparameter search for every candidate pick would cost 45 marginal
        likelihoods per pick; the block inverse below costs one small solve.
        Held hyperparameters are also the honest choice -- the surrogate has seen
        no new data, only a hypothesis about it.
        """
        import copy as _copy
        g = _copy.copy(self)
        Xnew = np.asarray(Xnew, dtype=float)
        Xnew = Xnew.reshape(1, -1) if Xnew.ndim == 1 else Xnew
        ynew = np.atleast_1d(np.asarray(ynew, dtype=float)) - self.mu
        B = self._K(Xnew, self.X_)                          # (m, n)
        D = self._K(Xnew, Xnew) + (self.noise + 1e-9) * np.eye(len(Xnew))
        Ainv = self.K_inv_
        S = D - B @ Ainv @ B.T
        S = S + 1e-10 * np.eye(len(Xnew))
        Sinv = np.linalg.inv(S)
        AinvB = Ainv @ B.T
        top = Ainv + AinvB @ Sinv @ AinvB.T
        g.K_inv_ = np.block([[top, -AinvB @ Sinv], [-Sinv @ AinvB.T, Sinv]])
        g.X_ = np.vstack([self.X_, Xnew])
        g.y_ = np.concatenate([self.y_, ynew])
        g.alpha_ = g.K_inv_ @ g.y_
        return g

    # ---- interpretation --------------------------------------------------
    def factor_scale(self) -> dict:
        return {j: float(v) for j, v in enumerate(self.ls)}

    def effective_factors(self, atol=0.05):
        lo = min(self.ls_grid)
        return [j for j, v in enumerate(self.ls) if v > lo + atol]
