"""A surrogate that survives a table of named levels.

A one-hot linear model cannot see that two ligands are similar, and a
kernel on raw integer codes invents an order that the labels do not have.
Both are wrong in a way that shows up as *confident* nonsense in a small
campaign. The compromise used here is the standard one for categorical BO:

  kernel  =  sum_f  k_f(a_f, b_f)
  k_f     =  exp(-||onehot(a_f) - onehot(b_f)||^2 / (2 l_f^2))

i.e. each factor contributes 1 when its level matches and exp(-1/(2 l_f^2))
when it does not, and a per-factor length scale is learned. A factor whose
length scale collapses is a factor the data says does not matter -- which is
the same statement contact-sid makes about a joint that cannot excite anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _rbf_from_codes(codes: np.ndarray, cardinality, ls: np.ndarray) -> np.ndarray:
    """Squared-distance matrix in one-hot space, summed over factors."""
    n = codes.shape[0]
    d2 = np.zeros((n, n))
    for j, card in enumerate(cardinality):
        a = codes[:, j][:, None]
        b = codes[:, j][None, :]
        neq = (a != b).astype(float)          # 0 when the level matches
        d2 += neq * 2.0                       # ||e_a - e_b||^2 = 2 when a != b
    return d2


@dataclass
class MixedGP:
    """Zero-mean GP on a finite categorical space.

    Hyperparameters (per-factor length scale, signal, noise) are fitted by
    maximising the marginal likelihood with a coarse multi-start; the space is
    small enough that a grid over length scales is cheaper than gradients and
    has no chance of returning a worse optimum than its own multi-start.
    """
    cardinality: tuple
    ls: np.ndarray = None
    signal: float = 1.0
    noise: float = 1e-2
    ls_grid: tuple = (0.18, 0.35, 0.7, 1.4, 3.0)
    max_fit: int = 300
    mu: float = 0.0

    def __post_init__(self):
        self.cardinality = tuple(self.cardinality)
        if self.ls is None:
            self.ls = np.full(len(self.cardinality), 0.7)
        else:
            self.ls = np.asarray(self.ls, dtype=float)
        # fitted state
        self.X_ = None
        self.y_ = None
        self.K_inv_ = None
        self.alpha_ = None

    # ---- kernel ----------------------------------------------------------
    def _K(self, A, B, ls=None, signal=None):
        """Additive kernel over factors: k(a,b) = signal * exp(-sum_j d2_j/(2 l_j^2))."""
        ls = self.ls if ls is None else ls
        signal = self.signal if signal is None else signal
        A = np.asarray(A, dtype=int).reshape(len(A), -1)
        B = np.asarray(B, dtype=int).reshape(len(B), -1)
        acc = np.zeros((A.shape[0], B.shape[0]))
        for j in range(len(self.cardinality)):
            d2j = (A[:, j][:, None] != B[:, j][None, :]).astype(float) * 2.0
            acc += d2j / (2.0 * ls[j] ** 2)
        return signal * np.exp(-acc)

    # ---- marginal likelihood --------------------------------------------
    def _nll(self, X, y, ls, signal, noise):
        K = self._K(X, X, ls, signal) + (noise + 1e-9) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        nll = 0.5 * y @ a + np.log(np.diag(L)).sum() + 0.5 * len(X) * math.log(2 * math.pi)
        return float(nll)

    def fit(self, X, y, signal_range=(0.5, 1.0, 2.0), noise_range=(1e-2, 5e-2, 2e-1)):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) > self.max_fit:
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
        if best[1] is None:                    # numerically hopeless: fall back
            best = (np.inf, (np.full(len(self.cardinality), 0.7), 1.0, 1e-2))
        self.ls, self.signal, self.noise = best[1]
        K = self._K(X, X) + (self.noise + 1e-9) * np.eye(len(X))
        self.K_inv_ = np.linalg.inv(K)
        self.X_, self.y_ = X, yc
        self.alpha_ = self.K_inv_ @ yc
        return self

    def _ls_candidates(self):
        """Every factor at one of the grid values, plus per-factor one-at-a-time
        moves from the current setting. Full grid over 4 factors would be 5^4=625
        per (signal, noise) pair; this keeps the fit cheap and still lets a
        single irrelevant factor collapse on its own."""
        g = self.ls_grid
        base = np.full(len(self.cardinality), 0.7)
        cands = [base]
        for v in g:
            cands.append(np.full(len(self.cardinality), v))
        for j in range(len(self.cardinality)):
            for v in g:
                c = base.copy(); c[j] = v; cands.append(c)
        seen, out = set(), []
        for c in cands:
            k = tuple(np.round(c, 6))
            if k not in seen:
                seen.add(k); out.append(c)
        return out

    # ---- prediction ------------------------------------------------------
    def predict(self, Xs, return_std=True, chunk=192):
        """Prediction in row chunks.

        Not a micro-optimisation: this process runs under a hard commit ceiling
        of a few hundred MB, so a single (n_candidates x n_fit) kernel over a
        few thousand candidates is the difference between running and not
        running. Chunking changes no number, only the peak allocation.
        """
        Xs = np.asarray(Xs, dtype=int)
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

    # ---- interpretation --------------------------------------------------
    def factor_scale(self) -> dict:
        """How much each factor is allowed to matter, after the fit."""
        return {j: float(v) for j, v in enumerate(self.ls)}

    def effective_factors(self, atol=0.05):
        """Factors whose length scale did NOT collapse to the smallest grid
        value -- i.e. the data still gives them room to matter."""
        lo = min(self.ls_grid)
        return [j for j, v in enumerate(self.ls) if v > lo + atol]
