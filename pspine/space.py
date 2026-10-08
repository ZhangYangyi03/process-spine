"""Categorical design spaces, and what "a candidate" means in them.

A process experiment has two kinds of knob: the ones you *set* (base, ligand,
temperature) and the ones you *read* (yield). This module is only about the
first kind: a finite, enumerable set of named levels per factor, which is what
a plate of reactions actually is.
"""
from __future__ import annotations

import json

import numpy as np
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Factor:
    """One knob. Either a set of named levels (ligand A/B/C) or a continuous
    range the experimenter can set to any value (ambient temperature).

    The distinction is not cosmetic. A named level carries no order, so the
    kernel must treat two levels as either equal or different. A continuous
    factor carries *metric* information -- 24 C is close to 25 C and far from
    35 C -- and binning it into 5 levels throws that away, which is exactly the
    wrong move on the continuous process tables (a power plant's output changes
    smoothly in ambient temperature, and a binner destroys the gradient.
    """
    name: str
    levels: tuple
    lo: float = None
    hi: float = None

    @property
    def continuous(self) -> bool:
        return self.lo is not None

    @property
    def value_of(self) -> np.ndarray:
        """The real value of each internal code: the levels for a categorical
        factor, an even grid across [lo, hi] for a continuous one."""
        if not self.continuous:
            return np.asarray(self.levels, dtype=object)
        return np.linspace(self.lo, self.hi, len(self.levels))

    @staticmethod
    def continuous_of(name: str, lo: float, hi: float, n: int = 24) -> "Factor":
        """A continuous factor, enumerable to `n` grid points.

        `n` is a *candidate resolution*, not a claim that the experimenter is
        restricted to 24 settings: it sets the spacing the campaign proposes at
        and the resolution the kernel's length scale is reported in. The
        candidates are enumerated only because every policy here scores the
        whole space exactly instead of solving an inner optimisation, which is
        what makes the comparison between arms honest.
        """
        if not (hi > lo):
            raise ValueError(f"continuous factor {name!r} needs hi > lo")
        if n < 3:
            raise ValueError("a continuous factor needs >=3 grid points")
        vals = np.linspace(lo, hi, n)
        return Factor(name, tuple(float(v) for v in vals), lo=float(lo), hi=float(hi))

    @staticmethod
    def of(name: str, levels: Iterable) -> "Factor":
        lv = tuple(levels)
        if len(lv) < 2:
            raise ValueError(f"factor {name!r} needs >=2 levels, got {len(lv)}")
        if len(set(lv)) != len(lv):
            raise ValueError(f"factor {name!r} has duplicate levels")
        return Factor(name, lv)

    def index(self, level) -> int:
        try:
            return self.levels.index(level)
        except ValueError:
            raise KeyError(f"{level!r} is not a level of {self.name!r}") from None


@dataclass
class DesignSpace:
    """A finite categorical design space, with a label-vs-code mapping.

    Levels are kept as the original labels (a ligand name, an additive name) so
    a recommendation can be read by a chemist without a lookup table. Internally
    everything is an integer code, which is all the surrogate ever needs.
    """
    factors: tuple
    _code_of: list = field(default_factory=list, repr=False)

    @staticmethod
    def of(factors: Sequence) -> "DesignSpace":
        fs = tuple(factors)
        if not fs:
            raise ValueError("a design space needs at least one factor")
        d = DesignSpace(fs)
        d._reindex()
        return d

    def _reindex(self):
        self._code_of = []
        for f in self.factors:
            self._code_of.append({lv: i for i, lv in enumerate(f.levels)})

    # ---- shape -----------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.factors)

    @property
    def size(self) -> int:
        n = 1
        for f in self.factors:
            n *= len(f.levels)
        return n

    @property
    def cardinality(self) -> tuple:
        return tuple(len(f.levels) for f in self.factors)

    @property
    def spacing(self) -> tuple:
        """Normalised grid step per factor, or 0.0 for a categorical one.

        For a continuous factor with n grid points spanning [lo, hi], one step
        is (hi-lo)/(n-1); dividing by the span leaves 1/(n-1), which is all the
        kernel needs to compute a squared distance in the *scaled* continuous
        coordinate. Categorical factors get 0.0 and are handled by the
        match/mismatch rule instead.
        """
        out = []
        for f in self.factors:
            n = len(f.levels)
            out.append(0.0 if not f.continuous else 1.0 / (n - 1))
        return tuple(out)

    @property
    def n_continuous(self) -> int:
        return sum(1 for f in self.factors if f.continuous)

    def values(self, code) -> dict:
        """Internal code -> the real value a person would set on the process."""
        return {f.name: (float(f.value_of[c]) if f.continuous else f.levels[c])
                for f, c in zip(self.factors, code)}

    def encode(self, point) -> tuple:
        """{"ligand": "XPhos", ...} or ("XPhos", ...) -> integer codes."""
        if isinstance(point, dict):
            vals = [point[f.name] for f in self.factors]
        else:
            vals = list(point)
        if len(vals) != self.dim:
            raise ValueError(f"expected {self.dim} values, got {len(vals)}")
        return tuple(self._code_of[i][v] for i, v in enumerate(vals))

    def decode(self, code) -> dict:
        return {f.name: f.levels[c] for f, c in zip(self.factors, code)}

    def point_for(self, code) -> tuple:
        return tuple(f.levels[c] for f, c in zip(self.factors, code))

    def all_codes(self):
        return list(product(*[range(len(f.levels)) for f in self.factors]))

    def __len__(self):
        return self.size

    def __repr__(self):
        inner = " x ".join(f"{f.name}({len(f.levels)})" for f in self.factors)
        return f"<DesignSpace {inner} = {self.size} points>"

    # ---- (de)serialisation ----------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"factors": [{"name": f.name, "levels": list(f.levels)}
                                       for f in self.factors]}, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(s: str) -> "DesignSpace":
        d = json.loads(s)
        return DesignSpace.of([Factor.of(f["name"], f["levels"]) for f in d["factors"]])
