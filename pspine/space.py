"""Categorical design spaces, and what "a candidate" means in them.

A process experiment has two kinds of knob: the ones you *set* (base, ligand,
temperature) and the ones you *read* (yield). This module is only about the
first kind: a finite, enumerable set of named levels per factor, which is what
a plate of reactions actually is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Factor:
    name: str
    levels: tuple

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
