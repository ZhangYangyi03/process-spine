"""process-spine: Bayesian experimental design over *measured* process tables.

The fuel is the point. The acquisition functions here are standard; what is
unusual is that every objective evaluation is a row of a table somebody ran in
a lab, so no simulator stands between the optimiser and the thing being
optimised.
"""
__version__ = "0.1.0"

from .space import DesignSpace, Factor
from .gp import MixedGP

__all__ = ["DesignSpace", "Factor", "MixedGP", "__version__"]
