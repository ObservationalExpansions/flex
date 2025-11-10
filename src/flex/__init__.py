"""
flex

Fourier-Laguerre basis function expansions

See README file and online documentation (https://flexable.readthedocs.io)
for further details and usage instructions.
"""
from .flexbase import FLEX
from importlib.metadata import version

__version__ = version("flex")
__all__ = ["FLEX"]
