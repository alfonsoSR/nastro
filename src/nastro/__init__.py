"""
Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import scipy.cluster``.

::

 types                      --- Vector Quantization / Kmeans
 catalog                    --- Catalog data
 constants                  --- Physical and mathematical constants and units
"""

from . import types, constants, catalog

__all__ = ["types", "constants", "catalog"]
