from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Scalars
Float: TypeAlias = np.float64
Int: TypeAlias = np.int64

# Common arrays
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

# Mesh-specific conveniences
Points3D: TypeAlias = NDArray[np.float64]  # shape (N, 3)
TetCells: TypeAlias = NDArray[np.int64]  # shape (M, 4) - indices
TriFaces: TypeAlias = NDArray[np.int64]  # shape (K, 3)
IndexArray: TypeAlias = NDArray[np.int64]  # shape (N,)

# Time series / signals
Signal: TypeAlias = NDArray[np.float64]  # shape (T,) or (T, C)

# Matrices
MatrixF: TypeAlias = NDArray[np.float64]
MatrixI: TypeAlias = NDArray[np.int64]
