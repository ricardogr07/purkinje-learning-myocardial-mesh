"""Finite element assembly helpers."""

from __future__ import annotations

import numpy as np
from scipy import sparse

# Re-use the exact math you already trust
from .geometry_utils import Bmatrix, localStiffnessMatrix

__all__ = ["assemble_K", "Bmatrix", "localStiffnessMatrix"]


def assemble_K(pts: np.ndarray, elm: np.ndarray, Gi: np.ndarray) -> sparse.csr_matrix:
    """Assemble the global stiffness/Laplacian matrix (legacy behavior).

    Args:
        pts: Node coordinates with shape (N, 3).
        elm: Tetrahedral connectivity with shape (T, 4).
        Gi: Intracellular conductivity tensor per cell, shape (T, 3, 3).

    Returns:
        scipy.sparse.csr_matrix: Global stiffness matrix with shape (N, N).
    """
    B, J = Bmatrix(pts, elm)
    Kloc = localStiffnessMatrix(B, J, Gi)  # (T,4,4)
    N = pts.shape[0]
    Kvals = Kloc.ravel("C")
    II = np.repeat(elm, 4, axis=1).ravel()
    JJ = np.tile(elm, 4).ravel()
    return sparse.coo_matrix((Kvals, (II, JJ)), shape=(N, N)).tocsr()
