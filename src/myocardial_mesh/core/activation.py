"""Activation utilities for legacy seeding, FIM solves, and sampling.

This module provides:
* Legacy per-seed (PMJ/PVC) projection onto the nearest tetra cell and analytic
  node updates (notebook-compatible behavior).
* A thin wrapper to run a Fast Iterative Method (FIM) solver from seed values.
* Probing of activation values at arbitrary coordinates using PyVista.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pyvista as pv
import vtk
from numpy.typing import NDArray

_LOGGER = logging.getLogger(__name__)


def legacy_seed_projection(
    *,
    x0: NDArray[np.float64],  # (M, 3) seed coords (PMJs/PVCs)
    x0_vals: NDArray[np.float64],  # (M,)   seed times (ms)
    vtk_locator: vtk.vtkCellLocator,
    cells: NDArray[np.int64],  # (T, 4) tetra indices
    D_cell: NDArray[np.float64],  # (T, 3, 3) conductivity metric per cell
    xyz: NDArray[np.float64],  # (N, 3) node coords
) -> NDArray[np.float64]:
    """Project seeds onto nearest cells and update the four nodes analytically.

    This follows the legacy/notebook approach:
      1) For each seed point, find the closest tetrahedral cell.
      2) Analytically update the four nodes of that cell using
         ``t_seed + sqrt(v^T G^{-1} v)`` with ``G = D_cell``.
      3) Return an activation array with ``inf`` everywhere except the updated
         nodes, using a min-reduction for multiple updates to the same node.

    Args:
        x0: Seed coordinates of shape ``(M, 3)``.
        x0_vals: Seed activation times (ms) of shape ``(M,)``.
        vtk_locator: Prebuilt VTK cell locator for closest-point queries.
        cells: Tetrahedral connectivity array of shape ``(T, 4)`` (node indices).
        D_cell: Per-cell conductivity metrics of shape ``(T, 3, 3)``.
        xyz: Node coordinates of shape ``(N, 3)``.

    Returns:
        NDArray[np.float64]: Activation array of shape ``(N,)`` with updated
        node values (ms) and ``inf`` elsewhere.
    """
    _LOGGER.info("Legacy seed projection: %d seeds", int(x0.shape[0]))
    N = int(xyz.shape[0])
    act: NDArray[np.float64] = np.full(N, np.inf, dtype=float)

    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    dist = vtk.reference(0.0)
    x_proj = [0.0, 0.0, 0.0]

    for k in range(int(x0.shape[0])):
        x_seed = x0[k, :]
        vtk_locator.FindClosestPoint(x_seed, x_proj, cellId, subId, dist)
        cid = int(cellId)
        Ginv = np.linalg.inv(D_cell[cid, ...])
        node_ids = cells[cid, :]
        v = xyz[node_ids, :] - x_seed[None, :]
        # sqrt(v^T Ginv v)
        inc = np.sqrt(np.einsum("ij,kj,ki->k", Ginv, v, v))
        new_vals = x0_vals[k] + inc
        act[node_ids] = np.minimum(act[node_ids], new_vals)

    _LOGGER.debug("Legacy seed projection complete.")
    return act


def fim_solve_from_seeds(
    *,
    act_init: NDArray[np.float64],  # (N,)
    fim_solver: Any,  # object with .comp_fim(mask, values)
) -> NDArray[np.float64]:
    """Run a FIM solve given an initialization vector with seed values.

    The input encodes seeds by placing finite values at seed nodes and ``inf``
    elsewhere. A boolean mask is derived internally and, together with the
    seed values, passed to the provided FIM solver.

    Args:
        act_init: Activation initialization of shape ``(N,)``; finite entries
            indicate seeds (their value is the seed time).
        fim_solver: Object exposing ``comp_fim(mask, seed_values)``. It may
            return a NumPy array or a CuPy array (in which case ``.get()`` is
            attempted).

    Returns:
        NDArray[np.float64]: Activation solution of shape ``(N,)`` in ms.
    """
    _LOGGER.info("FIM solve from seeds.")
    x0_mask = np.isfinite(act_init)
    seed_vals = act_init[x0_mask]
    sol = fim_solver.comp_fim(x0_mask, seed_vals)
    try:
        # GPU returns cupy arrays; bring back to host if available.
        sol = sol.get()
    except AttributeError:
        pass
    result = np.asarray(sol, dtype=float)
    _LOGGER.debug("FIM solve complete.")
    return result


def sample_activation_at_points(
    *,
    vtk_mesh: vtk.vtkUnstructuredGrid,  # VTK UnstructuredGrid
    points_xyz: NDArray[np.float64],  # (M, 3)
) -> NDArray[np.float64]:
    """Sample ``PointData['activation']`` of a VTK mesh at given coordinates.

    Uses PyVista's probing to sample the activation field at the provided
    coordinates. A small tolerance and snapping to nearest points are enabled.

    Args:
        vtk_mesh: Unstructured VTK mesh carrying ``PointData['activation']``.
        points_xyz: Query coordinates of shape ``(M, 3)``.

    Returns:
        NDArray[np.float64]: Activation values at the query points, shape
        ``(M,)`` in ms.

    Raises:
        RuntimeError: If any query point cannot be validly probed.
    """
    _LOGGER.info("Sampling activation at %d points.", int(points_xyz.shape[0]))
    x0_pv = pv.PolyData(cast(Any, points_xyz.copy()))
    result = x0_pv.sample(
        pv.UnstructuredGrid(vtk_mesh), tolerance=1e-6, snap_to_closest_point=True
    )
    mask = np.asarray(result["vtkValidPointMask"])
    if np.any(mask == 0):
        _LOGGER.error("Invalid samples while probing activation at given points.")
        raise RuntimeError("Invalid samples while probing activation at given points.")
    values = np.asarray(result["activation"], dtype=float)
    _LOGGER.debug("Sampling complete.")
    return values
