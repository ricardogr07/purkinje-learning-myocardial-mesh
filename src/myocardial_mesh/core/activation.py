from __future__ import annotations
import numpy as np
import vtk
import pyvista as pv


def legacy_seed_projection(
    *,
    x0: np.ndarray,  # (M,3) seed coords (PMJs/PVCs)
    x0_vals: np.ndarray,  # (M,)   seed times (ms)
    vtk_locator: vtk.vtkCellLocator,
    cells: np.ndarray,  # (T,4)  tetra indices
    D_cell: np.ndarray,  # (T,3,3) conductivity metric per cell
    xyz: np.ndarray,  # (N,3) node coords
) -> np.ndarray:
    """
    Legacy per-PMJ projection used in the notebook/legacy code:
    - For each seed point, find closest cell.
    - Analytic update of its 4 nodes: t_seed + sqrt(v^T G^{-1} v).
    - Return 'act_init' (N,) with inf except updated nodes (min-reduction).
    """
    N = xyz.shape[0]
    act = np.full(N, np.inf, dtype=float)

    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    dist = vtk.reference(0.0)
    x_proj = [0.0, 0.0, 0.0]

    for k in range(x0.shape[0]):
        x_seed = x0[k, :]
        vtk_locator.FindClosestPoint(x_seed, x_proj, cellId, subId, dist)
        cid = int(cellId)
        Ginv = np.linalg.inv(D_cell[cid, ...])
        node_ids = cells[cid, :]
        v = xyz[node_ids, :] - x_seed[None, :]
        # sqrt(v^T Ginv v)
        inc = np.sqrt(np.einsum('ij,kj,ki->k', Ginv, v, v))
        new_vals = x0_vals[k] + inc
        act[node_ids] = np.minimum(act[node_ids], new_vals)
    return act


def fim_solve_from_seeds(
    *,
    act_init: np.ndarray,  # (N,)
    fim_solver,  # fimpy solver with .comp_fim()
) -> np.ndarray:
    """
    Run FIM from the boolean mask + seed values encoded in act_init.
    """
    x0_mask = np.isfinite(act_init)
    seed_vals = act_init[x0_mask]
    sol = fim_solver.comp_fim(x0_mask, seed_vals)
    try:
        # GPU returns cupy
        sol = sol.get()
    except AttributeError:
        pass
    return np.asarray(sol, dtype=float)


def sample_activation_at_points(
    *,
    vtk_mesh,  # VTK UnstructuredGrid
    points_xyz: np.ndarray,  # (M,3)
) -> np.ndarray:
    """
    Sample PointData['activation'] of vtk_mesh at given coordinates using PyVista.
    """
    x0_pv = pv.PolyData(points_xyz.copy())
    result = x0_pv.sample(
        pv.UnstructuredGrid(vtk_mesh), tolerance=1e-6, snap_to_closest_point=True
    )
    mask = np.asarray(result['vtkValidPointMask'])
    if np.any(mask == 0):
        raise RuntimeError("Invalid samples while probing activation at given points.")
    return np.asarray(result['activation'], dtype=float)
