# tests/myocardial_mesh/test_activation.py
import numpy as np
import pytest
import vtk
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkTetra
from vtkmodules.numpy_interface import dataset_adapter as dsa

from myocardial_mesh.core.activation import (
    legacy_seed_projection,
    fim_solve_from_seeds,
    sample_activation_at_points,
)
from fimpy.solver import FIMPY


def _make_single_tet():
    # Simple tetra at origin
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=int)  # one tetra

    # VTK grid
    ug = vtkUnstructuredGrid()
    vtk_pts = vtk.vtkPoints()
    for p in points:
        vtk_pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    ug.SetPoints(vtk_pts)
    ug.Allocate(1, 1)
    tet = vtkTetra()
    for i, idx in enumerate(cells[0]):
        tet.GetPointIds().SetId(i, int(idx))
    ug.InsertNextCell(tet.GetCellType(), tet.GetPointIds())

    # Locator
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(ug)
    loc.BuildLocator()
    return ug, points, cells, loc


def test_legacy_seed_projection_sets_four_nodes():
    ug, xyz, cells, loc = _make_single_tet()

    # isotropic D per cell
    D = np.repeat(np.eye(3)[None, :, :], cells.shape[0], axis=0)

    # seed at node 0, time=5
    x0 = xyz[[0], :]
    x0_vals = np.array([5.0])

    act_init = legacy_seed_projection(
        x0=x0, x0_vals=x0_vals, vtk_locator=loc, cells=cells, D_cell=D, xyz=xyz
    )
    # node 0 exactly 5, others >= 5
    assert np.isclose(act_init[0], 5.0)
    assert np.all(act_init[cells[0]] >= 5.0)


def test_fim_solve_from_seeds_roundtrip_on_full_seeds():
    # single-tet mesh helper is assumed to exist in this file
    # and returns: (mesh, xyz, cells, something_else)
    _, xyz, cells, _ = _make_single_tet()

    # isotropic D per cell (shape: n_cells x 3 x 3)
    D = np.repeat(np.eye(3, dtype=float)[None, :, :], cells.shape[0], axis=0)

    # FIM solver
    fim = FIMPY.create_fim_solver(xyz, cells, D, device="cpu")

    # fully seeded field
    act_init = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    # run via the convenience wrapper under test
    sol = fim_solve_from_seeds(act_init=act_init, fim_solver=fim)

    assert sol.shape == act_init.shape

    # FIM solves the lower envelope: it will not increase any seeded time
    assert np.all(sol <= act_init + 1e-12)

    # the global minimum seed is preserved exactly
    i0 = int(np.argmin(act_init))
    np.testing.assert_allclose(sol[i0], act_init[i0], rtol=0.0, atol=1e-12)

    # sanity: min(sol) equals min(act_init)
    assert float(sol.min()) == pytest.approx(float(act_init.min()), abs=1e-12)


def test_sample_activation_at_points_returns_values():
    ug, xyz, _, _ = _make_single_tet()
    # Add activation field
    act = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    dd = dsa.WrapDataObject(ug)
    dd.PointData.append(act, "activation")

    # sample exactly at node 2
    vals = sample_activation_at_points(vtk_mesh=ug, points_xyz=xyz[[2], :])
    assert vals.shape == (1,)
    assert np.isclose(vals[0], 30.0)
