import pytest
import numpy as np
import vtk
from types import SimpleNamespace
from vtkmodules.numpy_interface import dataset_adapter as dsa

import myocardial_mesh.myocardial_mesh as mm_mod
from myocardial_mesh.io.mesh_io import MeshIO


def _tiny_tetra_mesh():
    pts = vtk.vtkPoints()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        pts.InsertNextPoint(*p)
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(pts)
    ids = vtk.vtkIdList()
    [ids.InsertNextId(i) for i in range(4)]
    ug.InsertNextCell(vtk.VTK_TETRA, ids)
    return ug


def _wrap(mesh):
    arr_pts = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
    arr_cells = np.array([4, 0, 1, 2, 3])

    class _PD(SimpleNamespace):
        def __getitem__(self, key):  # enables square-bracket access
            return getattr(self, key)

    dd = SimpleNamespace(Points=arr_pts, Cells=arr_cells)
    dd.PointData = _PD(append=lambda arr, name: setattr(dd.PointData, name, arr))
    return dd


class DummySolver:
    """Replaces LeadFieldSolver; its methods can be monkey-patched per test."""

    def __init__(self, *, mesh, electrode_pos=None, lead_fields_dict=None):
        self.mesh = mesh
        self.electrode_pos = electrode_pos
        self.lead_fields_dict = lead_fields_dict

    def get_ecg(self, **kwargs):
        return "ecg"

    def get_aux_Vl(self):
        return "aux"


@pytest.fixture(autouse=True)
def patch_myocardial_mesh(monkeypatch):
    mesh = _tiny_tetra_mesh()
    monkeypatch.setattr(MeshIO, "read_mesh", lambda _: mesh)
    monkeypatch.setattr(dsa, "WrapDataObject", _wrap)
    monkeypatch.setattr(mm_mod.MyocardialMesh, "_build_locator", lambda s, m: None)
    monkeypatch.setattr(mm_mod, "LeadFieldSolver", DummySolver)
    yield
