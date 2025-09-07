from pathlib import Path
import numpy as np
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa

from myocardial_mesh.io.mesh_io import load_unstructured, build_cell_locator

DATA = Path("data/crtdemo")


def test_load_unstructured_vtu_and_vtk():
    vtu = DATA / "nb" / "True_endo.vtu"
    vtkm = DATA / "crtdemo_mesh_oriented.vtk"
    ds1 = load_unstructured(str(vtu))
    ds2 = load_unstructured(str(vtkm))
    assert ds1.GetNumberOfPoints() > 0
    assert ds2.GetNumberOfPoints() > 0


def test_build_cell_locator_closest_point_identity():
    ds = load_unstructured(str(DATA / "crtdemo_mesh_oriented.vtk"))
    loc = build_cell_locator(ds)

    dd = dsa.WrapDataObject(ds)
    p = np.asarray(dd.Points[0], float)

    out = [0.0, 0.0, 0.0]
    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    dist = vtk.reference(0.0)

    loc.FindClosestPoint(p, out, cellId, subId, dist)
    assert np.linalg.norm(np.array(out) - p) < 1e-8
