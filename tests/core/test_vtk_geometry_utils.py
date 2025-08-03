import numpy as np
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa

from myocardial_mesh.core.vtk_geometry_utils import VTKGeometryUtils


def build_unstructured_grid(points: np.ndarray) -> vtk.vtkUnstructuredGrid:
    """
    Helper: create a vtkUnstructuredGrid with one VTK_VERTEX cell per point.
    """
    vtk_pts = vtk.vtkPoints()
    for p in points:
        vtk_pts.InsertNextPoint(*p)

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(vtk_pts)

    # Insert a vertex cell for each point
    for i in range(vtk_pts.GetNumberOfPoints()):
        ug.InsertNextCell(vtk.VTK_VERTEX, 1, [i])

    return ug


class FakeProbeFilter:
    """
    A stand-in for vtk.vtkProbeFilter that bypasses actual VTK internals.
    It simply returns the source mesh so that its 'activation' field
    can be read back unmodified.
    """

    def __init__(self):
        self._source = None
        self._input = None

    def SetSourceData(self, mesh):
        self._source = mesh

    def SetInputData(self, poly):
        self._input = poly

    def Update(self):
        # No-op: we won't actually probe
        pass

    def GetOutput(self):
        # Return the original source mesh which we assume has 'activation'
        return self._source


def test_probe_activation_single_and_multiple_points(monkeypatch):
    # Build a triangular mesh with an 'activation' field
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ug = build_unstructured_grid(pts)

    # Attach activation values [10, 20, 30]
    dd = dsa.WrapDataObject(ug)
    activations = np.array([10.0, 20.0, 30.0])
    dd.PointData.append(activations, "activation")

    # Monkey-patch vtkProbeFilter so no real VTK call is made
    monkeypatch.setattr(vtk, "vtkProbeFilter", lambda: FakeProbeFilter())

    # Now run probe_activation: it will just return the source mesh's activations
    # when querying at any point.
    query_points = np.array(
        [
            [0.0, 0.0, 0.0],  # exactly point 0
            [1.0, 0.0, 0.0],  # exactly point 1
            [0.5, 0.5, 0.0],  # in between 1 and 2
        ]
    )
    result = VTKGeometryUtils.probe_activation(ug, query_points)

    # Since FakeProbeFilter.GetOutput() returned the mesh itself,
    # probe_activation will just read the original activations in order.
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # For our stub, we expect result == activations
    assert np.allclose(result, activations)


def test_find_closest_pmjs_empty_array():
    """find_closest_pmjs should return an empty array when given no PMJs."""
    pts = np.array([[0, 0, 0], [1, 0, 0]])
    ug = build_unstructured_grid(pts)
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(ug)
    locator.BuildLocator()

    pmjs = np.zeros((0, 3))
    projected = VTKGeometryUtils.find_closest_pmjs(pmjs, locator)
    assert isinstance(projected, np.ndarray)
    # Should be empty with same shape
    assert projected.shape == (0, 3)


def test_find_closest_pmjs_identity():
    """If PMJs exactly equal mesh points, projection is a no-op."""
    pts = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    ug = build_unstructured_grid(pts)
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(ug)
    locator.BuildLocator()

    # PMJs exactly the same as pts
    pmjs = pts.copy()
    projected = VTKGeometryUtils.find_closest_pmjs(pmjs, locator)
    assert np.allclose(projected, pts)


def test_probe_activation_empty_and_missing_field():
    """Cover both the empty‐query and missing‐activation cases."""
    # 1) Empty query → empty result
    pts = np.array([[0, 0, 0]])
    ug = build_unstructured_grid(pts)
    dd = dsa.WrapDataObject(ug)
    dd.PointData.append(np.array([42.0]), "activation")

    empty = np.zeros((0, 3))
    sampled = VTKGeometryUtils.probe_activation(ug, empty)
    assert isinstance(sampled, np.ndarray)
    assert sampled.shape == (0,)

    # 2) Missing activation field → returns a single value (scalar or length-1 array)
    ug2 = build_unstructured_grid(pts)
    query = np.array([[0, 0, 0]])  # one point
    sampled2 = VTKGeometryUtils.probe_activation(ug2, query)
    # It may return a scalar or 1-element array
    assert isinstance(sampled2, np.ndarray)
    # Either shape == () (scalar) or shape == (1,)
    assert sampled2.size == 1
    val = sampled2.flatten()[0]
    # Value is either 0 or NaN (VTK default behavior for missing field)
    assert val == 0 or np.isnan(val)
