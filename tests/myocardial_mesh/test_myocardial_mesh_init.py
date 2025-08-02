# tests/test_myocardial_mesh_base.py
import numpy as np
import pickle
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
from types import SimpleNamespace

import myocardial_mesh.myocardial_mesh as mm_mod
from myocardial_mesh import MyocardialMesh
from myocardial_mesh.io.mesh_io import MeshIO


def make_tetra_mesh():
    pts = vtk.vtkPoints()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        pts.InsertNextPoint(*p)
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(pts)
    ids = vtk.vtkIdList()
    for i in range(4):
        ids.InsertNextId(i)
    ug.InsertNextCell(vtk.VTK_TETRA, ids)
    return ug


def patch_read_and_wrap(monkeypatch, mesh):
    """Patch MeshIO.read_mesh and dsa.WrapDataObject to use *mesh*."""
    monkeypatch.setattr(MeshIO, "read_mesh", lambda _: mesh)

    def fake_wrap(m):
        arr_pts = np.array([m.GetPoint(i) for i in range(m.GetNumberOfPoints())])
        arr_cells = np.array([4, 0, 1, 2, 3])
        dd = SimpleNamespace(Points=arr_pts, Cells=arr_cells)
        dd.PointData = SimpleNamespace(append=lambda arr, name: setattr(dd, name, arr))
        return dd

    monkeypatch.setattr(dsa, "WrapDataObject", fake_wrap)


def patch_dummy_solver(monkeypatch, log):
    """Replace LeadFieldSolver in module namespace with a recorder."""

    class DummySolver:
        def __init__(self, *, mesh, electrode_pos=None, lead_fields_dict=None):
            log["mesh"] = mesh
            log["electrode_pos"] = electrode_pos
            log["lead_fields_dict"] = lead_fields_dict

    monkeypatch.setattr(mm_mod, "LeadFieldSolver", DummySolver)


def test_init_base_case(monkeypatch):
    # 1) Provide the tiny mesh whenever MeshIO.read_mesh is called
    tiny_mesh = make_tetra_mesh()
    monkeypatch.setattr(MeshIO, "read_mesh", lambda path: tiny_mesh)

    # 2) WrapDataObject stub: expose Points / Cells and allow PointData.append
    def fake_wrap(mesh):
        obj = SimpleNamespace()
        obj.Points = np.array(
            [mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())]
        )
        # VTK’s internal cell array → numpy flat array [4,i0,i1,i2,i3]
        cell_ids = [mesh.GetCell(0).GetPointId(j) for j in range(4)]
        obj.Cells = np.array([4, *cell_ids])

        def append(arr, name):
            setattr(obj, name, arr)

        obj.PointData = SimpleNamespace(append=append)
        return obj

    monkeypatch.setattr(dsa, "WrapDataObject", fake_wrap)

    # 3) Disable locator construction
    monkeypatch.setattr(
        mm_mod.MyocardialMesh, "_build_locator", lambda self, mesh: None
    )

    # 4) Replace LeadFieldSolver inside myocardial_mesh.myocardial_mesh
    init_log = {}

    class DummySolver:
        def __init__(self, *, mesh, electrode_pos=None, lead_fields_dict=None):
            init_log["mesh"] = mesh
            init_log["electrode_pos"] = electrode_pos
            init_log["lead_fields_dict"] = lead_fields_dict

    monkeypatch.setattr(mm_mod, "LeadFieldSolver", DummySolver)

    # 5) Run constructor (base case)
    mm = MyocardialMesh(mesh_path="ignored.vtp")

    # ---------------- Assertions ----------------
    # Mesh stored
    assert mm.vtk_mesh is tiny_mesh
    # No fibres, no electrodes
    assert mm.f0 is None
    assert mm.electrode_pos is None
    # Activation array appended and is +inf
    act = getattr(mm.dd, "activation")
    assert np.isinf(act).all() and act.shape == (4,)
    # Dummy solver called with expected kwargs
    assert init_log == {
        "mesh": tiny_mesh,
        "electrode_pos": None,
        "lead_fields_dict": None,
    }


def test_init_with_electrodes(monkeypatch, tmp_path):
    mesh = make_tetra_mesh()
    patch_read_and_wrap(monkeypatch, mesh)
    monkeypatch.setattr(mm_mod.MyocardialMesh, "_build_locator", lambda s, m: None)

    # Create a pickle with dummy electrodes
    electrodes = {"RA": [1, 2, 3]}
    pkl = tmp_path / "elec.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(electrodes, f)

    log = {}
    patch_dummy_solver(monkeypatch, log)

    mm = MyocardialMesh(mesh_path="ignored", electrodes_position=str(pkl))

    assert mm.electrode_pos == electrodes
    assert log["electrode_pos"] == electrodes


def test_init_with_conductivity(monkeypatch):
    mesh = make_tetra_mesh()
    patch_read_and_wrap(monkeypatch, mesh)
    monkeypatch.setattr(mm_mod.MyocardialMesh, "_build_locator", lambda s, m: None)

    params = np.array([0.1, 0.2, 0.3])
    log = {}
    patch_dummy_solver(monkeypatch, log)

    mm = MyocardialMesh(mesh_path="ignored", conductivity_params=params)

    # D should be (n_cells,3,3) and diag == params
    assert mm.D.shape == (1, 3, 3)
    assert np.allclose(np.diag(mm.D[0]), params)


def test_init_with_fibers(monkeypatch, tmp_path):
    mesh = make_tetra_mesh()
    patch_read_and_wrap(monkeypatch, mesh)
    monkeypatch.setattr(mm_mod.MyocardialMesh, "_build_locator", lambda s, m: None)

    # Fake fibre dataset and reader
    fake_fibres = SimpleNamespace(name="fibre_ds")

    class FakeReader:
        def SetFileName(self, path):
            pass

        def ReadAllVectorsOn(self):
            pass

        def ReadAllScalarsOn(self):
            pass

        def Update(self):
            pass

        def GetOutput(self):
            return fake_fibres

    monkeypatch.setattr(vtk, "vtkDataSetReader", lambda: FakeReader())

    fibre_path = tmp_path / "fib.vtk"
    fibre_path.write_text("")  # file need not exist; reader is patched

    log = {}
    patch_dummy_solver(monkeypatch, log)

    mm = MyocardialMesh(mesh_path="ignored", fibers_path=str(fibre_path))

    assert mm.f0 is fake_fibres


def test_init_builds_locator(monkeypatch):
    mesh = make_tetra_mesh()
    patch_read_and_wrap(monkeypatch, mesh)

    def real_locator(self, m):
        loc = vtk.vtkCellLocator()
        loc.SetDataSet(m)
        loc.BuildLocator()
        return loc

    monkeypatch.setattr(mm_mod.MyocardialMesh, "_build_locator", real_locator)

    log = {}
    patch_dummy_solver(monkeypatch, log)

    mm = MyocardialMesh(mesh_path="ignored")
    assert isinstance(mm.vtk_locator, vtk.vtkCellLocator)
