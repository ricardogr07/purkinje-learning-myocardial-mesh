import numpy as np
from myocardial_mesh import MyocardialMesh
from pathlib import Path


def test_assemble_stiffness_matrix(monkeypatch):
    mm = MyocardialMesh(mesh_path="ignored")
    mm.D = np.ones((1, 3, 3))
    dummy_K = np.array([[[42.0]]])

    monkeypatch.setattr(
        "myocardial_mesh.myocardial_mesh.compute_Bmatrix",
        lambda xyz, cells: (np.zeros((1, 3, 4)), np.ones(1)),
    )
    monkeypatch.setattr(
        "myocardial_mesh.myocardial_mesh.compute_local_stiffness_matrix",
        lambda B, J, D: dummy_K,
    )

    mm.assemble_stiffness_matrix()
    assert np.array_equal(mm.K, dummy_K)


def test_compute_ecg(monkeypatch):
    mm = MyocardialMesh(mesh_path="ignored")
    mm.solver.compute_ecg_from_activation = lambda **kw: "ecg"
    assert mm.compute_ecg(record_array=False) == "ecg"


def test_compute_aux(monkeypatch):
    mm = MyocardialMesh(mesh_path="ignored")
    mm.solver.compute_aux_integrals = lambda: "aux"
    assert mm.compute_ecg_aux_field() == "aux"


def test_sample_activation_at(monkeypatch):
    mm = MyocardialMesh(mesh_path="ignored")
    monkeypatch.setattr(
        "myocardial_mesh.core.vtk_geometry_utils.VTKGeometryUtils.probe_activation",
        lambda mesh, pts: np.array([7.7]),
    )
    out = mm.sample_activation_at(np.zeros((1, 3)))
    assert np.allclose(out, [7.7])


def test_project_pmjs(monkeypatch):
    mm = MyocardialMesh(mesh_path="ignored")
    monkeypatch.setattr(
        "myocardial_mesh.core.vtk_geometry_utils.VTKGeometryUtils.find_closest_pmjs",
        lambda pmjs, loc: pmjs + 1,
    )
    pmjs = np.array([[1, 1, 1]])
    assert np.allclose(mm.project_pmjs(pmjs), pmjs + 1)


def test_save(monkeypatch, tmp_path):
    mm = MyocardialMesh(mesh_path="ignored")
    dest = tmp_path / "out.vtp"
    log = {}
    monkeypatch.setattr(
        "myocardial_mesh.io.mesh_io.MeshIO.write",
        lambda mesh, path, method: log.setdefault("path", Path(path)),
    )
    mm.save(dest)
    assert log["path"] == dest


def test_plot(monkeypatch):
    mm = MyocardialMesh(mesh_path="ignored")
    flag = {}
    monkeypatch.setattr(
        "myocardial_mesh.viz.data_plotter.DataPlotter.plot_mesh",
        lambda m: flag.setdefault("hit", True),
    )
    mm.plot()
    assert flag.get("hit")


def test_plot_ecg(monkeypatch):
    mm = MyocardialMesh(mesh_path="ignored")
    arr, names = np.zeros((2, 10)), ["V1", "V2"]
    flag = {}
    monkeypatch.setattr(
        "myocardial_mesh.viz.data_plotter.DataPlotter.plot_ecg",
        lambda a, n, t0, t1, nt: flag.setdefault("args", (a, n, t0, t1, nt)),
    )
    mm.plot_ecg(arr, names, 0, 1, 10)
    assert flag["args"][0] is arr and flag["args"][1] is names
