# tests/integration_tests/test_integration_demo.py
import numpy as np
import pytest
from pathlib import Path
import vtk

from myocardial_mesh.io.mesh_io import MeshIO
from myocardial_mesh.myocardial_mesh import MyocardialMesh
from myocardial_mesh.core.lead_field_solver import LeadFieldSolver
from myocardial_mesh.core.vtk_geometry_utils import VTKGeometryUtils

DEMO_DIR = Path("data/crtdemo")
MESH_FILE = DEMO_DIR / "crtdemo_mesh_oriented.vtk"
FIBRE_FILE = DEMO_DIR / "crtdemo_f0_oriented.vtk"
ELECTRODE_FILE = DEMO_DIR / "electrode_pos.pkl"

pytestmark = pytest.mark.skipif(
    not (MESH_FILE.exists() and FIBRE_FILE.exists() and ELECTRODE_FILE.exists()),
    reason="CRT-demo files not found",
)


class _NoECGSolver(LeadFieldSolver):
    def get_ecg(self, **kw):  # type: ignore[override]
        return "dummy"

    def get_aux_Vl(self):  # type: ignore[override]
        return "dummy"


def test_myocardialmesh_with_real_demo(tmp_path, monkeypatch):
    import myocardial_mesh.myocardial_mesh as mm_mod

    monkeypatch.setattr(mm_mod, "LeadFieldSolver", _NoECGSolver, raising=True)

    mm = MyocardialMesh(
        mesh_path=str(MESH_FILE),
        fibers_path=str(FIBRE_FILE),
        electrodes_position=str(ELECTRODE_FILE),
        conductivity_params=np.array([0.2, 0.2, 0.2]),
    )

    assert mm.xyz.shape[0] > 0 and mm.cells.shape[0] > 0
    assert mm.f0 is not None
    assert isinstance(mm.electrode_pos, dict) and len(mm.electrode_pos) > 0

    mm.assemble_stiffness_matrix()
    assert mm.K is not None and mm.K.shape == (mm.cells.shape[0], 4, 4)

    bounds = mm.vtk_mesh.GetBounds()
    rng = np.random.default_rng(0)
    guesses = rng.uniform(
        [bounds[0], bounds[2], bounds[4]],
        [bounds[1], bounds[3], bounds[5]],
        size=(3, 3),
    )
    proj = mm.project_pmjs(guesses)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(mm.vtk_mesh)
    locator.BuildLocator()

    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for p in proj:
        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
        assert dist2 < 1e-6

    sampled = VTKGeometryUtils.probe_activation(mm.vtk_mesh, mm.xyz[:2])
    assert sampled.shape == (2,) and (~np.isfinite(sampled)).all()

    out_file = tmp_path / "mesh_saved.vtk"
    call_log = {}

    def _stub_write(mesh, path, *args, **kwargs):
        call_log["mesh"] = mesh
        call_log["path"] = path
        return True

    monkeypatch.setattr(MeshIO, "write", _stub_write, raising=True)

    mm.save(str(out_file))

    assert call_log["mesh"] is mm.vtk_mesh
    assert call_log["path"] == out_file
