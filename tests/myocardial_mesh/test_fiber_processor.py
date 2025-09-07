from pathlib import Path
import numpy as np

from myocardial_mesh import MyocardialMesh
from myocardial_mesh.io.fiber_processor import process_fibers

DATA = Path("data/crtdemo")
NB = DATA / "nb"


def _load_xyz_cells():
    myo = MyocardialMesh(
        myo_mesh=str(NB / "True_endo.vtu"),
        electrodes_position=str(DATA / "electrode_pos.pkl"),
        fibers=str(DATA / "crtdemo_f0_oriented.vtk"),
        device="cpu",
    )
    # use the class just to read xyz/cells reliably
    return np.asarray(myo.xyz, float), myo.cells


def test_fiber_processor_shapes_and_norms():
    xyz, cells = _load_xyz_cells()
    fr = process_fibers(str(DATA / "crtdemo_f0_oriented.vtk"), xyz=xyz, cells=cells)

    assert fr.l_nodes.shape == (xyz.shape[0], 3)
    assert fr.l_cell.shape == (cells.shape[0], 3)
    assert fr.Gi_nodal.shape == (xyz.shape[0], 3, 3)
    assert fr.Gi_cell.shape == (cells.shape[0], 3, 3)
    assert fr.D.shape == (cells.shape[0], 3, 3)

    # unit norms
    assert np.allclose(np.linalg.norm(fr.l_nodes, axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.linalg.norm(fr.l_cell, axis=1), 1.0, atol=1e-8)

    # symmetry
    assert np.allclose(fr.Gi_nodal, np.swapaxes(fr.Gi_nodal, -1, -2), atol=1e-12)
    assert np.allclose(fr.Gi_cell, np.swapaxes(fr.Gi_cell, -1, -2), atol=1e-12)
    assert np.allclose(fr.D, np.swapaxes(fr.D, -1, -2), atol=1e-12)


def test_fiber_processor_cv_positive():
    xyz, cells = _load_xyz_cells()
    fr = process_fibers(str(DATA / "crtdemo_f0_oriented.vtk"), xyz=xyz, cells=cells)
    assert fr.cv_fiber_m_per_s > 0.0
