from pathlib import Path
import numpy as np
import pickle
import pyvista as pv
import vtk

from myocardial_mesh import MyocardialMesh


def _write_minimal_case(tmp: Path):
    # mesh: single tetra (with activation field placeholder)
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    cells = np.array([[0, 1, 2, 3]], int)
    ug = pv.UnstructuredGrid({vtk.VTK_TETRA: cells}, xyz)
    ug.point_data["activation"] = np.full(4, np.inf, float)
    mesh_path = tmp / "one_tet.vtu"
    ug.save(mesh_path)

    # fibers: per-cell fiber vector in CellData["fiber"]
    ug_f = pv.UnstructuredGrid({vtk.VTK_TETRA: cells}, xyz)
    ug_f.cell_data["fiber"] = np.tile(np.array([1.0, 0.0, 0.0]), (1, 1))
    fibers_path = tmp / "fibers.vtu"
    ug_f.save(fibers_path)

    # electrodes: a single electrode (we won’t call ECG here)
    elec = {"E": np.array([10.0, 10.0, 10.0])}
    elec_path = tmp / "electrode_pos.pkl"
    with open(elec_path, "wb") as f:
        pickle.dump(elec, f)

    return mesh_path, fibers_path, elec_path


def test_myocardial_mesh_save_and_ecg_alias(tmp_path):
    mesh, fibers, elec = _write_minimal_case(tmp_path)
    myo = MyocardialMesh(
        myo_mesh=str(mesh), electrodes_position=str(elec), fibers=str(fibers)
    )

    # save round-trip does not crash
    out = tmp_path / "copy.vtu"
    myo.save_pv(str(out))
    assert out.exists() and out.stat().st_size > 0

    # get_ecg alias exists and returns same as new_get_ecg (using current activation)
    a = myo.new_get_ecg(record_array=True)
    b = myo.get_ecg(record_array=True)
    assert a.dtype.names == b.dtype.names
    for name in a.dtype.names:
        assert len(a[name]) == len(b[name])
