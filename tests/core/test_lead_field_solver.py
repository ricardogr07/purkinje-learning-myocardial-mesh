import pytest
import numpy as np
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa

from myocardial_mesh.core.lead_field_solver import LeadFieldSolver


# Helper to build a minimal mesh with custom activation array
def make_activation_mesh(pts: np.ndarray, activation: np.ndarray):
    # Create a vtkPolyData with given points
    vtk_pts = vtk.vtkPoints()
    for p in pts:
        vtk_pts.InsertNextPoint(*p)
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_pts)
    # Wrap and assign activation
    dd = dsa.WrapDataObject(mesh)
    dd.PointData.append(activation, "activation")
    return mesh


def test_get_lead_field_from_dict():
    # Precomputed lead_fields_dict should be returned as-is
    leads = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}
    solver = LeadFieldSolver(
        mesh=vtk.vtkPolyData(),
        lead_fields_dict=leads,
        stiffness_matrix=np.eye(2),
    )
    assert solver.lead_field is leads


def test_get_lead_field_from_positions():
    # Two points at (0,0,0) and (1,0,0); electrode at (0,0,0)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mesh = make_activation_mesh(pts, np.zeros(2))
    electrode_pos = {"E": [0.0, 0.0, 0.0]}
    solver = LeadFieldSolver(
        mesh=mesh,
        electrode_pos=electrode_pos,
        stiffness_matrix=np.eye(2),
    )
    # lead_field should be 1 / distance: [inf, 1.0]
    lf = solver.lead_field["E"]
    assert np.isinf(lf[0]) and pytest.approx(lf[1]) == 1.0


def test_get_lead_field_no_input_raises():
    with pytest.raises(RuntimeError):
        LeadFieldSolver(mesh=vtk.vtkPolyData())


def test_compute_aux_integrals_missing_Gi_nodal():
    # Provide a dummy lead_fields_dict so init doesn't try to compute from mesh
    dummy_leads = {"A": np.zeros(1)}
    solver = LeadFieldSolver(
        mesh=vtk.vtkPolyData(),
        lead_fields_dict=dummy_leads,
        stiffness_matrix=np.eye(1),
        # gi_nodal omitted
    )
    with pytest.raises(RuntimeError) as exc:
        solver.compute_aux_integrals()
    assert "Missing electrode positions or Gi_nodal" in str(exc.value)


def test_compute_aux_integrals_missing_electrodes():
    # Provide Gi_nodal so init doesn't error, but no electrode_pos
    gi = np.zeros((1, 3, 1))
    solver = LeadFieldSolver(
        mesh=vtk.vtkPolyData(),
        gi_nodal=gi,
        stiffness_matrix=np.eye(1),
        lead_fields_dict={"A": np.zeros(1)},  # needed to skip _get_lead_field errors
    )
    # Remove electrode_pos manually to simulate missing
    solver.electrode_pos = None
    with pytest.raises(RuntimeError) as exc:
        solver.compute_aux_integrals()
    assert "Missing electrode positions or Gi_nodal" in str(exc.value)


def test_compute_ecg_from_activation_missing_K_or_lead_field():
    # Provide lead_fields_dict but no K
    mesh = make_activation_mesh(np.array([[0, 0, 0]]), np.array([0.0]))
    with pytest.raises(RuntimeError):
        solver = LeadFieldSolver(
            mesh=mesh,
            lead_fields_dict={"RA": np.array([1.0])},
            # stiffness_matrix omitted
        )
        solver.compute_ecg_from_activation()


def test_compute_ecg_from_activation_missing_wilson_lead():
    # Provide K and RA,LA but missing LL -> error in Wilson terminal
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    activation = np.array([0.0, 0.0])
    mesh = make_activation_mesh(pts, activation)
    # trivial stiffness: identity 2x2
    K = np.eye(2)
    electrode_pos = {"RA": [0, 0, 0], "LA": [1, 0, 0]}  # missing LL
    solver = LeadFieldSolver(
        mesh=mesh,
        electrode_pos=electrode_pos,
        stiffness_matrix=K,
    )
    with pytest.raises(RuntimeError) as exc:
        solver.compute_ecg_from_activation()
    assert "Missing required electrode" in str(exc.value)


def test_compute_ecg_full_lead_fields_dict():
    # Two-point mesh
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    activation = np.array([0.0, 1.0])
    mesh = make_activation_mesh(pts, activation)
    K = np.eye(2)

    # Build a full lead_fields_dict: one entry per standard lead, values match mesh size
    lead_names = [
        "RA",
        "LA",
        "LL",  # used for Wilson
        "E1",
        "E2",
        "E3",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    # Each key gets a vector of length num_points
    full_leads = {name: np.ones(2) for name in lead_names}

    solver = LeadFieldSolver(
        mesh=mesh,
        lead_fields_dict=full_leads,
        stiffness_matrix=K,
    )

    # Non-record array path
    ecg = solver.compute_ecg_from_activation()
    # Should produce 12 leads x 206 time points
    assert isinstance(ecg, np.ndarray)
    assert ecg.shape == (12, 206)

    # Record-array path
    rec = solver.compute_ecg_from_activation(record_array=True)
    assert rec.dtype.names == (
        "E1",
        "E2",
        "E3",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    assert rec.shape == (206,)
