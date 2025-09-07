import numpy as np
import pytest
from scipy import sparse

from myocardial_mesh.lead_field.lead_field_solver import LeadFieldSolver


def _eye_Gi(N: int) -> np.ndarray:
    Gi = np.zeros((N, 3, 3), dtype=float)
    Gi[:, 0, 0] = 1.0
    Gi[:, 1, 1] = 1.0
    Gi[:, 2, 2] = 1.0
    return Gi


@pytest.mark.parametrize("N", [3, 5])
def test_compute_aux_Vl_shapes_and_values_isotropic(N):
    """
    Nodes on +Z axis, electrode at origin, Gi = I.
    Expected directional weight along z is 1/z^2.
    Accept both solver outputs:
      - scalar (N,): equals 1/z^2
      - vector (N,3): z-component equals 1/z^2 and x,y are ~0
    """
    z = np.arange(1, N + 1, dtype=float)
    xyz = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    Gi_nodal = _eye_Gi(N)
    K = sparse.eye(N, format="csr")
    electrode_pos = {"E": np.array([0.0, 0.0, 0.0])}

    lead = LeadFieldSolver(xyz=xyz, electrode_pos=electrode_pos, Gi_nodal=Gi_nodal, K=K)
    aux = lead.compute_aux_Vl()
    assert isinstance(aux, dict) and "E" in aux

    v = aux["E"]
    if v.ndim == 1:
        # scalar contraction case
        expected = 1.0 / (z**2)
        np.testing.assert_allclose(v, expected, rtol=0.0, atol=1e-12)
    else:
        assert v.shape == (N, 3)
        np.testing.assert_allclose(v[:, 2], 1.0 / (z**2), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(v[:, 0], 0.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(v[:, 1], 0.0, rtol=0.0, atol=1e-12)


def test_compute_aux_Vl_respects_anisotropy():
    """
    Nodes on +X axis, electrode at origin, Gi scales X by 2.
    Expected directional weight along x is 2/x^2.
    Accept both scalar and vector outputs (see above).
    """
    N = 4
    x = np.arange(1, N + 1, dtype=float)
    xyz = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

    Gi_nodal = _eye_Gi(N)
    Gi_nodal[:, 0, 0] = 2.0  # anisotropy along X
    K = sparse.eye(N, format="csr")
    electrode_pos = {"E": np.array([0.0, 0.0, 0.0])}

    lead = LeadFieldSolver(xyz=xyz, electrode_pos=electrode_pos, Gi_nodal=Gi_nodal, K=K)
    aux = lead.compute_aux_Vl()
    v = aux["E"]

    if v.ndim == 1:
        expected = 2.0 / (x**2)
        np.testing.assert_allclose(v, expected, rtol=0.0, atol=1e-12)
    else:
        assert v.shape == (N, 3)
        np.testing.assert_allclose(v[:, 0], 2.0 / (x**2), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(v[:, 1], 0.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(v[:, 2], 0.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("N", [3, 6])
def test_get_lead_field_shapes_and_values(N):
    """
    get_lead_field() must return 1/||r|| for each electrode and node.
    With nodes on +Y, electrode A at origin: 1/y.
    Electrode B at y=1: 1/|y-1| (has a singular point → inf).
    """
    y = np.arange(1, N + 1, dtype=float)
    xyz = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)

    Gi_nodal = _eye_Gi(N)  # unused by get_lead_field but required at init
    K = sparse.eye(N, format="csr")
    electrodes = {
        "A": np.array([0.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
    }

    lead = LeadFieldSolver(xyz=xyz, electrode_pos=electrodes, Gi_nodal=Gi_nodal, K=K)
    lf = lead.get_lead_field()

    expected_A = 1.0 / y
    np.testing.assert_allclose(lf["A"], expected_A, rtol=0.0, atol=1e-12)

    expected_B = 1.0 / np.abs(y - 1.0)
    mask = y != 1.0  # avoid the singularity (inf)
    np.testing.assert_allclose(lf["B"][mask], expected_B[mask], rtol=0.0, atol=1e-12)
    assert np.isinf(lf["B"][~mask]).all()


def test_ecg_from_activation_shapes_and_recordarray_flag():
    N = 5
    xyz = np.zeros((N, 3), dtype=float)
    Gi_nodal = np.repeat(np.eye(3)[None, :, :], N, axis=0)
    K = sparse.eye(N, format="csr")

    lead = LeadFieldSolver(xyz=xyz, electrode_pos={}, Gi_nodal=Gi_nodal, K=K)

    # activation field (monotone ramp)
    u = np.linspace(0.0, 10.0, N, dtype=float)

    # synthetic lead-field weights for required leads
    ones = np.ones(N, dtype=float)
    lead_field = {
        "RA": ones,
        "LA": ones,
        "LL": ones,
        "V1": ones,
        "V2": ones,
        "V3": ones,
        "V4": ones,
        "V5": ones,
        "V6": ones,
    }

    # record_array=True
    out = lead.ecg_from_activation(u, record_array=True, lead_field=lead_field)
    assert out.dtype.names == (
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
    T = len(out["E1"])
    assert T > 0 and all(len(out[k]) == T for k in out.dtype.names)

    # record_array=False
    out2 = lead.ecg_from_activation(u, record_array=False, lead_field=lead_field)
    assert out2.shape == (12, T)
