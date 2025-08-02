import numpy as np
import pyvista as pv
import pytest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myocardial_mesh.viz.data_plotter import DataPlotter


@pytest.fixture
def dummy_mesh() -> pv.PolyData:
    """Create a simple dummy mesh."""
    return pv.Sphere(radius=1.0)


def test_plot_single_mesh(dummy_mesh):
    """Test plotting a single mesh."""
    try:
        DataPlotter.plot_mesh(dummy_mesh, show=False)
    except Exception as e:
        pytest.fail(f"plot_mesh raised an exception: {e}")


def test_plot_multiple_meshes(dummy_mesh):
    """Test plotting multiple meshes."""
    try:
        DataPlotter.plot_mesh([dummy_mesh, dummy_mesh.copy()], show=False)
    except Exception as e:
        pytest.fail(f"plot_mesh with multiple inputs raised: {e}")


def test_plot_mesh_with_show(monkeypatch):
    """Test plot_mesh with show=True to cover plotter.show()."""

    # Create dummy mesh
    sphere = pv.Sphere()

    # Patch the PyVista plotter show method
    with patch.object(pv.Plotter, "show", return_value=None) as mock_show:
        DataPlotter.plot_mesh(sphere, show=True)
        mock_show.assert_called_once()


def test_plot_ecg_valid():
    """Test ECG plotting with realistic lead names and values."""
    n_leads = 12
    n_times = 100
    req_time_ini = 0.0
    req_time_fin = 1.0
    leads = [f"V{i}" for i in range(1, n_leads + 1)]
    arrays = [np.sin(np.linspace(0, 2 * np.pi, n_times) + i) for i in range(n_leads)]

    try:
        DataPlotter.plot_ecg(
            arrays=np.array(arrays),
            leads_names=leads,
            req_time_ini=req_time_ini,
            req_time_fin=req_time_fin,
            n_times=n_times,
        )
    except Exception as e:
        pytest.fail(f"plot_ecg raised an exception: {e}")
    finally:
        plt.close("all")


def test_plot_ecg_invalid_shape():
    """Should raise ValueError if number of arrays doesn't match lead names."""
    n_times = 100
    arrays = np.random.randn(10, n_times)  # 10 leads
    leads = [f"V{i}" for i in range(12)]  # 12 names, mismatch

    with pytest.raises(ValueError):
        DataPlotter.plot_ecg(
            arrays=arrays,
            leads_names=leads,
            req_time_ini=0.0,
            req_time_fin=1.0,
            n_times=n_times,
        )
