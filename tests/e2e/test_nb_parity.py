"""
Baseline parity vs the notebook (“ground truth”) physiology.

What this test proves
---------------------
Given the *same* endocardial activation field that the original Jupyter
notebook produced (NB/True_endo.vtu, which contains a PointData array
named 'activation'), our refactored MyocardialMesh must synthesize the
exact same 12-lead ECG (NB/True_ecg pickled structured array).

This isolates the Myocardium + ECG synthesis pieces from Purkinje growth
and coupling details. If this test passes, any waveform mismatch you see
elsewhere is NOT an ECG synthesis problem; it’s upstream (trees/coupling).
"""

from pathlib import Path
import pickle
import numpy as np
import pytest
import os

from myocardial_mesh import MyocardialMesh
from myocardial_mesh.plotting import ecg_12lead_plot


DATA = Path("data/crtdemo")
NB = DATA / "nb"

TOL_RMSE = 1e-6
SHOW_FIG = False


@pytest.mark.baseline
def test_nb_parity_ecg_from_true_endo(tmp_path):
    """
    Load the notebook’s endocardium (already contains 'activation'),
    run ONLY the ECG synthesis in our code, and assert bit-level parity.
    """
    myo = MyocardialMesh(
        myo_mesh=str(NB / "True_endo.vtu"),
        electrodes_position=str(DATA / "electrode_pos.pkl"),
        fibers=str(DATA / "crtdemo_f0_oriented.vtk"),
    )

    ecg_hat = myo.new_get_ecg(record_array=True)

    with open(NB / "True_ecg", "rb") as f:
        ecg_ref = pickle.load(f)

    assert ecg_hat.dtype.names == ecg_ref.dtype.names
    n = len(ecg_ref[ecg_ref.dtype.names[0]])
    assert all(len(ecg_ref[name]) == n for name in ecg_ref.dtype.names)
    assert all(len(ecg_hat[name]) == n for name in ecg_hat.dtype.names)

    for lead in ecg_ref.dtype.names:
        a = np.asarray(ecg_ref[lead], float)
        b = np.asarray(ecg_hat[lead], float)
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        np.testing.assert_allclose(
            b, a, rtol=0.0, atol=TOL_RMSE, err_msg=f"Lead {lead} (rmse={rmse:.3e})"
        )

    # simple ECG plot
    fig, _ = ecg_12lead_plot(ecg_hat, suptitle="NB parity: ECG from True_endo.vtu")
    out_png = tmp_path / "crtdemo_12lead.png"
    fig.savefig(out_png, dpi=120)

    assert out_png.exists(), f"Expected ECG image at {out_png} but it was not created."
    size = out_png.stat().st_size
    assert size > 0, f"ECG image {out_png} is empty (0 bytes)."
    print(f"[plot] saved to: {out_png.resolve()}  ({size} bytes)")

    if SHOW_FIG:
        try:
            p = str(out_png.resolve())
            os.startfile(p)
            print("[plot] opened in default viewer.")
        except Exception as e:
            print(f"[plot] could not open image automatically: {e}")
