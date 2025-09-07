from pathlib import Path
import numpy as np
import pytest
import os

from myocardial_mesh import MyocardialMesh
from myocardial_mesh.orchestrator import run_ecg_core
from myocardial_mesh.plotting import ecg_12lead_plot

from purkinje_uv import FractalTreeParameters, FractalTree, PurkinjeTree

SHOW_FIG = False  # set True to show figures during test (if run interactively)

DATA = Path("data/crtdemo")

# Paper/demo “ground-truth” seeds & params
LV_SEEDS = (388, 412)
RV_SEEDS = (198, 186)

LV_INIT_LENGTH = float(35.931537038275316)
RV_INIT_LENGTH = float(79.86354832236707)

LV_FAS_LEN = [float(0.5 * 4.711579058738858), float(0.5 * 9.129484609771032)]
RV_FAS_LEN = [float(0.5 * 21.703867933650002), float(0.5 * 5.79561866201451)]

LV_FAS_ANG = [float(0.1 * 0.14448952070696136), float(0.1 * 0.23561944901923448)]
RV_FAS_ANG = [float(0.1 * 0.23561944901923448), float(0.1 * 0.23561944901923448)]

COMMON = dict(length=8.0, w=0.1, l_segment=1.0, branch_angle=0.15, N_it=20)


def _build_tree(meshfile: Path, seeds, init_len, fas_len, fas_ang) -> PurkinjeTree:
    params = FractalTreeParameters(
        meshfile=str(meshfile),
        init_node_id=seeds[0],
        second_node_id=seeds[1],
        init_length=init_len,
        length=COMMON["length"],
        w=COMMON["w"],
        l_segment=COMMON["l_segment"],
        fascicles_length=fas_len,
        fascicles_angles=fas_ang,
        branch_angle=COMMON["branch_angle"],
        N_it=COMMON["N_it"],
    )
    ft = FractalTree(params=params)
    ft.grow_tree()
    return PurkinjeTree(
        nodes=np.asarray(ft.nodes_xyz, dtype=float),
        connectivity=np.asarray(ft.connectivity, dtype=int),
        end_nodes=np.asarray(ft.end_nodes, dtype=int),
    )


@pytest.mark.acceptance
def test_e2e_acceptance_crtdemo(tmp_path):
    myo = MyocardialMesh(
        myo_mesh=str(DATA / "crtdemo_mesh_oriented.vtk"),
        electrodes_position=str(DATA / "electrode_pos.pkl"),
        fibers=str(DATA / "crtdemo_f0_oriented.vtk"),
    )

    LV = _build_tree(
        DATA / "crtdemo_LVendo_heart_cut.obj",
        LV_SEEDS,
        LV_INIT_LENGTH,
        LV_FAS_LEN,
        LV_FAS_ANG,
    )
    RV = _build_tree(
        DATA / "crtdemo_RVendo_heart_cut.obj",
        RV_SEEDS,
        RV_INIT_LENGTH,
        RV_FAS_LEN,
        RV_FAS_ANG,
    )

    ecg, info = run_ecg_core(
        myocardium=myo,
        lv_tree=LV,
        rv_tree=RV,
        lv_root_idx=0,
        rv_root_idx=0,
        lv_root_time_ms=-75.0,
        rv_root_time_ms=-75.0,
        purkinje_cv_m_per_s=2.0,
        kmax=3,
        tol_act=1e-6,
        verbose=False,
        return_diagnostics=True,
    )

    # assertions: structure, finiteness, non-trivial energy
    assert isinstance(ecg, np.ndarray) and ecg.dtype.names
    lead_names = ecg.dtype.names
    n = len(ecg[lead_names[0]])
    assert n > 100 and all(len(ecg[lead]) == n for lead in lead_names)

    # finite & non-zero variance on all leads
    finite_ok = sum(
        np.isfinite(ecg[lead]).all() and float(np.var(ecg[lead])) > 1e-14
        for lead in lead_names
    )
    assert finite_ok >= 12

    # iterations bounded
    assert 1 <= info.get("iterations", 0) <= 3

    # simple ECG plot
    fig, _ = ecg_12lead_plot(ecg, suptitle="CRT Demo (acceptance)")
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
