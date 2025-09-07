from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pytest

from myocardial_mesh import MyocardialMesh
from myocardial_mesh.orchestrator import run_ecg_core
from myocardial_mesh.plotting import ecg_12lead_plot

from purkinje_uv import FractalTreeParameters, FractalTree, PurkinjeTree

# Toggle to auto-open the saved PNGs in the default viewer (Windows only here)
SHOW_FIG = False

DATA = Path("data/crtdemo")

# Demo seeds/params (same as acceptance test)
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


def _ecg_to_matrix(ecg_struct) -> np.ndarray:
    """(12,T) float matrix from structured array."""
    leads = ecg_struct.dtype.names
    return np.stack([np.asarray(ecg_struct[lead], float) for lead in leads], axis=0)


def _corr(a, b) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.mark.jax
def test_jax_e2e_vs_uv(tmp_path):
    """
    Apples-to-apples E2E: UV Purkinje pass vs JAX Purkinje pass.

    Notebook-like settings:
      - CV = 2.5 m/s  (1 m/s = 1 mm/ms)
      - root offsets = 0 ms
      - PMJ dedup OFF
      - legacy “seed every PMJ point” projection style
      - ≥2 iterations, early-stop with ECG relative error (1e-2)
    """

    # Myocardium
    myo = MyocardialMesh(
        myo_mesh=str(DATA / "crtdemo_mesh_oriented.vtk"),
        electrodes_position=str(DATA / "electrode_pos.pkl"),
        fibers=str(DATA / "crtdemo_f0_oriented.vtk"),
    )

    # Trees
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

    # --- UV path (tree’s own solver) ---
    ecg_uv, info_uv = run_ecg_core(
        myocardium=myo,
        lv_tree=LV,
        rv_tree=RV,
        lv_root_idx=0,
        rv_root_idx=0,
        lv_root_time_ms=0.0,
        rv_root_time_ms=0.0,
        purkinje_cv_m_per_s=2.5,
        kmax=3,
        tol_act=1e-6,
        tol_ecg=1e-2,
        verbose=False,
        return_diagnostics=True,
        purkinje_engine="uv",
        dedup_pmj_nodes=False,
    )
    assert isinstance(ecg_uv, np.ndarray) and ecg_uv.dtype.names
    L = ecg_uv.dtype.names
    T = len(ecg_uv[L[0]])
    assert T > 100 and all(len(ecg_uv[lead]) == T for lead in L)

    # Save a figure
    fig_uv, _ = ecg_12lead_plot(
        ecg_uv, suptitle=f"CRT Demo (UV) — iter={info_uv.get('iterations')}"
    )
    png_uv = tmp_path / "ecg_uv.png"
    fig_uv.savefig(png_uv, dpi=120)
    assert png_uv.exists() and png_uv.stat().st_size > 0

    # --- JAX path (distance-based SSSP) ---
    ecg_jax, info_jax = run_ecg_core(
        myocardium=myo,
        lv_tree=LV,
        rv_tree=RV,
        lv_root_idx=0,
        rv_root_idx=0,
        lv_root_time_ms=0.0,
        rv_root_time_ms=0.0,
        purkinje_cv_m_per_s=2.5,
        kmax=3,
        tol_act=1e-6,
        tol_ecg=1e-2,
        verbose=False,
        return_diagnostics=True,
        purkinje_engine="jax",
        dedup_pmj_nodes=False,
    )
    assert isinstance(ecg_jax, np.ndarray) and ecg_jax.dtype.names == L
    assert all(len(ecg_jax[lead]) == T for lead in L)

    # Save a figure
    fig_jax, _ = ecg_12lead_plot(
        ecg_jax, suptitle=f"CRT Demo (JAX) — iter={info_jax.get('iterations')}"
    )
    png_jax = tmp_path / "ecg_jax.png"
    fig_jax.savefig(png_jax, dpi=120)
    assert png_jax.exists() and png_jax.stat().st_size > 0

    # --- Compare JAX vs UV (log metrics; keep assertions gentle) ---
    M_uv = _ecg_to_matrix(ecg_uv)  # (12,T)
    M_jax = _ecg_to_matrix(ecg_jax)  # (12,T)

    cors, rmses = [], []
    for i, lead in enumerate(L):
        a = M_uv[i]
        b = M_jax[i]
        cors.append(_corr(a, b))
        rmses.append(float(np.sqrt(np.mean((a - b) ** 2))))

    # Log summary
    print("\nJAX vs UV per-lead metrics (corr, rmse):")
    for lead, c, r in zip(L, cors, rmses):
        print(f"  {lead:>3}: corr={c: .3f}  rmse={r: .3f}")

    # Light sanity assertions to keep the test robust across environments
    assert np.isfinite(M_uv).all() and np.isfinite(M_jax).all()
    # Require at least "some" agreement
    assert np.median(cors) > 0.2  # relax if needed
    # Prevent wild divergence
    assert np.max(rmses) < 20.0  # mV-scale guard

    # Optionally open the plots (Windows)
    if SHOW_FIG:
        try:
            os.startfile(str(png_uv))  # type: ignore[attr-defined]
            os.startfile(str(png_jax))  # type: ignore[attr-defined]
        except Exception:
            pass
