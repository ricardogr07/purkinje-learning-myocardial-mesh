from __future__ import annotations

import numpy as np
import pytest

from myocardial_mesh.orchestrator import (
    _purkinje_pmj_times,
    _purkinje_pmj_times_dijkstra,
    run_ecg_core,
)


class DummyTree:
    def __init__(
        self,
        xyz: np.ndarray,
        connectivity: np.ndarray,
        pmj: np.ndarray,
        *,
        activate_returns: np.ndarray | None = None,
        fail: bool = False,
    ) -> None:
        self.xyz = np.asarray(xyz, dtype=float)
        self.connectivity = np.asarray(connectivity, dtype=int)
        self.pmj = np.asarray(pmj, dtype=int)
        self._activate_returns = activate_returns
        self._fail = fail

    def activate_fim(self, src_idx: np.ndarray, src_vals: np.ndarray) -> np.ndarray:
        if self._fail:
            raise RuntimeError("boom")
        if self._activate_returns is not None:
            return np.asarray(self._activate_returns, dtype=float)
        return np.zeros(self.pmj.shape[0], dtype=float)


class DummyMyocardium:
    def __init__(self, xyz: np.ndarray, pmj_count: int) -> None:
        self.xyz = np.asarray(xyz, dtype=float)
        self._pmj_count = int(pmj_count)

    def activate_fim(self, x0: np.ndarray, x0_vals: np.ndarray, return_only_pmjs: bool):
        _ = x0
        _ = return_only_pmjs
        return np.asarray(x0_vals, dtype=float)[: self._pmj_count]

    def new_get_ecg(self, record_array: bool = True) -> np.ndarray:
        _ = record_array
        return np.zeros((12, 5), dtype=float)


def test_purkinje_pmj_times_dijkstra_line_tree():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]], dtype=int)
    tree = DummyTree(xyz=xyz, connectivity=edges, pmj=np.array([2], dtype=int))

    pmj_idx = np.array([2], dtype=int)
    root_idx = np.array([0], dtype=int)
    out = _purkinje_pmj_times_dijkstra(tree, pmj_idx, root_idx)

    assert np.allclose(out, [2.0])


def test_purkinje_pmj_times_fast_path_and_fallback():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]], dtype=int)
    pmj_idx = np.array([2], dtype=int)
    root_idx = np.array([0], dtype=int)
    root_t_tree = np.array([0.0], dtype=float)
    prev_pmj_tree = np.array([1.0], dtype=float)

    tree_ok = DummyTree(
        xyz=xyz,
        connectivity=edges,
        pmj=pmj_idx,
        activate_returns=np.array([0.0, 4.5], dtype=float),
    )
    out_ok = _purkinje_pmj_times(tree_ok, pmj_idx, root_idx, root_t_tree, prev_pmj_tree)
    assert np.allclose(out_ok, [4.5])

    tree_fail = DummyTree(
        xyz=xyz,
        connectivity=edges,
        pmj=pmj_idx,
        fail=True,
    )
    out_fail = _purkinje_pmj_times(
        tree_fail, pmj_idx, root_idx, root_t_tree, prev_pmj_tree
    )
    assert np.allclose(out_fail, [2.0])


def test_run_ecg_core_with_pvc_trace_verbose(monkeypatch):
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = np.array([[0, 1]], dtype=int)

    lv_tree = DummyTree(xyz=xyz, connectivity=edges, pmj=np.array([1], dtype=int))
    rv_tree = DummyTree(xyz=xyz, connectivity=edges, pmj=np.array([1], dtype=int))
    myo = DummyMyocardium(xyz=xyz, pmj_count=2)

    ecg, info = run_ecg_core(
        myocardium=myo,
        lv_tree=lv_tree,
        rv_tree=rv_tree,
        lv_root_idx=0,
        rv_root_idx=0,
        lv_root_time_ms=0.0,
        rv_root_time_ms=0.0,
        purkinje_cv_m_per_s=1.0,
        pvc_idx=np.array([0], dtype=int),
        pvc_t=np.array([0.0], dtype=float),
        kmax=1,
        tol_act=1e-6,
        tol_ecg=1e-6,
        verbose=True,
        trace=True,
        return_diagnostics=True,
        purkinje_engine="uv",
        dedup_pmj_nodes=False,
    )

    assert isinstance(ecg, np.ndarray)
    assert ecg.shape == (12, 5)
    assert info["trace"] is not None
    assert len(info["trace"]) == 1


def test_run_ecg_core_invalid_engine_raises():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = np.array([[0, 1]], dtype=int)

    lv_tree = DummyTree(xyz=xyz, connectivity=edges, pmj=np.array([1], dtype=int))
    rv_tree = DummyTree(xyz=xyz, connectivity=edges, pmj=np.array([1], dtype=int))
    myo = DummyMyocardium(xyz=xyz, pmj_count=2)

    with pytest.raises(ValueError):
        run_ecg_core(
            myocardium=myo,
            lv_tree=lv_tree,
            rv_tree=rv_tree,
            lv_root_idx=0,
            rv_root_idx=0,
            purkinje_engine="nope",
            kmax=1,
        )
