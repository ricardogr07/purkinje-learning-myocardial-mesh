"""Coupling loop between Purkinje trees and myocardium FIM.

This module implements the iterative procedure used to compute PMJ activation
times from Purkinje trees and to couple them with a myocardial activation
solver, matching the original notebook behavior (legacy seeding, clamping,
and early-stop based on ECG change).
"""

from __future__ import annotations

import logging
import time
from heapq import heappop, heappush
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from myocardial_mesh._types import FloatArray, IntArray

_LOGGER = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def _purkinje_pmj_times_dijkstra(
    tree: object, pmj_idx: NDArray[np.int64], root_idx: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute path-length distances from root to PMJs via Dijkstra on the tree.

    This is a pure-NumPy fallback that builds an adjacency list with Euclidean
    edge lengths and runs single-source shortest paths (SSSP).

    Args:
        tree: Object exposing ``xyz`` (points) and ``connectivity`` (edges).
        pmj_idx: PMJ node indices into ``tree.xyz``.
        root_idx: Root node index (wrapped as length-1 array).

    Returns:
        NDArray[np.float64]: Distances (in *tree units*, i.e., mm) for each PMJ.
    """
    xyz = np.asarray(getattr(tree, "xyz"), dtype=float)
    edges = np.asarray(getattr(tree, "connectivity"), dtype=int)
    n = int(xyz.shape[0])

    # adjacency with edge weights (Euclidean length)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    p0 = xyz[edges[:, 0]]
    p1 = xyz[edges[:, 1]]
    w = np.linalg.norm(p0 - p1, axis=1)
    for (a, b), ww in zip(edges, w):
        adj[a].append((int(b), float(ww)))
        adj[b].append((int(a), float(ww)))

    root: int = int(np.asarray(root_idx, dtype=int).ravel()[0])
    dist: FloatArray = np.full(n, np.inf, dtype=float)
    dist[root] = 0.0

    # Dijkstra
    pq: List[Tuple[float, int]] = []
    heappush(pq, (0.0, root))
    while pq:
        d, u = heappop(pq)
        if d != dist[u]:
            continue
        for v, ww in adj[u]:
            nd = d + ww
            if nd < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))

    return cast(NDArray[np.float64], dist[np.asarray(pmj_idx, dtype=int)])


def _purkinje_pmj_times(
    tree: object,
    pmj_idx: NDArray[np.int64],
    root_idx: NDArray[np.int64],
    root_t_tree: NDArray[np.float64],
    prev_pmj_tree: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return PMJ activation times in tree units using a fast solver or fallback.

    Preferred path:
      * If the tree exposes a fast solver (e.g., ``activate_fim(src_idx, src_vals)``),
        call it with ``[root, pmjs]`` and previous PMJ times as warm-start; return PMJ
        times (skipping the root if present in the return).

    Fallback:
      * Run Dijkstra on the geometric tree to get path-lengths from the root.

    Args:
        tree: Tree object; optionally exposes ``activate_fim``.
        pmj_idx: PMJ node indices into ``tree.xyz``.
        root_idx: Root node index (wrapped as length-1 array).
        root_t_tree: Root time in tree units (mm) as length-1 array.
        prev_pmj_tree: Previous PMJ times in tree units (warm-start).

    Returns:
        NDArray[np.float64]: PMJ times in tree units (mm).
    """
    # Try the tree-native solver first (matches the notebook loop)
    if hasattr(tree, "activate_fim"):
        try:
            src_idx = np.concatenate(
                [
                    np.asarray(root_idx, dtype=int).ravel(),
                    np.asarray(pmj_idx, dtype=int).ravel(),
                ]
            )
            src_vals = np.concatenate(
                [
                    np.asarray(root_t_tree, dtype=float).ravel(),
                    np.asarray(prev_pmj_tree, dtype=float).ravel(),
                ]
            )
            _LOGGER.debug("Calling tree.activate_fim with %d sources", src_idx.size)
            out = np.asarray(
                getattr(tree, "activate_fim")(src_idx, src_vals), dtype=float
            )
            # Some implementations return [root, pmjs]; others return just pmjs.
            if out.shape[0] == pmj_idx.shape[0] + 1:
                out = out[1:]
            return out
        except Exception:
            _LOGGER.debug("Fast path failed; falling back to Dijkstra.", exc_info=True)
            # Fall back to pure-NumPy graph SSSP if the fast path misbehaves
            pass

    # Fallback: geometric SSSP in tree space
    return _purkinje_pmj_times_dijkstra(tree, pmj_idx, root_idx)


def run_ecg_core(
    myocardium: object,
    lv_tree: object,
    rv_tree: object,
    lv_root_idx: int,
    rv_root_idx: int,
    *,
    lv_root_time_ms: float = -75.0,
    rv_root_time_ms: float = -75.0,
    purkinje_cv_m_per_s: float = 2.0,
    pvc_idx: Optional[NDArray[np.int64]] = None,
    pvc_t: Optional[NDArray[np.float64]] = None,
    kmax: int = 8,
    tol_act: float = 1e-6,
    tol_ecg: float = 1e-2,
    verbose: bool = False,
    log: Optional[LogFn] = None,
    trace: bool = False,
    return_diagnostics: bool = False,
    purkinje_engine: str = "jax",
    dedup_pmj_nodes: bool = False,
) -> Tuple[NDArray[np.float64], Dict[str, object]]:
    """Run the Purkinje ↔ Myocardium coupling loop (notebook-compatible).

    The loop alternates between PMJ timing from Purkinje trees and a myocardial
    FIM solve seeded at all PMJ locations (plus optional PVC seeds), clamping
    times per PMJ and stopping early if there is no improvement or ECG change is
    below a threshold.

    Args:
        myocardium: Object exposing ``xyz``, ``activate_fim(...)``, and ``new_get_ecg(...)``.
        lv_tree: Left-ventricular tree object with ``xyz``, ``connectivity``, and ``pmj``.
        rv_tree: Right-ventricular tree object with ``xyz``, ``connectivity``, and ``pmj``.
        lv_root_idx: LV root node index into ``lv_tree.xyz``.
        rv_root_idx: RV root node index into ``rv_tree.xyz``.
        lv_root_time_ms: LV root time offset (ms).
        rv_root_time_ms: RV root time offset (ms).
        purkinje_cv_m_per_s: Conduction velocity (m/s). Note 1 m/s = 1 mm/ms.
        pvc_idx: Optional myocardial node indices for PVC seeding.
        pvc_t: Optional fixed activation times for PVC seeds (ms).
        kmax: Maximum iterations of the coupling loop.
        tol_act: Per-PMJ minimal improvement (ms) to continue iterating.
        tol_ecg: Relative ECG change threshold for early stop.
        verbose: Whether to emit progress via the provided ``log`` function or ``print``.
        log: Optional logging callback; used only when ``verbose=True``.
        trace: Whether to collect per-iteration trace data.
        return_diagnostics: Whether to return a diagnostics dict alongside ECG.
        purkinje_engine: ``"jax"`` (recommended) or ``"uv"`` (fallback path).
        dedup_pmj_nodes: Kept for parity with notebook; ignored (no dedup performed).

    Returns:
        Tuple[NDArray[np.float64], Dict[str, object]]: ECG array and diagnostics
        (empty dict if ``return_diagnostics=False``).
    """
    _log: LogFn = log if (verbose and log is not None) else (lambda s: None)
    if verbose and log is None:
        # Preserve original behavior (print) when verbose is True but no logger is given.
        _log = lambda s: print(s, flush=True)  # noqa: E731
    _LOGGER.info("Starting ECG coupling loop (kmax=%d)", kmax)

    t0_all = time.perf_counter()

    # --- PMJs (indices & coords) ---
    lv_pmj_idx: IntArray = np.asarray(getattr(lv_tree, "pmj"), dtype=int)
    rv_pmj_idx: IntArray = np.asarray(getattr(rv_tree, "pmj"), dtype=int)
    lv_pmj_xyz: FloatArray = np.asarray(getattr(lv_tree, "xyz"), dtype=float)[
        lv_pmj_idx
    ]
    rv_pmj_xyz: FloatArray = np.asarray(getattr(rv_tree, "xyz"), dtype=float)[
        rv_pmj_idx
    ]

    # --- optional PVCs (myocardial nodes with fixed times in ms) ---
    pvc_idx_arr: Optional[IntArray] = (
        None
        if pvc_idx is None or pvc_t is None or len(pvc_idx) == 0
        else np.asarray(pvc_idx, dtype=int)
    )
    pvc_t_arr: Optional[FloatArray] = (
        None if pvc_idx_arr is None else np.asarray(pvc_t, dtype=float)
    )

    # --- Units: 1 m/s == 1 mm/ms
    cv_mm_per_ms: float = float(purkinje_cv_m_per_s)
    _log(
        f"[core] PMJs: LV={lv_pmj_idx.size}, RV={rv_pmj_idx.size}, "
        f"total={lv_pmj_idx.size + rv_pmj_idx.size}"
    )
    _log(
        f"[core] CV={cv_mm_per_ms} mm/ms, root times (ms): "
        f"LV={lv_root_time_ms}, RV={rv_root_time_ms}"
    )

    # --- State in "tree units" (mm) we iterate with; ms companions kept in parallel
    t_lv_tree: FloatArray = np.full(lv_pmj_idx.shape[0], np.inf, dtype=float)
    t_rv_tree: FloatArray = np.full(rv_pmj_idx.shape[0], np.inf, dtype=float)
    t_lv_ms: FloatArray = np.full(lv_pmj_idx.shape[0], np.inf, dtype=float)
    t_rv_ms: FloatArray = np.full(rv_pmj_idx.shape[0], np.inf, dtype=float)

    # roots as arrays for the helper
    lv_root_idx_arr: IntArray = np.asarray([int(lv_root_idx)], dtype=int)
    rv_root_idx_arr: IntArray = np.asarray([int(rv_root_idx)], dtype=int)
    zero_tree: FloatArray = np.asarray([0.0], dtype=float)

    # ECG early-stop
    prev_ecg_vec: Optional[FloatArray] = None
    ecg_err_hist: List[float] = []

    trace_data: Optional[List[Dict[str, object]]] = [] if trace else None
    last_improve_lv: Optional[int] = None
    last_improve_rv: Optional[int] = None

    for it in range(1, kmax + 1):
        t_it = time.perf_counter()
        _log(f"[iter {it}] start")
        _LOGGER.debug("Iteration %d started", it)

        # --- Purkinje pass (tree units via JAX SSSP or the tree's solver)
        t0 = time.perf_counter()
        if purkinje_engine == "jax":
            t_lv_purk_tree = _purkinje_pmj_times(
                lv_tree, lv_pmj_idx, lv_root_idx_arr, zero_tree, t_lv_tree
            )
            t_rv_purk_tree = _purkinje_pmj_times(
                rv_tree, rv_pmj_idx, rv_root_idx_arr, zero_tree, t_rv_tree
            )
        elif purkinje_engine == "uv":
            t_lv_purk_tree = _purkinje_pmj_times(
                lv_tree, lv_pmj_idx, lv_root_idx_arr, zero_tree, t_lv_tree
            )
            t_rv_purk_tree = _purkinje_pmj_times(
                rv_tree, rv_pmj_idx, rv_root_idx_arr, zero_tree, t_rv_tree
            )
        else:
            raise ValueError(f"Unknown purkinje_engine={purkinje_engine!r}")
        _log(f"[iter {it}] Purkinje pass: {time.perf_counter() - t0:.3f}s")

        # Convert to ms and add root offsets (this is what myocardium expects)
        t_lv_purk_ms: FloatArray = t_lv_purk_tree / cv_mm_per_ms + float(
            lv_root_time_ms
        )
        t_rv_purk_ms: FloatArray = t_rv_purk_tree / cv_mm_per_ms + float(
            rv_root_time_ms
        )

        # --- Legacy seeding style: pass EVERY PMJ point (no dedup), sample back at PMJ coords
        x0_xyz: FloatArray = np.vstack([lv_pmj_xyz, rv_pmj_xyz])  # (Mlv+Mrv, 3)
        x0_vals: FloatArray = np.concatenate([t_lv_purk_ms, t_rv_purk_ms])  # (Mlv+Mrv,)

        # Optional PVCs: append to x0 as additional seeds
        if pvc_idx_arr is not None:
            # PVC positions are in myocardial index space; we need xyz coords
            nodes_xyz: FloatArray = np.asarray(getattr(myocardium, "xyz"), dtype=float)
            x0_xyz = np.vstack([x0_xyz, nodes_xyz[pvc_idx_arr]])
            x0_vals = np.concatenate([x0_vals, cast(FloatArray, pvc_t_arr)])

        # --- Myocardium pass: full FIM with legacy projection; sample at PMJ coords
        t0 = time.perf_counter()
        myo_vals_all = getattr(myocardium, "activate_fim")(
            x0=x0_xyz, x0_vals=x0_vals, return_only_pmjs=True
        )
        myo_vals_all = np.asarray(myo_vals_all, dtype=float)  # (Mlv+Mrv,)
        _log(
            f"[iter {it}] Myocardium solve: {time.perf_counter() - t0:.3f}s "
            f"(n_seeds={x0_xyz.shape[0]})"
        )

        # Split back LV/RV
        Mlv = int(lv_pmj_idx.size)
        t_myo_lv_ms: FloatArray = myo_vals_all[:Mlv]
        t_myo_rv_ms: FloatArray = myo_vals_all[Mlv:]

        # --- Clamp in ms exactly like the notebook
        new_t_lv_ms: FloatArray = np.minimum(t_lv_purk_ms, t_myo_lv_ms)
        new_t_rv_ms: FloatArray = np.minimum(t_rv_purk_ms, t_myo_rv_ms)

        # improvements
        imp_lv = int(np.sum((t_lv_ms - new_t_lv_ms) > float(tol_act)))
        imp_rv = int(np.sum((t_rv_ms - new_t_rv_ms) > float(tol_act)))
        last_improve_lv, last_improve_rv = imp_lv, imp_rv
        _log(
            f"[iter {it}] improvements: LV={imp_lv}, RV={imp_rv}; "
            f"Δmax(ms): LV={np.nanmax(t_lv_ms - new_t_lv_ms):.3e}, "
            f"RV={np.nanmax(t_rv_ms - new_t_rv_ms):.3e}"
        )

        t_lv_ms, t_rv_ms = new_t_lv_ms, new_t_rv_ms

        # back to tree units for next Purkinje pass (nonnegative clamp)
        t_lv_tree = np.maximum((t_lv_ms - float(lv_root_time_ms)) * cv_mm_per_ms, 0.0)
        t_rv_tree = np.maximum((t_rv_ms - float(rv_root_time_ms)) * cv_mm_per_ms, 0.0)

        # --- Optional trace bundle
        if trace and trace_data is not None:
            trace_data.append(
                {
                    "t_lv_purk_tree": t_lv_purk_tree.copy(),
                    "t_rv_purk_tree": t_rv_purk_tree.copy(),
                    "t_lv_purk_ms": t_lv_purk_ms.copy(),
                    "t_rv_purk_ms": t_rv_purk_ms.copy(),
                    "t_lv_myo_ms": t_myo_lv_ms.copy(),
                    "t_rv_myo_ms": t_myo_rv_ms.copy(),
                    "imp_lv": imp_lv,
                    "imp_rv": imp_rv,
                }
            )

        # --- ECG early-stop (like the notebook)
        ecg_vec = getattr(myocardium, "new_get_ecg")(
            record_array=False
        )  # shape (12, T)
        ecg_vec = np.asarray(ecg_vec, dtype=float)
        ecg_err = np.inf
        if prev_ecg_vec is not None:
            num = float(np.linalg.norm(ecg_vec - prev_ecg_vec))
            den = float(np.linalg.norm(prev_ecg_vec)) + 1e-12
            ecg_err = num / den
            ecg_err_hist.append(ecg_err)
            _log(f"[iter {it}] ECG error = {ecg_err:.6g}")
        prev_ecg_vec = ecg_vec

        _log(f"[iter {it}] total {time.perf_counter() - t_it:.3f}s")

        # stop if *either* no PMJ improved or ECG change is small
        if (imp_lv == 0 and imp_rv == 0) or (ecg_err < float(tol_ecg)):
            _log(
                f"[iter {it}] converged "
                f"({'no-improve' if (imp_lv == 0 and imp_rv == 0) else f'ECG<{tol_ecg}'})"
            )
            break

    # Final structured ECG
    ecg = getattr(myocardium, "new_get_ecg")(record_array=True)

    info: Dict[str, object] = {
        "iterations": it,
        "last_improve_lv": last_improve_lv,
        "last_improve_rv": last_improve_rv,
        "t_lv_pmj": t_lv_ms,
        "t_rv_pmj": t_rv_ms,
        "n_lv_pmj": int(t_lv_ms.shape[0]),
        "n_rv_pmj": int(t_rv_ms.shape[0]),
        "lv_root_time_ms": float(lv_root_time_ms),
        "rv_root_time_ms": float(rv_root_time_ms),
        "purkinje_cv_m_per_s": float(purkinje_cv_m_per_s),
        "elapsed_s": time.perf_counter() - t0_all,
        "ecg_err_hist": ecg_err_hist,
        "trace": trace_data,
    }
    _log(f"[core] done in {info['elapsed_s']:.3f}s; iterations={it}")
    _LOGGER.info(
        "Coupling loop done in %.3fs over %d iterations", info["elapsed_s"], it
    )

    return (
        (cast(NDArray[np.float64], ecg), info)
        if return_diagnostics
        else (cast(NDArray[np.float64], ecg), {})
    )
