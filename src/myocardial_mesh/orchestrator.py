from __future__ import annotations
import time
from typing import Dict, Optional, Tuple, Callable, List
import numpy as np
from heapq import heappush, heappop

LogFn = Callable[[str], None]


def _purkinje_pmj_times_dijkstra(
    tree, pmj_idx: np.ndarray, root_idx: np.ndarray
) -> np.ndarray:
    """
    Fallback: compute SSSP distances on the tree with Euclidean edge lengths.
    Returns times in *tree units* (mm), i.e., path length from the root.
    """
    xyz = np.asarray(tree.xyz, dtype=float)
    edges = np.asarray(tree.connectivity, dtype=int)
    n = xyz.shape[0]

    # adjacency with edge weights (Euclidean length)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    p0 = xyz[edges[:, 0]]
    p1 = xyz[edges[:, 1]]
    w = np.linalg.norm(p0 - p1, axis=1)
    for (a, b), ww in zip(edges, w):
        adj[a].append((b, ww))
        adj[b].append((a, ww))

    root = int(np.asarray(root_idx, int).ravel()[0])
    dist = np.full(n, np.inf, dtype=float)
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

    return dist[np.asarray(pmj_idx, int)]


def _purkinje_pmj_times(
    tree,
    pmj_idx: np.ndarray,
    root_idx: np.ndarray,
    root_t_tree: np.ndarray,
    prev_pmj_tree: np.ndarray,
) -> np.ndarray:
    """
    Unified helper used by run_ecg_core.

    Preferred path:
      - If the tree exposes a fast solver (e.g., Purkinje-UV/JAX) via
        `activate_fim(src_idx, src_vals)`, call it with [root, pmjs] and
        previous PMJ times as warm-start; return PMJ times.

    Fallback:
      - Run Dijkstra on the geometric tree to get path-lengths from the root
        (tree-units, mm). Previous PMJ times are ignored in this fallback.
    """
    # Try the tree-native solver first (matches the notebook loop)
    if hasattr(tree, "activate_fim"):
        try:
            src_idx = np.concatenate(
                [np.asarray(root_idx, int).ravel(), np.asarray(pmj_idx, int).ravel()]
            )
            src_vals = np.concatenate(
                [
                    np.asarray(root_t_tree, float).ravel(),
                    np.asarray(prev_pmj_tree, float).ravel(),
                ]
            )
            out = np.asarray(tree.activate_fim(src_idx, src_vals), dtype=float)
            # Some implementations return [root, pmjs]; others return just pmjs.
            if out.shape[0] == pmj_idx.shape[0] + 1:
                out = out[1:]
            return out
        except Exception:
            # Fall back to pure-NumPy graph SSSP if the fast path misbehaves
            pass

    # Fallback: geometric SSSP in tree space
    return _purkinje_pmj_times_dijkstra(tree, pmj_idx, root_idx)


def run_ecg_core(
    myocardium,
    lv_tree,
    rv_tree,
    lv_root_idx: int,
    rv_root_idx: int,
    *,
    lv_root_time_ms: float = -75.0,
    rv_root_time_ms: float = -75.0,
    purkinje_cv_m_per_s: float = 2.0,
    pvc_idx: Optional[np.ndarray] = None,
    pvc_t: Optional[np.ndarray] = None,
    kmax: int = 8,
    tol_act: float = 1e-6,
    tol_ecg: float = 1e-2,  # <— ECG early-stop like the notebook
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
    trace: bool = False,
    return_diagnostics: bool = False,
    purkinje_engine: str = "jax",  # "jax" (recommended) or "uv" if you have a tree solver
    dedup_pmj_nodes: bool = False,  # <— keep False to match notebook behavior
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Purkinje ↔ Myocardium coupling, as close as possible to the original notebook loop:

      - PMJ times from Purkinje (tree distance / CV + root offset).
      - Myocardium FIM seeded at *every PMJ point* (no dedup), using legacy projection
        (closest-cell analytic update of its four nodes), then full FIM solve, then
        sampling activation back at the PMJ coordinates (return_only_pmjs=True).
      - Clamp per-PMJ with min(Purkinje, Myocardium) in ms.
      - Iterate, and stop early if either:
          (1) no PMJ improved by more than tol_act, or
          (2) relative ECG error < tol_ecg.
    """
    _log = (
        (log if log is not None else (lambda s: print(s, flush=True)))
        if verbose
        else (lambda s: None)
    )
    t0_all = time.perf_counter()

    # --- PMJs (indices & coords) ---
    lv_pmj_idx = np.asarray(getattr(lv_tree, "pmj"), dtype=int)
    rv_pmj_idx = np.asarray(getattr(rv_tree, "pmj"), dtype=int)
    lv_pmj_xyz = np.asarray(lv_tree.xyz, dtype=float)[lv_pmj_idx]
    rv_pmj_xyz = np.asarray(rv_tree.xyz, dtype=float)[rv_pmj_idx]

    # --- optional PVCs (myocardial nodes with fixed times in ms) ---
    pvc_idx = (
        None
        if pvc_idx is None or pvc_t is None or len(pvc_idx) == 0
        else np.asarray(pvc_idx, int)
    )
    pvc_t = None if pvc_idx is None else np.asarray(pvc_t, float)

    # --- Units: 1 m/s == 1 mm/ms
    cv_mm_per_ms = float(purkinje_cv_m_per_s)
    _log(
        f"[core] PMJs: LV={lv_pmj_idx.size}, RV={rv_pmj_idx.size}, total={lv_pmj_idx.size+rv_pmj_idx.size}"
    )
    _log(
        f"[core] CV={cv_mm_per_ms} mm/ms, root times (ms): LV={lv_root_time_ms}, RV={rv_root_time_ms}"
    )

    # --- State in "tree units" (mm) we iterate with; ms companions kept in parallel
    t_lv_tree = np.full(lv_pmj_idx.shape[0], np.inf)
    t_rv_tree = np.full(rv_pmj_idx.shape[0], np.inf)
    t_lv_ms = np.full(lv_pmj_idx.shape[0], np.inf)
    t_rv_ms = np.full(rv_pmj_idx.shape[0], np.inf)

    # roots (as arrays for the helper)
    lv_root_idx = np.asarray([int(lv_root_idx)], dtype=int)
    rv_root_idx = np.asarray([int(rv_root_idx)], dtype=int)
    zero = np.asarray([0.0], dtype=float)

    # ECG early-stop
    prev_ecg_vec = None
    ecg_err_hist: List[float] = []

    trace_data = [] if trace else None
    last_improve_lv = last_improve_rv = None

    for it in range(1, kmax + 1):
        t_it = time.perf_counter()
        _log(f"[iter {it}] start")

        # --- Purkinje pass (tree units via JAX SSSP or the tree's solver)
        t0 = time.perf_counter()
        if purkinje_engine == "jax":
            t_lv_purk_tree = _purkinje_pmj_times(
                lv_tree, lv_pmj_idx, lv_root_idx, zero, t_lv_tree
            )
            t_rv_purk_tree = _purkinje_pmj_times(
                rv_tree, rv_pmj_idx, rv_root_idx, zero, t_rv_tree
            )
        elif purkinje_engine == "uv":
            # If you have a tree-side solver mirroring the notebook, plug it here.
            # Fallback to JAX if not provided.
            t_lv_purk_tree = _purkinje_pmj_times(
                lv_tree, lv_pmj_idx, lv_root_idx, zero, t_lv_tree
            )
            t_rv_purk_tree = _purkinje_pmj_times(
                rv_tree, rv_pmj_idx, rv_root_idx, zero, t_rv_tree
            )
        else:
            raise ValueError(f"Unknown purkinje_engine={purkinje_engine!r}")
        _log(f"[iter {it}] Purkinje pass: {time.perf_counter()-t0:.3f}s")

        # Convert to ms and add root offsets (this is what myocardium expects)
        t_lv_purk_ms = t_lv_purk_tree / cv_mm_per_ms + lv_root_time_ms
        t_rv_purk_ms = t_rv_purk_tree / cv_mm_per_ms + rv_root_time_ms

        # --- Legacy seeding style: pass EVERY PMJ point (no dedup), sample back at PMJ coords
        x0_xyz = np.vstack([lv_pmj_xyz, rv_pmj_xyz])  # (Mlv+Mrv,3)
        x0_vals = np.concatenate([t_lv_purk_ms, t_rv_purk_ms])  # (Mlv+Mrv,)

        # Optional PVCs: append to x0 as additional seeds
        if pvc_idx is not None:
            # PVC positions are in myocardial index space; we need xyz coords
            nodes_xyz = np.asarray(myocardium.xyz, dtype=float)
            x0_xyz = np.vstack([x0_xyz, nodes_xyz[pvc_idx]])
            x0_vals = np.concatenate([x0_vals, pvc_t])

        # --- Myocardium pass: full FIM with legacy projection; sample at PMJ coords
        t0 = time.perf_counter()
        myo_vals_all = myocardium.activate_fim(
            x0=x0_xyz, x0_vals=x0_vals, return_only_pmjs=True
        )
        myo_vals_all = np.asarray(myo_vals_all, dtype=float)  # (Mlv+Mrv,)
        _log(
            f"[iter {it}] Myocardium solve: {time.perf_counter()-t0:.3f}s (n_seeds={x0_xyz.shape[0]})"
        )

        # Split back LV/RV
        Mlv = lv_pmj_idx.size
        t_myo_lv_ms = myo_vals_all[:Mlv]
        t_myo_rv_ms = myo_vals_all[Mlv:]

        # --- Clamp in ms exactly like the notebook
        new_t_lv_ms = np.minimum(t_lv_purk_ms, t_myo_lv_ms)
        new_t_rv_ms = np.minimum(t_rv_purk_ms, t_myo_rv_ms)

        # improvements
        imp_lv = int(np.sum((t_lv_ms - new_t_lv_ms) > tol_act))
        imp_rv = int(np.sum((t_rv_ms - new_t_rv_ms) > tol_act))
        last_improve_lv, last_improve_rv = imp_lv, imp_rv
        _log(
            f"[iter {it}] improvements: LV={imp_lv}, RV={imp_rv}; "
            f"Δmax(ms): LV={np.nanmax(t_lv_ms-new_t_lv_ms):.3e}, RV={np.nanmax(t_rv_ms-new_t_rv_ms):.3e}"
        )

        t_lv_ms, t_rv_ms = new_t_lv_ms, new_t_rv_ms

        # back to tree units for next Purkinje pass (nonnegative clamp)
        t_lv_tree = np.maximum((t_lv_ms - lv_root_time_ms) * cv_mm_per_ms, 0.0)
        t_rv_tree = np.maximum((t_rv_ms - rv_root_time_ms) * cv_mm_per_ms, 0.0)

        # --- Optional trace bundle
        if trace:
            trace_data.append(
                dict(
                    t_lv_purk_tree=t_lv_purk_tree.copy(),
                    t_rv_purk_tree=t_rv_purk_tree.copy(),
                    t_lv_purk_ms=t_lv_purk_ms.copy(),
                    t_rv_purk_ms=t_rv_purk_ms.copy(),
                    t_lv_myo_ms=t_myo_lv_ms.copy(),
                    t_rv_myo_ms=t_myo_rv_ms.copy(),
                    imp_lv=imp_lv,
                    imp_rv=imp_rv,
                )
            )

        # --- ECG early-stop (like the notebook)
        ecg_vec = myocardium.new_get_ecg(record_array=False)  # shape (12, T)
        ecg_vec = np.asarray(ecg_vec, float)
        ecg_err = np.inf
        if prev_ecg_vec is not None:
            num = float(np.linalg.norm(ecg_vec - prev_ecg_vec))
            den = float(np.linalg.norm(prev_ecg_vec)) + 1e-12
            ecg_err = num / den
            ecg_err_hist.append(ecg_err)
            _log(f"[iter {it}] ECG error = {ecg_err:.6g}")
        prev_ecg_vec = ecg_vec

        _log(f"[iter {it}] total {time.perf_counter()-t_it:.3f}s")

        # stop if *either* no PMJ improved or ECG change is small
        if (imp_lv == 0 and imp_rv == 0) or (ecg_err < tol_ecg):
            _log(
                f"[iter {it}] converged "
                f"({'no-improve' if (imp_lv==0 and imp_rv==0) else f'ECG<{tol_ecg}'})"
            )
            break

    # Final structured ECG
    ecg = myocardium.new_get_ecg(record_array=True)

    info: Dict[str, object] = {
        "iterations": it,
        "last_improve_lv": last_improve_lv,
        "last_improve_rv": last_improve_rv,
        "t_lv_pmj": t_lv_ms,
        "t_rv_pmj": t_rv_ms,
        "n_lv_pmj": t_lv_ms.shape[0],
        "n_rv_pmj": t_rv_ms.shape[0],
        "lv_root_time_ms": lv_root_time_ms,
        "rv_root_time_ms": rv_root_time_ms,
        "purkinje_cv_m_per_s": purkinje_cv_m_per_s,
        "elapsed_s": time.perf_counter() - t0_all,
        "ecg_err_hist": ecg_err_hist,
        "trace": trace_data,
    }
    _log(f"[core] done in {info['elapsed_s']:.3f}s; iterations={it}")
    return (ecg, info) if return_diagnostics else (ecg, {})
