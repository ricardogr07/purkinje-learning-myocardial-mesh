from __future__ import annotations

import argparse
import heapq
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

from myocardial_mesh import MyocardialMesh


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        for stream in self._streams:
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._streams[0], name)


# =========================
# Path helpers
# =========================


def resolve_path(base: Path, user_path: Path | None, rel: str) -> Path:
    if user_path is None:
        return base / rel
    user_path = user_path.expanduser()
    if user_path.is_absolute():
        return user_path
    return base / user_path


def create_run_dir(base_out_dir: Path) -> Path:
    base_out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M")

    run_idx = 1
    for child in base_out_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(f"{date_str}-"):
            continue
        if "-run" not in name:
            continue
        tail = name.rsplit("-run", 1)[-1]
        try:
            idx = int(tail)
        except ValueError:
            continue
        if idx >= run_idx:
            run_idx = idx + 1

    run_dir = base_out_dir / f"{date_str}-{time_str}-run{run_idx}"
    while run_dir.exists():
        run_idx += 1
        run_dir = base_out_dir / f"{date_str}-{time_str}-run{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# =========================
# Leadfields: volume -> surface
# =========================


def load_biv_map(biv_nod_path: Path) -> np.ndarray:
    """
    Reads S62_BP_structs_2lyr.biv.nod (binary little-endian int32),
    interpreted as pairs [vol_node_id, _]. We use first column.

    Returns:
      vol_ids_for_surface: (N_surface,) int64
    """
    arr = np.fromfile(str(biv_nod_path), dtype="<i4")
    if arr.size % 2 != 0:
        raise RuntimeError(f"Unexpected biv.nod size (not even): {arr.size}")
    arr2 = arr.reshape((-1, 2))
    vol_ids = arr2[:, 0].astype(np.int64)
    if np.any(vol_ids < 0):
        raise RuntimeError("biv.nod contains negative indices (unexpected).")
    return vol_ids


def load_leadfields_projected(
    leadfields_dir: Path,
    mapping_csv: Path,
    vol_ids_for_surface: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Loads electrode leadfields defined on the VOLUME mesh and projects them onto
    the current surface mesh via vol_ids_for_surface.
    """
    df = pd.read_csv(mapping_csv, sep=r"\s+")
    df["elec"] = df["elec"].astype(str).str.strip()
    df = df[df["elec"] != "RL"]

    lf: dict[str, np.ndarray] = {}
    max_id = int(vol_ids_for_surface.max())
    for _, row in df.iterrows():
        elec = row["elec"]
        node = int(row["node"])
        f = leadfields_dir / f"LF_Z_extra_Ref_347195_Field_{node}.dat"

        v_vol = np.loadtxt(f, dtype=np.float32)
        if max_id >= v_vol.size:
            raise RuntimeError(
                f"biv.nod references vol node {max_id}, but leadfield {f.name} has length {v_vol.size}."
            )
        v_surf = v_vol[vol_ids_for_surface]
        lf[elec] = v_surf

    return lf


# =========================
# Purkinje graph + Dijkstra
# =========================


def extract_line_edges_from_vtu(tree: pv.UnstructuredGrid) -> List[Tuple[int, int]]:
    """
    Extract edges from a VTU UnstructuredGrid by iterating cells and selecting 2-point cells (LINE).
    """
    n_cells = tree.n_cells
    edges: List[Tuple[int, int]] = []
    for i in range(n_cells):
        cell = tree.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 2:
            a = int(cell.GetPointId(0))
            b = int(cell.GetPointId(1))
            if a != b:
                edges.append((a, b))
    if not edges:
        raise RuntimeError(
            "No 2-point LINE cells found in tree VTU. Cannot build graph."
        )
    return edges


def build_adjacency(
    points: np.ndarray, edges: List[Tuple[int, int]]
) -> Dict[int, List[Tuple[int, float]]]:
    """Weighted adjacency: weight = Euclidean distance between endpoints (mesh units)."""
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(points.shape[0])}
    for a, b in edges:
        w = float(np.linalg.norm(points[a] - points[b]))
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def dijkstra_multi_source(
    adj: Dict[int, List[Tuple[int, float]]],
    sources: List[int],
    n_nodes: int,
) -> np.ndarray:
    """Multi-source Dijkstra. Returns distances (inf if unreachable)."""
    dist = np.full((n_nodes,), np.inf, dtype=np.float64)
    pq: List[Tuple[float, int]] = []

    for s in sources:
        if 0 <= s < n_nodes:
            dist[s] = 0.0
            heapq.heappush(pq, (0.0, s))

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist


def pmj_times_from_tree(
    tree_vtu: Path,
    pmj_vtp: Path,
    roots: List[int],
    cv_purk_m_per_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pmj_pts: (N,3) float32
      pmj_t:   (N,)  float32   times in ms (assuming coords in mm => m/s == mm/ms numerically)
    """
    tree = pv.read(str(tree_vtu))
    pmj = pv.read(str(pmj_vtp))

    pts = np.asarray(tree.points, dtype=np.float32)
    edges = extract_line_edges_from_vtu(tree)
    adj = build_adjacency(pts, edges)

    dist = dijkstra_multi_source(adj, roots, pts.shape[0])
    time_nodes_ms = dist / float(cv_purk_m_per_s)

    pmj_pts = np.asarray(pmj.points, dtype=np.float32)
    pmj_t = np.empty((pmj_pts.shape[0],), dtype=np.float32)

    unreachable = 0
    for i, p in enumerate(pmj_pts):
        idx = int(tree.find_closest_point(p))
        t = float(time_nodes_ms[idx])
        if not np.isfinite(t):
            unreachable += 1
            t = 0.0
        pmj_t[i] = t

    if unreachable:
        print(
            f"[WARN] {unreachable}/{pmj_pts.shape[0]} PMJs mapped to unreachable nodes (time set to 0)."
        )

    return pmj_pts, pmj_t


def combine_lv_rv_seeds(
    lv_tree_vtu: Path,
    rv_tree_vtu: Path,
    lv_pmj_vtp: Path,
    rv_pmj_vtp: Path,
    lv_roots: List[int],
    rv_roots: List[int],
    cv_purk_m_per_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lv_pts, lv_t = pmj_times_from_tree(
        lv_tree_vtu, lv_pmj_vtp, lv_roots, cv_purk_m_per_s
    )
    rv_pts, rv_t = pmj_times_from_tree(
        rv_tree_vtu, rv_pmj_vtp, rv_roots, cv_purk_m_per_s
    )

    x0_pts = np.vstack([lv_pts, rv_pts]).astype(np.float32)
    x0_vals = np.concatenate([lv_t, rv_t]).astype(np.float32)

    if not np.isfinite(x0_vals).all():
        bad = np.where(~np.isfinite(x0_vals))[0][:20].tolist()
        raise RuntimeError(
            f"Non-finite PMJ times after graph mapping. Example indices: {bad}"
        )

    return x0_pts, x0_vals


def pmj_seeds_zero_times(
    lv_pmj_vtp: Path, rv_pmj_vtp: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x0_pts:  (N,3) float32
      x0_vals: (N,)  float32  (all zeros)
    """
    lv_pmj = pv.read(str(lv_pmj_vtp))
    rv_pmj = pv.read(str(rv_pmj_vtp))

    x0_pts = np.vstack([lv_pmj.points, rv_pmj.points]).astype(np.float32)
    x0_vals = np.zeros((x0_pts.shape[0],), dtype=np.float32)

    if x0_pts.shape[0] == 0:
        raise RuntimeError("PMJ VTPs are empty.")
    return x0_pts, x0_vals


# =========================
# Metrics (QRS)
# =========================


def load_observed_json(obs_json: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Expected structure:
      {
        "t": [ ... seconds ... ],
        "ecg": { "I": [...], "II": [...], ..., "V6": [...] }
      }
    """
    with obs_json.open("r", encoding="utf-8") as f:
        d = json.load(f)

    if "t" not in d or "ecg" not in d:
        raise KeyError("Observed JSON must contain keys: 't' and 'ecg'.")

    t = np.asarray(d["t"], dtype=float)
    ecg_raw = d["ecg"]
    if not isinstance(ecg_raw, dict):
        raise TypeError("Observed JSON field 'ecg' must be a dict of lead -> array.")

    ecg = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in ecg_raw.items()}

    for k in LEADS_12:
        if k not in ecg:
            raise KeyError(
                f"Observed ECG missing lead '{k}'. Found: {list(ecg.keys())}"
            )
        if ecg[k].shape[0] != t.shape[0]:
            raise ValueError(
                f"Observed lead '{k}' length {ecg[k].shape[0]} != t length {t.shape[0]}"
            )

    return t, ecg


def prepare_simulated(
    sim: Dict[str, np.ndarray], sim_dt: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not sim:
        raise RuntimeError("Simulated ECG dict is empty.")

    lengths = {k: np.asarray(v).reshape(-1).shape[0] for k, v in sim.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Simulated ECG leads have mismatched lengths: {lengths}")

    n = next(iter(lengths.values()))
    t = np.arange(n, dtype=float) * float(sim_dt)

    sig = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in sim.items()}

    if "I" not in sig and "E1" in sig:
        mapping = {"E1": "I", "E2": "II", "E3": "III"}
        for src, dst in mapping.items():
            if src in sig and dst not in sig:
                sig[dst] = sig[src]

    sim_out = {}
    for k in LEADS_12:
        if k not in sig:
            raise KeyError(
                f"Simulated ECG missing lead '{k}'. Found: {list(sig.keys())}"
            )
        if sig[k].shape[0] != t.shape[0]:
            raise ValueError(
                f"Sim lead '{k}' length {sig[k].shape[0]} != t length {t.shape[0]}"
            )
        sim_out[k] = sig[k]

    return t, sim_out


def energy_peak_time(
    t: np.ndarray, ecg: Dict[str, np.ndarray], leads: List[str] | None = None
) -> float:
    """Peak time by multi-lead energy: sqrt(sum(lead^2))."""
    if leads is None:
        leads = ["II", "V2", "V3", "V4", "V5", "V6"]
    x = np.vstack([ecg[k] for k in leads])
    e = np.sqrt(np.sum(x * x, axis=0))
    idx = int(np.argmax(e))
    return float(t[idx])


def interp_to(t_src: np.ndarray, y_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Linear interpolation; out of range -> NaN."""
    return np.interp(t_tgt, t_src, y_src, left=np.nan, right=np.nan)


def baseline_correct(
    y: np.ndarray, t_rel: np.ndarray, b0: float, b1: float
) -> np.ndarray:
    """Subtract mean in baseline window [b0, b1] seconds relative to peak."""
    mask = np.isfinite(y) & (t_rel >= b0) & (t_rel <= b1)
    if mask.sum() < 3:
        return y - np.nanmean(y)
    return y - np.nanmean(y[mask])


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    d = x[mask] - y[mask]
    return float(np.sqrt(np.mean(d * d)))


def rms(x: np.ndarray) -> float:
    mask = np.isfinite(x)
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean(x[mask] * x[mask])))


def compute_alpha_lsq(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    """alpha = argmin || y_obs - alpha*y_sim ||^2, finite-safe."""
    mask = np.isfinite(y_obs) & np.isfinite(y_sim)
    if mask.sum() < 5:
        return 1.0
    denom = float(np.dot(y_sim[mask], y_sim[mask]))
    if denom <= 0.0:
        return 1.0
    return float(np.dot(y_obs[mask], y_sim[mask]) / denom)


def compute_metrics(
    t_obs: np.ndarray,
    obs: Dict[str, np.ndarray],
    t_sim: np.ndarray,
    sim: Dict[str, np.ndarray],
    qrs_pre: float,
    qrs_post: float,
    baseline0: float,
    baseline1: float,
    normalize: str,
    lsq_scale: bool,
) -> Tuple[
    np.ndarray,
    Dict[str, Dict[str, float]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """
    Returns:
      t_rel_win (seconds)
      metrics dict per lead
      obs_win (processed) per lead
      sim_win (processed, scaled) per lead
    """
    t0_obs = energy_peak_time(t_obs, obs)
    t0_sim = energy_peak_time(t_sim, sim)

    t_rel_obs = t_obs - t0_obs
    t_rel_sim = t_sim - t0_sim

    win_mask = (t_rel_obs >= -qrs_pre) & (t_rel_obs <= qrs_post)
    if win_mask.sum() < 10:
        raise RuntimeError("QRS window too small/empty. Adjust --qrs-pre/--qrs-post.")
    t_rel_win = t_rel_obs[win_mask]

    metrics: Dict[str, Dict[str, float]] = {}
    obs_out: Dict[str, np.ndarray] = {}
    sim_out: Dict[str, np.ndarray] = {}

    for lead in LEADS_12:
        y_obs = obs[lead][win_mask].astype(float)

        y_sim_interp = interp_to(t_rel_sim, sim[lead].astype(float), t_rel_win)

        y_obs_bc = baseline_correct(y_obs, t_rel_win, baseline0, baseline1)
        y_sim_bc = baseline_correct(y_sim_interp, t_rel_win, baseline0, baseline1)

        if lsq_scale:
            alpha = compute_alpha_lsq(y_obs_bc, y_sim_bc)
        else:
            alpha = 1.0
        y_sim_scaled = alpha * y_sim_bc

        nrm = normalize.lower()
        if nrm == "none":
            y_obs_n = y_obs_bc
            y_sim_n = y_sim_scaled
        elif nrm == "rms":
            s_obs = rms(y_obs_bc)
            if not np.isfinite(s_obs) or s_obs == 0.0:
                y_obs_n = y_obs_bc
                y_sim_n = y_sim_scaled
            else:
                y_obs_n = y_obs_bc / s_obs
                y_sim_n = y_sim_scaled / s_obs
        elif nrm == "z":
            mu = np.nanmean(y_obs_bc)
            sd = np.nanstd(y_obs_bc)
            if not np.isfinite(sd) or sd == 0.0:
                y_obs_n = y_obs_bc
                y_sim_n = y_sim_scaled
            else:
                y_obs_n = (y_obs_bc - mu) / sd
                y_sim_n = (y_sim_scaled - mu) / sd
        else:
            raise ValueError("normalize must be one of: none, rms, z")

        rho = pearson_r(y_obs_bc, y_sim_scaled)

        metrics[lead] = {
            "alpha": alpha,
            "rmse_raw": rmse(y_obs_bc, y_sim_scaled),
            "rmse_norm": rmse(y_obs_n, y_sim_n),
            "rho": rho,
            "abs_rho": abs(rho) if np.isfinite(rho) else float("nan"),
        }

        obs_out[lead] = y_obs_n
        sim_out[lead] = y_sim_n

    return t_rel_win, metrics, obs_out, sim_out


def save_metrics_csv(out_csv: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    lines = ["lead,alpha,rmse_raw,rmse_norm,rho,abs_rho"]
    for lead in LEADS_12:
        m = metrics[lead]
        lines.append(
            f"{lead},{m['alpha']:.6g},{m['rmse_raw']:.6g},{m['rmse_norm']:.6g},{m['rho']:.6g},{m['abs_rho']:.6g}"
        )
    out_csv.write_text("\n".join(lines), encoding="utf-8")


def plot_overlay(
    out_png: Path,
    t_ms: np.ndarray,
    obs: Dict[str, np.ndarray],
    sim: Dict[str, np.ndarray],
    title: str,
) -> None:
    show = ["II", "V1", "V2", "V5", "V6", "aVR"]
    plt.figure(figsize=(10, 6))
    for lead in show:
        plt.plot(t_ms, obs[lead], label=f"{lead} obs", linewidth=2)
        plt.plot(t_ms, sim[lead], label=f"{lead} sim", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel("Time (ms) relative to peak")
    plt.ylabel("Amplitude (baseline-corrected; normalized if enabled)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_top3(
    out_png: Path,
    t_ms: np.ndarray,
    obs: Dict[str, np.ndarray],
    sim: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    vals = []
    for lead in LEADS_12:
        r = metrics[lead]["abs_rho"]
        if np.isfinite(r):
            vals.append((r, lead))
    vals.sort(reverse=True)
    top = [x[1] for x in vals[:3]] if vals else ["II", "V2", "V6"]

    plt.figure(figsize=(9, 5))
    for lead in top:
        plt.plot(t_ms, obs[lead], label=f"{lead} obs", linewidth=2)
        plt.plot(t_ms, sim[lead], label=f"{lead} sim", linestyle="--", linewidth=2)
    plt.title(f"Top-3 leads by |rho|: {', '.join(top)}")
    plt.xlabel("Time (ms) relative to peak")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def summarize(metrics: Dict[str, Dict[str, float]]) -> str:
    def _mean(key: str) -> float:
        xs = [metrics[k][key] for k in LEADS_12 if np.isfinite(metrics[k][key])]
        return float(np.mean(xs)) if xs else float("nan")

    mean_rmse_raw = _mean("rmse_raw")
    mean_rmse_norm = _mean("rmse_norm")
    mean_rho = _mean("rho")
    mean_abs_rho = _mean("abs_rho")

    top = sorted(
        LEADS_12,
        key=lambda k: metrics[k]["abs_rho"]
        if np.isfinite(metrics[k]["abs_rho"])
        else -1.0,
        reverse=True,
    )[:3]

    top_str = ", ".join(
        [
            f"{k}:rho={metrics[k]['rho']:.3f},alpha={metrics[k]['alpha']:.3g}"
            for k in top
        ]
    )

    return (
        f"Mean RMSE(raw)={mean_rmse_raw:.6g} | Mean RMSE(norm)={mean_rmse_norm:.6g} | "
        f"Mean rho={mean_rho:.6g} | Mean |rho|={mean_abs_rho:.6g}\n"
        f"Top-3 by |rho|: {top_str}"
    )


# =========================
# Main
# =========================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Myocardial mesh smoke test: forward ECG + QRS metrics."
    )
    p.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Base data folder (Karli dataset root).",
    )
    p.add_argument(
        "--obs-json",
        type=Path,
        required=True,
        help="Observed ECG mean-beat JSON (12-lead).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder (base; run subfolder created inside, default: scripts/output).",
    )

    p.add_argument(
        "--pmj-mode", choices=["tree", "zero"], default="tree", help="PMJ timing mode."
    )
    p.add_argument(
        "--cv-purk", type=float, default=2.0, help="Purkinje conduction velocity (m/s)."
    )
    p.add_argument(
        "--lv-roots",
        type=int,
        nargs="+",
        default=[742, 984],
        help="LV root node ids in LV tree VTU.",
    )
    p.add_argument(
        "--rv-roots",
        type=int,
        nargs="+",
        default=[282, 195],
        help="RV root node ids in RV tree VTU.",
    )

    p.add_argument(
        "--metrics-mode", choices=["lsq", "plain"], default="lsq", help="Metrics mode."
    )
    p.add_argument(
        "--normalize",
        choices=["none", "rms", "z"],
        default="rms",
        help="Normalization for RMSE/plots.",
    )
    p.add_argument(
        "--sim-dt",
        type=float,
        default=0.001,
        help="Seconds per sample for simulated ECG.",
    )
    p.add_argument("--qrs-pre", type=float, default=0.04)
    p.add_argument("--qrs-post", type=float, default=0.12)
    p.add_argument("--baseline0", type=float, default=-0.04)
    p.add_argument("--baseline1", type=float, default=-0.02)
    p.add_argument(
        "--npz-name",
        type=str,
        default=None,
        help="Output NPZ name (default uses pmj-mode).",
    )
    p.add_argument("--skip-plots", action="store_true", help="Skip PNG plots.")

    p.add_argument("--mesh-vtk", type=Path, default=None)
    p.add_argument("--fibers-vtk", type=Path, default=None)
    p.add_argument("--biv-nod", type=Path, default=None)
    p.add_argument("--leadfield-dir", type=Path, default=None)
    p.add_argument("--leadfield-map", type=Path, default=None)
    p.add_argument("--lv-tree-vtu", type=Path, default=None)
    p.add_argument("--rv-tree-vtu", type=Path, default=None)
    p.add_argument("--lv-pmj-vtp", type=Path, default=None)
    p.add_argument("--rv-pmj-vtp", type=Path, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = args.base.expanduser().resolve()
    obs_json = args.obs_json.expanduser()
    if not obs_json.is_absolute():
        obs_json = base / obs_json

    base_out_dir = (
        args.out_dir.expanduser()
        if args.out_dir
        else Path(__file__).resolve().parent / "output"
    )
    out_dir = create_run_dir(base_out_dir)

    log_path = out_dir / "smoke_myocardial_mesh.log"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Run output dir: {out_dir}")
    print(f"Log file: {log_path}")

    mesh_vtk = resolve_path(
        base, args.mesh_vtk, "S62_BP_structs_2lyr_mesh_oriented.vtk"
    )
    fibers_vtk = resolve_path(
        base, args.fibers_vtk, "S62_BP_structs_2lyr_f0_oriented.vtk"
    )
    biv_nod = resolve_path(base, args.biv_nod, "S62_BP_structs_2lyr.biv.nod")
    leadf_dir = resolve_path(base, args.leadfield_dir, "leadfields")
    leadf_map = resolve_path(
        base, args.leadfield_map, "S62_BP_structs_2lyr.torso_ecg_locs.header.nodes.csv"
    )

    lv_tree_vtu = resolve_path(base, args.lv_tree_vtu, "out_s62/S62_LV_purkinje.vtu")
    rv_tree_vtu = resolve_path(base, args.rv_tree_vtu, "out_s62/S62_RV_purkinje.vtu")
    lv_pmj_vtp = resolve_path(base, args.lv_pmj_vtp, "out_s62/S62_LV_pmj.vtp")
    rv_pmj_vtp = resolve_path(base, args.rv_pmj_vtp, "out_s62/S62_RV_pmj.vtp")

    vol_ids = load_biv_map(biv_nod)
    print(
        f"biv.nod surface points: {vol_ids.size}  vol_id min/max: "
        f"{int(vol_ids.min())} / {int(vol_ids.max())}"
    )

    lead_fields = load_leadfields_projected(leadf_dir, leadf_map, vol_ids)
    print("Leadfield keys:", list(lead_fields.keys()))
    print("Leadfield (projected) sizes:", {k: v.size for k, v in lead_fields.items()})

    if args.pmj_mode == "tree":
        x0_pts, x0_vals = combine_lv_rv_seeds(
            lv_tree_vtu,
            rv_tree_vtu,
            lv_pmj_vtp,
            rv_pmj_vtp,
            lv_roots=list(args.lv_roots),
            rv_roots=list(args.rv_roots),
            cv_purk_m_per_s=args.cv_purk,
        )
    else:
        x0_pts, x0_vals = pmj_seeds_zero_times(lv_pmj_vtp, rv_pmj_vtp)

    print(
        f"PMJs total: {x0_pts.shape[0]}  seed times (ms) min/max: "
        f"{float(x0_vals.min()):.3f} / {float(x0_vals.max()):.3f} "
        f"(mode={args.pmj_mode})"
    )

    myo = MyocardialMesh(
        myo_mesh=str(mesh_vtk),
        electrodes_position=None,
        fibers=str(fibers_vtk),
        device="cpu",
        conductivity_params=None,
        lead_fields_dict=lead_fields,
    )

    n_nodes = np.asarray(myo.xyz).shape[0]
    for k, v in lead_fields.items():
        if v.size != n_nodes:
            raise RuntimeError(f"Mismatch {k}: {v.size} vs mesh points {n_nodes}")
    print("Myocardium mesh points:", n_nodes)

    sol = myo.activate_fim(x0_pts, x0_vals, return_only_pmjs=False)
    print("Activation field min/max:", float(np.min(sol)), float(np.max(sol)))

    ecg = myo.new_get_ecg(record_array=True)
    sim = {name: np.asarray(ecg[name]).reshape(-1) for name in ecg.dtype.names}

    npz_name = args.npz_name or f"out_forward_ecg_{args.pmj_mode}.npz"
    out_npz = out_dir / npz_name
    t_sim = np.arange(next(iter(sim.values())).shape[0], dtype=float) * float(
        args.sim_dt
    )
    np.savez(out_npz, t=t_sim, **sim)
    print("Saved:", out_npz)

    t_obs, obs = load_observed_json(obs_json)
    t_sim, sim_12 = prepare_simulated(sim, sim_dt=args.sim_dt)

    t_rel_win, metrics, obs_win, sim_win = compute_metrics(
        t_obs=t_obs,
        obs=obs,
        t_sim=t_sim,
        sim=sim_12,
        qrs_pre=args.qrs_pre,
        qrs_post=args.qrs_post,
        baseline0=args.baseline0,
        baseline1=args.baseline1,
        normalize=args.normalize,
        lsq_scale=(args.metrics_mode == "lsq"),
    )

    metrics_name = (
        "metrics_qrs_lsq.csv" if args.metrics_mode == "lsq" else "metrics_qrs.csv"
    )
    out_csv = out_dir / metrics_name
    save_metrics_csv(out_csv, metrics)

    if not args.skip_plots:
        t_ms = t_rel_win * 1000.0
        overlay_name = (
            "overlay_obs_vs_sim_lsq.png"
            if args.metrics_mode == "lsq"
            else "overlay_obs_vs_sim.png"
        )
        top_name = (
            "top3_by_abs_rho_lsq.png"
            if args.metrics_mode == "lsq"
            else "top3_by_abs_rho.png"
        )
        plot_overlay(
            out_dir / overlay_name,
            t_ms,
            obs_win,
            sim_win,
            "Observed vs Simulated (QRS window)",
        )
        plot_top3(out_dir / top_name, t_ms, obs_win, sim_win, metrics)
        print("Saved:", out_dir / overlay_name)
        print("Saved:", out_dir / top_name)

    print("Saved:", out_csv)
    print(summarize(metrics))
    log_file.close()


if __name__ == "__main__":
    main()
