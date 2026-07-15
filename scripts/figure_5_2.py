"""
Generate Figure 5.2 outputs:
- 5.2a: Activation map (T(x)) on biventricular geometry
- 5.2b: Simulated 12-lead ECG in QRS window (aligned/normalized)

Modes:
1) Plot from existing activation/ECG npz files.
2) Compute activation/ECG from mesh + fibers + electrodes + Purkinje tree/PMJ files.
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from myocardial_mesh import MyocardialMesh
from vtkmodules.numpy_interface import dataset_adapter as dsa


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def resolve_path(base: Path, user_path: Path | None, rel: str) -> Path:
    if user_path is None:
        return base / rel
    user_path = user_path.expanduser()
    return user_path if user_path.is_absolute() else base / user_path


# -------------------------
# Purkinje graph helpers
# -------------------------


def extract_line_edges_from_vtu(tree: pv.UnstructuredGrid) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for i in range(tree.n_cells):
        cell = tree.GetCell(i)
        if cell.GetNumberOfPoints() == 2:
            a = int(cell.GetPointId(0))
            b = int(cell.GetPointId(1))
            if a != b:
                edges.append((a, b))
    if not edges:
        raise RuntimeError("No 2-point LINE cells found in tree VTU.")
    return edges


def build_adjacency(
    points: np.ndarray, edges: List[Tuple[int, int]]
) -> Dict[int, List[Tuple[int, float]]]:
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(points.shape[0])}
    for a, b in edges:
        w = float(np.linalg.norm(points[a] - points[b]))
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def dijkstra_multi_source(
    adj: Dict[int, List[Tuple[int, float]]],
    sources: Iterable[int],
    n_nodes: int,
) -> np.ndarray:
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
    roots: Iterable[int],
    cv_purk_m_per_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    tree = pv.read(str(tree_vtu))
    pmj = pv.read(str(pmj_vtp))

    pts = np.asarray(tree.points, dtype=np.float32)
    edges = extract_line_edges_from_vtu(tree)
    adj = build_adjacency(pts, edges)
    dist = dijkstra_multi_source(adj, list(roots), pts.shape[0])
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
            f"[WARN] {unreachable}/{pmj_pts.shape[0]} PMJs mapped to unreachable nodes."
        )

    return pmj_pts, pmj_t


def combine_lv_rv_seeds(
    lv_tree_vtu: Path,
    rv_tree_vtu: Path,
    lv_pmj_vtp: Path,
    rv_pmj_vtp: Path,
    lv_roots: Iterable[int],
    rv_roots: Iterable[int],
    cv_purk_m_per_s: float,
    lv_root_time_ms: float,
    rv_root_time_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lv_pts, lv_t = pmj_times_from_tree(
        lv_tree_vtu, lv_pmj_vtp, lv_roots, cv_purk_m_per_s
    )
    rv_pts, rv_t = pmj_times_from_tree(
        rv_tree_vtu, rv_pmj_vtp, rv_roots, cv_purk_m_per_s
    )

    lv_t = lv_t + float(lv_root_time_ms)
    rv_t = rv_t + float(rv_root_time_ms)

    x0_pts = np.vstack([lv_pts, rv_pts]).astype(np.float32)
    x0_vals = np.concatenate([lv_t, rv_t]).astype(np.float32)

    if not np.isfinite(x0_vals).all():
        bad = np.where(~np.isfinite(x0_vals))[0][:20].tolist()
        raise RuntimeError(
            f"Non-finite PMJ times after graph mapping. Example indices: {bad}"
        )

    return x0_pts, x0_vals


def pmj_seeds_zero_times(
    lv_pmj_vtp: Path,
    rv_pmj_vtp: Path,
    lv_root_time_ms: float,
    rv_root_time_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lv_pmj = pv.read(str(lv_pmj_vtp))
    rv_pmj = pv.read(str(rv_pmj_vtp))

    x0_pts = np.vstack([lv_pmj.points, rv_pmj.points]).astype(np.float32)
    lv_t = np.zeros((lv_pmj.n_points,), dtype=np.float32) + float(lv_root_time_ms)
    rv_t = np.zeros((rv_pmj.n_points,), dtype=np.float32) + float(rv_root_time_ms)
    x0_vals = np.concatenate([lv_t, rv_t]).astype(np.float32)

    if x0_pts.shape[0] == 0:
        raise RuntimeError("PMJ VTPs are empty.")
    return x0_pts, x0_vals


# -------------------------
# ECG helpers
# -------------------------


def ecg_struct_to_dict(ecg_struct) -> Dict[str, np.ndarray]:
    names = ecg_struct.dtype.names or []
    raw = {name: np.asarray(ecg_struct[name], float).reshape(-1) for name in names}
    if "I" not in raw and "E1" in raw:
        raw["I"] = raw["E1"]
    if "II" not in raw and "E2" in raw:
        raw["II"] = raw["E2"]
    if "III" not in raw and "E3" in raw:
        raw["III"] = raw["E3"]
    missing = [lead for lead in LEADS_12 if lead not in raw]
    if missing:
        raise RuntimeError(f"Missing leads in ECG: {missing}")
    return {lead: raw[lead] for lead in LEADS_12}


def energy_peak_time(t: np.ndarray, ecg: Dict[str, np.ndarray]) -> float:
    leads = ["II", "V2", "V3", "V4", "V5", "V6"]
    x = np.vstack([ecg[k] for k in leads])
    e = np.sqrt(np.sum(x * x, axis=0))
    idx = int(np.argmax(e))
    return float(t[idx])


def baseline_correct(y: np.ndarray, t_rel: np.ndarray, b0: float, b1: float) -> np.ndarray:
    mask = np.isfinite(y) & (t_rel >= b0) & (t_rel <= b1)
    if mask.sum() < 3:
        return y - np.nanmean(y)
    return y - np.nanmean(y[mask])


def rms(x: np.ndarray) -> float:
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return float("nan")
    return float(np.sqrt(np.mean(x[mask] * x[mask])))


def prepare_qrs_window(
    t: np.ndarray,
    ecg: Dict[str, np.ndarray],
    qrs_pre: float,
    qrs_post: float,
    baseline0: float,
    baseline1: float,
    normalize: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t0 = energy_peak_time(t, ecg)
    t_rel = t - t0
    win = (t_rel >= -qrs_pre) & (t_rel <= qrs_post)
    if win.sum() < 10:
        raise RuntimeError("QRS window too small/empty.")
    t_ms = t_rel[win] * 1000.0

    out: Dict[str, np.ndarray] = {}
    norm = normalize.lower()
    for lead in LEADS_12:
        y = ecg[lead][win].astype(float)
        y_bc = baseline_correct(y, t_rel[win], baseline0, baseline1)
        if norm == "none":
            y_n = y_bc
        elif norm == "rms":
            s = rms(y_bc)
            y_n = y_bc if not np.isfinite(s) or s == 0.0 else y_bc / s
        elif norm == "z":
            mu = float(np.nanmean(y_bc))
            sd = float(np.nanstd(y_bc))
            y_n = y_bc if not np.isfinite(sd) or sd == 0.0 else (y_bc - mu) / sd
        else:
            raise ValueError("normalize must be one of: none, rms, z")
        out[lead] = y_n

    return t_ms, out


def plot_ecg_12lead(
    t_ms: np.ndarray,
    ecg: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    fig, axs = plt.subplots(3, 4, figsize=(10, 7), dpi=140, sharex=True, sharey=True)
    axs = np.asarray(axs)
    for ax, lead in zip(axs.ravel(), LEADS_12):
        ax.plot(t_ms, ecg[lead], linewidth=1.0)
        ax.set_title(lead)
        ax.grid(True, alpha=0.4)
    for ax in axs[-1, :]:
        ax.set_xlabel("Time (ms) rel. peak")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_activation_map(mesh_path: Path, activation: np.ndarray, out_path: Path) -> None:
    mesh = pv.read(str(mesh_path))
    if activation.shape[0] != mesh.n_points:
        raise ValueError(
            f"Activation field size {activation.shape[0]} != mesh points {mesh.n_points}"
        )

    act = np.asarray(activation, dtype=float)
    if not np.isfinite(act).all():
        finite = act[np.isfinite(act)]
        max_finite = float(np.max(finite)) if finite.size else 0.0
        act = act.copy()
        act[~np.isfinite(act)] = max_finite

    mesh.point_data["Activation (ms)"] = act
    surf = mesh.extract_surface().clean()

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("white")
    plotter.add_mesh(
        surf,
        scalars="Activation (ms)",
        cmap="turbo",
        scalar_bar_args={"title": "Activation (ms)"},
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.show(screenshot=str(out_path), auto_close=True)


# -------------------------
# Main
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Figure 5.2a (activation map) and 5.2b (ECG QRS)."
    )
    p.add_argument("--data-root", type=Path, default=Path("data/crtdemo"))
    p.add_argument("--out-dir", type=Path, default=Path("scripts/output"))

    p.add_argument("--mesh", type=Path, default=None, help="Mesh VTK path.")
    p.add_argument("--fibers", type=Path, default=None, help="Fibers VTK path.")
    p.add_argument("--electrodes", type=Path, default=None, help="Electrodes PKL path.")

    p.add_argument("--purkinje-out-dir", type=Path, default=None)
    p.add_argument("--lv-tree-vtu", type=Path, default=None)
    p.add_argument("--rv-tree-vtu", type=Path, default=None)
    p.add_argument("--lv-pmj-vtp", type=Path, default=None)
    p.add_argument("--rv-pmj-vtp", type=Path, default=None)
    p.add_argument("--lv-roots", type=int, nargs="+", default=[742, 984])
    p.add_argument("--rv-roots", type=int, nargs="+", default=[282, 195])
    p.add_argument("--pmj-mode", choices=["tree", "zero"], default="tree")
    p.add_argument("--cv-purk", type=float, default=2.0)
    p.add_argument("--lv-root-time-ms", type=float, default=-75.0)
    p.add_argument("--rv-root-time-ms", type=float, default=-75.0)

    p.add_argument("--activation", type=Path, default=None)
    p.add_argument("--ecg", type=Path, default=None)

    p.add_argument("--sim-dt", type=float, default=0.001)
    p.add_argument("--qrs-pre", type=float, default=0.04)
    p.add_argument("--qrs-post", type=float, default=0.12)
    p.add_argument("--baseline0", type=float, default=-0.04)
    p.add_argument("--baseline1", type=float, default=-0.02)
    p.add_argument("--normalize", choices=["none", "rms", "z"], default="rms")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = (
        args.data_root.expanduser()
        if args.data_root.is_absolute()
        else repo_root / args.data_root
    )
    out_dir = (
        args.out_dir.expanduser()
        if args.out_dir.is_absolute()
        else repo_root / args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = resolve_path(data_root, args.mesh, "crtdemo_mesh_oriented.vtk")

    purk_dir = args.purkinje_out_dir
    if purk_dir is not None:
        purk_dir = purk_dir.expanduser()
        if not purk_dir.is_absolute():
            purk_dir = repo_root / purk_dir

    lv_tree_vtu = args.lv_tree_vtu
    rv_tree_vtu = args.rv_tree_vtu
    lv_pmj_vtp = args.lv_pmj_vtp
    rv_pmj_vtp = args.rv_pmj_vtp
    if purk_dir is not None:
        if lv_tree_vtu is None:
            lv_tree_vtu = purk_dir / "S62_LV_purkinje.vtu"
        if rv_tree_vtu is None:
            rv_tree_vtu = purk_dir / "S62_RV_purkinje.vtu"
        if lv_pmj_vtp is None:
            lv_pmj_vtp = purk_dir / "S62_LV_pmj.vtp"
        if rv_pmj_vtp is None:
            rv_pmj_vtp = purk_dir / "S62_RV_pmj.vtp"

    activation = None
    ecg_dict: Dict[str, np.ndarray] | None = None
    t_ecg: np.ndarray | None = None

    if args.activation is not None:
        act_npz = np.load(args.activation)
        if "sol" in act_npz:
            activation = act_npz["sol"]
        elif "activation" in act_npz:
            activation = act_npz["activation"]
        else:
            raise RuntimeError("Activation npz must contain 'sol' or 'activation'.")

    if args.ecg is not None:
        ecg_npz = np.load(args.ecg)
        if "t" not in ecg_npz:
            raise RuntimeError("ECG npz must contain 't'.")
        t_ecg = np.asarray(ecg_npz["t"], dtype=float)
        raw = {k: ecg_npz[k] for k in ecg_npz.files if k != "t"}
        if "I" not in raw and "E1" in raw:
            raw["I"] = raw["E1"]
        if "II" not in raw and "E2" in raw:
            raw["II"] = raw["E2"]
        if "III" not in raw and "E3" in raw:
            raw["III"] = raw["E3"]
        missing = [lead for lead in LEADS_12 if lead not in raw]
        if missing:
            raise RuntimeError(f"ECG npz missing leads: {missing}")
        ecg_dict = {lead: raw[lead] for lead in LEADS_12}

    if activation is None or ecg_dict is None:
        fibers_path = resolve_path(data_root, args.fibers, "crtdemo_f0_oriented.vtk")
        electrodes_path = resolve_path(
            data_root, args.electrodes, "electrode_pos.pkl"
        )
        if purk_dir is not None and (
            args.mesh is None or args.fibers is None or args.electrodes is None
        ):
            print(
                "[WARN] Using default CRT demo mesh/fibers/electrodes with Purkinje files. "
                "Ensure these inputs refer to the same anatomy."
            )

        myo = MyocardialMesh(
            myo_mesh=str(mesh_path),
            electrodes_position=str(electrodes_path),
            fibers=str(fibers_path),
            device="cpu",
        )

        if activation is None:
            if not (lv_tree_vtu and rv_tree_vtu and lv_pmj_vtp and rv_pmj_vtp):
                raise RuntimeError(
                    "Provide LV/RV tree and PMJ files to compute activation."
                )
            if args.pmj_mode == "tree":
                x0_pts, x0_vals = combine_lv_rv_seeds(
                    Path(lv_tree_vtu),
                    Path(rv_tree_vtu),
                    Path(lv_pmj_vtp),
                    Path(rv_pmj_vtp),
                    lv_roots=args.lv_roots,
                    rv_roots=args.rv_roots,
                    cv_purk_m_per_s=args.cv_purk,
                    lv_root_time_ms=args.lv_root_time_ms,
                    rv_root_time_ms=args.rv_root_time_ms,
                )
            else:
                x0_pts, x0_vals = pmj_seeds_zero_times(
                    Path(lv_pmj_vtp),
                    Path(rv_pmj_vtp),
                    lv_root_time_ms=args.lv_root_time_ms,
                    rv_root_time_ms=args.rv_root_time_ms,
                )
            activation = myo.activate_fim(x0_pts, x0_vals, return_only_pmjs=False)
        else:
            if activation.shape[0] != myo.xyz.shape[0]:
                raise RuntimeError(
                    "Activation size does not match mesh points for ECG synthesis."
                )
            dd = dsa.WrapDataObject(myo.vtk_mesh)
            dd.PointData["activation"][:] = np.asarray(activation, dtype=float)

        if ecg_dict is None:
            ecg_struct = myo.new_get_ecg(record_array=True)
            ecg_dict = ecg_struct_to_dict(ecg_struct)
            n = next(iter(ecg_dict.values())).shape[0]
            t_ecg = np.arange(n, dtype=float) * float(args.sim_dt)

    if activation is None or ecg_dict is None or t_ecg is None:
        raise RuntimeError("Missing activation or ECG data.")

    act_map_path = out_dir / "fig5_2a_activation_map.png"
    plot_activation_map(mesh_path, activation, act_map_path)
    print(f"Saved activation map: {act_map_path}")

    t_ms, ecg_win = prepare_qrs_window(
        t_ecg,
        ecg_dict,
        qrs_pre=args.qrs_pre,
        qrs_post=args.qrs_post,
        baseline0=args.baseline0,
        baseline1=args.baseline1,
        normalize=args.normalize,
    )
    ecg_path = out_dir / "fig5_2b_ecg_qrs.png"
    plot_ecg_12lead(t_ms, ecg_win, ecg_path, "Simulated ECG (QRS window)")
    print(f"Saved ECG plot: {ecg_path}")


if __name__ == "__main__":
    main()
