"""Build a myocardial mesh, embed a Purkinje UV tree, and compute a lead field."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Tuple

import numpy as np
import pyvista as pv

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from myocardial_mesh import MyocardialMesh  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "crtdemo"
DEFAULT_TREE_PATH = DEFAULT_DATA_DIR / "nb" / "True_LVtree.vtu"
DEFAULT_SCREENSHOT = Path(__file__).resolve().parent / "build_mesh_and_lead_field.png"


@dataclass(frozen=True)
class PurkinjeUVTree:
    """Minimal view of a Purkinje UV tree loaded from VTK."""

    xyz: np.ndarray
    edges: np.ndarray
    pmj_idx: np.ndarray


def load_uv_tree(tree_path: Path) -> PurkinjeUVTree:
    """Load a Purkinje UV tree saved as a VTK unstructured grid.

    Args:
        tree_path: Path to a ``.vtu`` file produced by the UV toolchain.

    Returns:
        PurkinjeUVTree: Points, edge connectivity, and PMJ indices (leaf nodes).
    """
    grid = pv.read(tree_path)
    xyz = np.asarray(grid.points, dtype=float)

    # Each cell is [2, a, b] for line elements (VTK_LINE → id=3)
    cells = np.asarray(grid.cells, dtype=int).reshape(-1, 3)[:, 1:]
    deg = np.zeros(xyz.shape[0], dtype=int)
    for a, b in cells:
        deg[a] += 1
        deg[b] += 1
    pmj_idx = np.flatnonzero(deg == 1)  # leaves act as PMJs for this demo

    LOGGER.info(
        "Loaded UV tree: %d nodes, %d edges, %d PMJs (leaves).",
        xyz.shape[0],
        cells.shape[0],
        pmj_idx.size,
    )
    return PurkinjeUVTree(xyz=xyz, edges=cells, pmj_idx=pmj_idx)


def map_pmjs_to_myocardium(
    pmj_xyz: np.ndarray, myocardium: MyocardialMesh
) -> Tuple[np.ndarray, np.ndarray]:
    """Snap PMJ coordinates to the closest myocardial mesh nodes.

    Args:
        pmj_xyz: PMJ coordinates from the Purkinje tree (P, 3).
        myocardium: Instantiated ``MyocardialMesh``.

    Returns:
        Tuple of (node_indices, node_xyz) on the myocardium.
    """
    grid = pv.UnstructuredGrid(myocardium.vtk_mesh)
    indices = np.array(
        [grid.find_closest_point(p) for p in np.asarray(pmj_xyz, dtype=float)],
        dtype=int,
    )
    unique_idx = np.unique(indices)
    snapped_xyz = np.asarray(myocardium.xyz, dtype=float)[unique_idx]
    LOGGER.info(
        "Mapped %d PMJs to %d unique myocardial nodes.",
        pmj_xyz.shape[0],
        unique_idx.size,
    )
    return unique_idx, snapped_xyz


def plot_activation(
    myocardium: MyocardialMesh,
    pmj_nodes: Iterable[int],
    screenshot_path: Path,
    *,
    clim: Tuple[float, float] | None = None,
) -> Path:
    """Plot the activation field with PMJs highlighted and save a screenshot."""
    grid = pv.UnstructuredGrid(myocardium.vtk_mesh)

    act = np.asarray(grid.point_data["activation"], dtype=float)
    finite_mask = np.isfinite(act)
    if not finite_mask.any():
        raise RuntimeError("Activation field is empty; run activate_fim first.")

    if clim is None:
        finite_vals = act[finite_mask]
        clim = (float(np.min(finite_vals)), float(np.max(finite_vals)))

    pmj_array = np.asarray(list(pmj_nodes), dtype=int)

    def _matplotlib_fallback() -> Path:
        LOGGER.warning(
            "PyVista off-screen rendering is unavailable; using Matplotlib fallback."
        )
        import matplotlib.pyplot as plt

        pts = np.asarray(grid.points, dtype=float)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=act, cmap="plasma", s=2, alpha=0.9)
        ax.scatter(
            pts[pmj_array, 0],
            pts[pmj_array, 1],
            c="white",
            edgecolors="black",
            s=20,
            label="PMJs",
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Activation field (fallback view)")
        ax.legend(loc="upper right")
        fig.colorbar(sc, ax=ax, label="Activation (ms)")
        fig.tight_layout()
        fig.savefig(screenshot_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved fallback screenshot to %s", screenshot_path.resolve())
        return screenshot_path

    if not pv.system_supports_plotting():  # pragma: no cover - environment-dependent
        screenshot_path = Path(screenshot_path)
        return _matplotlib_fallback()

    try:
        pv.start_xvfb()
    except OSError as exc:  # pragma: no cover - environment-dependent
        LOGGER.warning(
            "Xvfb is unavailable (%s); falling back to default off-screen path.", exc
        )

    plotter = pv.Plotter(off_screen=True, window_size=(960, 720))
    plotter.add_mesh(
        grid,
        scalars="activation",
        cmap="plasma",
        clim=clim,
        opacity=0.85,
        show_edges=False,
    )
    pmj_pts = grid.points[pmj_array]
    plotter.add_points(
        pmj_pts, color="white", point_size=10, render_points_as_spheres=True
    )
    plotter.add_title("Activation field with PMJs", font_size=10)

    screenshot_path = Path(screenshot_path)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.show(screenshot=str(screenshot_path), auto_close=True)
        LOGGER.info("Saved screenshot to %s", screenshot_path.resolve())
        return screenshot_path
    except Exception as exc:  # pragma: no cover - environment-dependent
        LOGGER.warning(
            "PyVista rendering failed (%s); falling back to a Matplotlib snapshot.", exc
        )
        return _matplotlib_fallback()


def run_example(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    tree_path: Path = DEFAULT_TREE_PATH,
    sample_pmjs: int = 256,
    screenshot: Path | None = DEFAULT_SCREENSHOT,
    skip_plot: bool = False,
) -> dict:
    """End-to-end example: load mesh, embed tree, solve, and (optionally) plot."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    myo = MyocardialMesh(
        myo_mesh=str(data_dir / "crtdemo_mesh_oriented.vtk"),
        electrodes_position=str(data_dir / "electrode_pos.pkl"),
        fibers=str(data_dir / "crtdemo_f0_oriented.vtk"),
        device="cpu",
    )

    tree = load_uv_tree(tree_path)
    rng = np.random.default_rng(0)
    pmj_idx = (
        rng.choice(tree.pmj_idx, size=sample_pmjs, replace=False)
        if sample_pmjs and sample_pmjs < tree.pmj_idx.size
        else tree.pmj_idx
    )
    pmj_xyz = tree.xyz[pmj_idx]
    pmj_nodes, pmj_nodes_xyz = map_pmjs_to_myocardium(pmj_xyz, myocardium=myo)

    activation = myo.activate_fim(
        x0=pmj_nodes_xyz,
        x0_vals=np.zeros(pmj_nodes_xyz.shape[0], dtype=float),
        return_only_pmjs=False,
    )
    LOGGER.info(
        "Solved activation: min=%.3f ms, max=%.3f ms",
        float(np.min(activation[np.isfinite(activation)])),
        float(np.max(activation[np.isfinite(activation)])),
    )

    lead_field = myo.get_lead_field()
    ecg = myo.new_get_ecg(record_array=True)
    LOGGER.info(
        "Computed ECG with %d leads over %d samples.", len(ecg.dtype.names), len(ecg)
    )

    shot = None
    if screenshot and not skip_plot:
        shot = plot_activation(myo, pmj_nodes, screenshot)

    return {
        "myocardium": myo,
        "tree": tree,
        "pmj_nodes": pmj_nodes,
        "activation": activation,
        "lead_field": lead_field,
        "ecg": ecg,
        "screenshot": shot,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build myocardial mesh, embed a Purkinje UV tree, and compute lead fields."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing crtdemo meshes and electrodes.",
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=DEFAULT_TREE_PATH,
        help="Path to a Purkinje UV tree (.vtu).",
    )
    parser.add_argument(
        "--sample-pmjs",
        type=int,
        default=256,
        help="Subsample this many PMJs for seeding (use all if <=0 or above total).",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=DEFAULT_SCREENSHOT,
        help="Path for the activation screenshot (ignored if --skip-plot is set).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip PyVista plotting (useful for headless smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_example(
        data_dir=args.data_dir,
        tree_path=args.tree,
        sample_pmjs=args.sample_pmjs,
        screenshot=args.screenshot,
        skip_plot=args.skip_plot,
    )


if __name__ == "__main__":
    main()
