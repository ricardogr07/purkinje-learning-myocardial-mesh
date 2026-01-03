"""JAX-based Purkinje tree propagation utilities.

This module provides a lightweight JAX graph view of a Purkinje tree and
functions to compute shortest-path activation times using scatter-min
relaxation sweeps (JAX-jitted).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp
    from jax import Array
except Exception as e:
    raise RuntimeError(
        "JAX is required for the JAX Purkinje propagator. "
        "Install jax (and jaxlib) first."
    ) from e

_LOGGER = logging.getLogger(__name__)


@dataclass
class JaxPurkinjeGraph:
    """Lightweight JAX graph view of a Purkinje tree."""

    xyz: Array  # (N, 3)
    edges_src: Array  # (E,)
    edges_dst: Array  # (E,)
    pmj_idx: Array  # (P,)

    @classmethod
    def from_purkinje_tree(cls, tree: Any) -> JaxPurkinjeGraph:
        """Construct a JAX graph representation from a Purkinje tree object.

        Args:
            tree: Object exposing ``xyz`` and either ``connectivity`` or ``edges``.
                Optionally ``pmj`` for explicit PMJ indices.

        Returns:
            JaxPurkinjeGraph: Graph view with JAX arrays for nodes, edges, and PMJs.
        """
        xyz_np = np.asarray(getattr(tree, "xyz"), dtype=float)
        xyz = jnp.asarray(xyz_np)

        # Edges: try common attribute names, fall back to connectivity
        if hasattr(tree, "connectivity"):
            conn_np = np.asarray(tree.connectivity, dtype=int)
        elif hasattr(tree, "edges"):
            conn_np = np.asarray(tree.edges, dtype=int)
        else:
            raise ValueError("PurkinjeTree must expose 'connectivity' (E,2)")

        # Make undirected & deduplicated
        a = np.minimum(conn_np[:, 0], conn_np[:, 1])
        b = np.maximum(conn_np[:, 0], conn_np[:, 1])
        und = np.unique(np.stack([a, b], axis=1), axis=0)

        src = jnp.asarray(und[:, 0])
        dst = jnp.asarray(und[:, 1])

        # PMJs: prefer explicit 'pmj', else compute leaves (degree==1)
        if hasattr(tree, "pmj"):
            pmj = jnp.asarray(np.asarray(tree.pmj, dtype=int))
        else:
            n = xyz_np.shape[0]
            deg = np.zeros(n, dtype=int)
            for s, d in und:
                deg[s] += 1
                deg[d] += 1
            pmj = jnp.asarray(np.where(deg == 1)[0].astype(int))

        _LOGGER.debug(
            "Created JaxPurkinjeGraph: %d nodes, %d edges, %d PMJs",
            xyz.shape[0],
            src.shape[0],
            pmj.shape[0],
        )
        return cls(xyz=xyz, edges_src=src, edges_dst=dst, pmj_idx=pmj)


def _edge_lengths_mm(xyz: Array, src: Array, dst: Array) -> Array:
    """Compute Euclidean edge lengths in millimeters."""
    # jnp.linalg.norm has incomplete type info; cast to Array for mypy.
    return cast(Array, jnp.linalg.norm(xyz[dst] - xyz[src], axis=1))


@jax.jit  # type: ignore[misc]
def _relax_times(times: Array, src: Array, dst: Array, w: Array) -> Array:
    """One symmetric relaxation sweep over undirected edges using scatter-min."""
    cand_dst = times[src] + w
    cand_src = times[dst] + w
    t2 = times.at[dst].min(cand_dst)
    t2 = t2.at[src].min(cand_src)
    return t2


def sssp_times_ms(
    graph: JaxPurkinjeGraph,
    root_idx: int,
    root_time_ms: float,
    cv_mm_per_ms: float,
    *,
    pmj_fixed_ms: Optional[NDArray[np.float64]] = None,
    tol_ms: float = 1e-6,
    max_steps: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute single-source shortest-path times in milliseconds.

    Runs symmetric relaxation sweeps over the undirected tree edges until
    convergence or until the maximum number of steps is reached.

    Args:
        graph: JAX Purkinje graph.
        root_idx: Root node index.
        root_time_ms: Root time offset in ms.
        cv_mm_per_ms: Conduction velocity in mm/ms (1 m/s = 1 mm/ms).
        pmj_fixed_ms: Optional fixed PMJ activation times (ms).
        tol_ms: Convergence tolerance (ms).
        max_steps: Optional maximum number of sweeps; defaults to max(8, 2N).

    Returns:
        NDArray[np.float64]: Activation times (ms) for all nodes.
    """
    N = int(graph.xyz.shape[0])
    src, dst = graph.edges_src, graph.edges_dst

    w = _edge_lengths_mm(graph.xyz, src, dst) / float(cv_mm_per_ms)

    t0 = jnp.full((N,), jnp.inf, dtype=w.dtype)
    t0 = t0.at[int(root_idx)].set(float(root_time_ms))

    if pmj_fixed_ms is not None:
        pmj_fixed_ms_jax = jnp.asarray(pmj_fixed_ms, dtype=w.dtype)
        t0 = t0.at[graph.pmj_idx].min(pmj_fixed_ms_jax)

    steps = int(max_steps if max_steps is not None else max(8, 2 * N))

    def body_fun(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        t, k, delta = state
        t_new = _relax_times(t, src, dst, w)
        d = jnp.max(jnp.abs(t_new - t))
        return (t_new, k + 1, d)

    def cond_fun(state: tuple[Array, Array, Array]) -> Array:
        _t, k, d = state
        return jnp.logical_and(k < steps, d > tol_ms)

    t_final, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, (t0, jnp.asarray(0), jnp.asarray(jnp.inf))
    )

    result = cast(NDArray[np.float64], np.asarray(t_final))
    _LOGGER.debug("SSSP finished: %d nodes, steps=%d", N, steps)
    return result


def pmj_times_ms(
    tree: Any,
    root_idx: int,
    root_time_ms: float,
    cv_mm_per_ms: float,
    *,
    pmj_fixed_ms: Optional[NDArray[np.float64]] = None,
    tol_ms: float = 1e-6,
    max_steps: Optional[int] = None,
) -> NDArray[np.float64]:
    """Convenience wrapper to compute activation times only at PMJs (ms).

    Args:
        tree: Tree object to be converted into a JAX graph.
        root_idx: Root node index.
        root_time_ms: Root time offset in ms.
        cv_mm_per_ms: Conduction velocity in mm/ms.
        pmj_fixed_ms: Optional fixed PMJ activation times (ms).
        tol_ms: Convergence tolerance (ms).
        max_steps: Optional maximum number of sweeps.

    Returns:
        NDArray[np.float64]: Activation times at PMJ nodes, in ms.
    """
    g = JaxPurkinjeGraph.from_purkinje_tree(tree)
    t_all: NDArray[np.float64] = sssp_times_ms(
        g,
        root_idx,
        root_time_ms,
        cv_mm_per_ms,
        pmj_fixed_ms=pmj_fixed_ms,
        tol_ms=tol_ms,
        max_steps=max_steps,
    )
    idx: NDArray[np.int64] = np.asarray(g.pmj_idx, dtype=int)
    return t_all[idx]
