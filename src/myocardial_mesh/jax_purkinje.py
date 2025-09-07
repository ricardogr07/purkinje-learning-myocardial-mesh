# src/myocardial_mesh/jax_purkinje.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    raise RuntimeError(
        "JAX is required for the JAX Purkinje propagator. "
        "Install jax (and jaxlib) first."
    ) from e


@dataclass
class JaxPurkinjeGraph:
    """Lightweight JAX graph view of a Purkinje tree."""

    xyz: jnp.ndarray  # (N, 3)
    edges_src: jnp.ndarray  # (E,)
    edges_dst: jnp.ndarray  # (E,)
    pmj_idx: jnp.ndarray  # (P,)

    @classmethod
    def from_purkinje_tree(cls, tree) -> "JaxPurkinjeGraph":
        # Nodes
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

        return cls(xyz=xyz, edges_src=src, edges_dst=dst, pmj_idx=pmj)


def _edge_lengths_mm(
    xyz: jnp.ndarray, src: jnp.ndarray, dst: jnp.ndarray
) -> jnp.ndarray:
    return jnp.linalg.norm(xyz[dst] - xyz[src], axis=1)


@jax.jit
def _relax_times(
    times: jnp.ndarray, src: jnp.ndarray, dst: jnp.ndarray, w: jnp.ndarray
) -> jnp.ndarray:
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
    pmj_fixed_ms: np.ndarray | None = None,
    tol_ms: float = 1e-6,
    max_steps: int | None = None,
) -> np.ndarray:
    """
    Single-source shortest-path times on the undirected tree, in milliseconds.
    Optional boundary conditions on PMJ nodes via `pmj_fixed_ms` (length P).
    """
    N = int(graph.xyz.shape[0])
    src, dst = graph.edges_src, graph.edges_dst

    w = _edge_lengths_mm(graph.xyz, src, dst) / float(cv_mm_per_ms)

    t0 = jnp.full((N,), jnp.inf, dtype=w.dtype)
    t0 = t0.at[int(root_idx)].set(float(root_time_ms))

    # If PMJ fixed times are provided (ms), apply them as boundary conditions.
    if pmj_fixed_ms is not None:
        pmj_fixed_ms = jnp.asarray(pmj_fixed_ms, dtype=w.dtype)
        t0 = t0.at[graph.pmj_idx].min(pmj_fixed_ms)

    steps = int(max_steps if max_steps is not None else max(8, 2 * N))

    def body_fun(state):
        t, k, delta = state
        t_new = _relax_times(t, src, dst, w)
        d = jnp.max(jnp.abs(t_new - t))
        return (t_new, k + 1, d)

    def cond_fun(state):
        _t, k, d = state
        return jnp.logical_and(k < steps, d > tol_ms)

    t_final, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, (t0, jnp.asarray(0), jnp.asarray(jnp.inf))
    )
    return np.asarray(t_final)


def pmj_times_ms(
    tree,
    root_idx: int,
    root_time_ms: float,
    cv_mm_per_ms: float,
    *,
    pmj_fixed_ms: np.ndarray | None = None,
    tol_ms: float = 1e-6,
    max_steps: int | None = None,
) -> np.ndarray:
    """
    Convenience wrapper: return times only at PMJs, in ms.
    """
    g = JaxPurkinjeGraph.from_purkinje_tree(tree)
    t_all = sssp_times_ms(
        g,
        root_idx,
        root_time_ms,
        cv_mm_per_ms,
        pmj_fixed_ms=pmj_fixed_ms,
        tol_ms=tol_ms,
        max_steps=max_steps,
    )
    return t_all[np.asarray(g.pmj_idx)]
