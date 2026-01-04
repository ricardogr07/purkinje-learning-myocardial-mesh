"""Fiber processing utilities for myocardial meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os

import numpy as np
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkDataSet
import pyvista as pv
from scipy.spatial.distance import cdist

__all__ = ["FiberResult", "process_fibers"]


@dataclass
class FiberResult:
    """Container for processed fiber directions and conductivity tensors."""

    l_nodes: np.ndarray  # (N,3) float64, unit vectors
    l_cell: np.ndarray  # (T,3) float64, unit vectors
    Gi_nodal: np.ndarray  # (N,3,3) float64
    Gi_cell: np.ndarray  # (T,3,3) float64
    D: np.ndarray  # (T,3,3) float64 (mm^2/ms^2)
    cv_fiber_m_per_s: float  # scalar


def _read_unstructured(path: str) -> vtkDataSet:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtu":
        rdr = vtk.vtkXMLUnstructuredGridReader()
    elif ext == ".vtk":
        rdr = vtk.vtkDataSetReader()
    else:
        # try XML then fallback
        rdr = vtk.vtkXMLUnstructuredGridReader()
        try:
            rdr.SetFileName(path)
            rdr.Update()
            if rdr.GetOutput() is None:
                raise RuntimeError
        except Exception:
            rdr = vtk.vtkDataSetReader()
    rdr.SetFileName(path)
    # legacy readers only:
    if hasattr(rdr, "ReadAllVectorsOn"):
        rdr.ReadAllVectorsOn()
    if hasattr(rdr, "ReadAllScalarsOn"):
        rdr.ReadAllScalarsOn()
    rdr.Update()
    out = rdr.GetOutput()
    if out is None:
        raise RuntimeError(f"Failed to read fibers mesh: {path}")
    return out


def _normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return (v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)).astype(
        np.float64
    )


def process_fibers(
    fibers_path: str,
    *,
    xyz: np.ndarray,  # (N,3) myocardium points
    cells: np.ndarray,  # (T,4) tetra connectivity
    conductivity_params: Optional[Dict[str, float]] = None,
) -> FiberResult:
    """Load and process fiber directions for a myocardium mesh.

    This loads the fiber field from PointData["fiber"] or CellData["fiber"],
    computes normalized directions at nodes and cells, and derives Gi/Gm/D
    tensors using the legacy parameterization.
    """
    f0 = _read_unstructured(fibers_path)
    dd_f0 = dsa.WrapDataObject(f0)

    if 'fiber' in dd_f0.PointData.keys():
        xyz_f0 = np.asarray(dd_f0.Points, dtype=float)
        fibers_pt = np.asarray(dd_f0.PointData['fiber'], dtype=float)
        ci = np.argmin(cdist(np.asarray(xyz, float), xyz_f0), axis=1)
        if len(ci) != len(set(ci)):
            raise AssertionError(
                "Fiber point mapping has duplicates; meshes may be mismatched."
            )
        if (
            float(np.max(np.linalg.norm(np.asarray(xyz, float) - xyz_f0[ci], axis=1)))
            > 1e-3
        ):
            raise AssertionError("Fiber point mapping error: nearest distance > 1e-3.")
        l_nodes = fibers_pt[ci]  # (N,3)
        l_cell = np.mean(l_nodes[cells], axis=1)  # (T,3)
    elif 'fiber' in dd_f0.CellData.keys():
        l_cell = np.asarray(dd_f0.CellData['fiber'], dtype=float)
        tmp = pv.UnstructuredGrid(f0)
        tmp["l"] = l_cell
        l_nodes = np.asarray(tmp.cell_data_to_point_data()["l"], dtype=float)
    else:
        raise ValueError(
            "Fibers directions should be named 'fiber' in PointData or CellData."
        )

    l_nodes = _normalize(l_nodes, axis=1)
    l_cell = _normalize(l_cell, axis=1)

    if conductivity_params is None:
        sigma_il, sigma_el = 3.0, 3.0
        sigma_it, sigma_et = 0.3, 1.2
        alpha, beta = 2.0, 800.0
    else:
        sigma_il = conductivity_params["sigma_il"]
        sigma_el = conductivity_params["sigma_el"]
        sigma_it = conductivity_params["sigma_it"]
        sigma_et = conductivity_params["sigma_et"]
        alpha = conductivity_params["alpha"]
        beta = conductivity_params["beta"]

    I_ = np.eye(3, dtype=float)
    sigma_mt = (sigma_et * sigma_it) / (sigma_et + sigma_it)
    sigma_ml = (sigma_el * sigma_il) / (sigma_el + sigma_il)

    Gi_nodal = (
        sigma_it * I_[None, :, :]
        + (sigma_il - sigma_it) * l_nodes[:, :, None] @ l_nodes[:, None, :]
    )
    Gi_cell = (
        sigma_it * I_[None, :, :]
        + (sigma_il - sigma_it) * l_cell[:, :, None] @ l_cell[:, None, :]
    )

    # Monodomain effective Gm and D (legacy scaling to mm^2/ms^2)
    Gm = (
        sigma_mt * I_[None, :, :]
        + (sigma_ml - sigma_mt) * l_cell[:, :, None] @ l_cell[:, None, :]
    )
    D = (alpha**2 / beta) * Gm * 100.0

    cv_fiber = float(np.sqrt((alpha**2 / beta) * sigma_ml * 100.0))  # m/s
    return FiberResult(
        l_nodes=l_nodes,
        l_cell=l_cell,
        Gi_nodal=Gi_nodal,
        Gi_cell=Gi_cell,
        D=D,
        cv_fiber_m_per_s=cv_fiber,
    )
