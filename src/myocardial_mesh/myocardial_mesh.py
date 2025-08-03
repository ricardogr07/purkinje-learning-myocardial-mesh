import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa

from myocardial_mesh.io.mesh_io import MeshIO, WriteMethod
from myocardial_mesh.viz.data_plotter import DataPlotter
from myocardial_mesh.core.lead_field_solver import LeadFieldSolver
from myocardial_mesh.core.vtk_geometry_utils import VTKGeometryUtils
from myocardial_mesh.core.fem_utils import (
    compute_Bmatrix,
    compute_local_stiffness_matrix,
)

logger = logging.getLogger(__name__)


class MyocardialMesh:
    """Class to operate on myocardial VTK mesh and compute activation and ECG."""

    def __init__(
        self,
        mesh_path: str,
        fibers_path: Optional[str] = None,
        electrodes_position: Optional[str] = None,
        lead_fields_dict: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        conductivity_params: Optional[np.ndarray] = None,
    ):
        logger.info("Initializing MyocardialMesh...")
        mesh_path_obj = Path(mesh_path)
        self.vtk_mesh = MeshIO.read_mesh(mesh_path_obj)

        # Build spatial locator for coupling Purkinje tree to myocardium
        self.vtk_locator = self._build_locator(self.vtk_mesh)

        # Wrap to numpy adapter
        self.dd = dsa.WrapDataObject(self.vtk_mesh)
        self.xyz = self.dd.Points  # (N_points, 3)
        self.cells = self.dd.Cells.reshape((-1, 5))[
            :, 1:
        ]  # (N_cells, 4) or (N_cells, …)

        # Metadata
        self.device = device
        self.fibers_path = fibers_path
        self.f0 = self._load_fibers(self.fibers_path)
        self.lead_fields_dict = lead_fields_dict
        self.electrode_pos = self._load_electrode_positions(electrodes_position)

        # Ensure conductivity tensor array always set
        if conductivity_params is None:
            conductivity_params = np.ones(3, dtype=float)
        self._initialize_conductivity(conductivity_params)

        # Initialize activation field to +∞
        self._reset_activation()

        # Solver (lead fields computed on init)
        self.solver = LeadFieldSolver(
            mesh=self.vtk_mesh,
            electrode_pos=self.electrode_pos,
            lead_fields_dict=self.lead_fields_dict,
            stiffness_matrix=None,
            gi_nodal=None,
        )

        # Global stiffness matrix (populated on first ECG compute)
        self.K: Optional[np.ndarray] = None

    def _build_locator(self, mesh: vtk.vtkDataSet) -> vtk.vtkCellLocator:
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        return locator

    def _load_fibers(self, path: Optional[str]) -> Optional[vtk.vtkDataSet]:
        if not path:
            return None
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(path)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        return reader.GetOutput()

    def _load_electrode_positions(
        self, path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _reset_activation(self) -> None:
        activation = np.full(self.xyz.shape[0], np.inf)
        self.dd.PointData.append(activation, "activation")

    def _initialize_conductivity(self, conductivity_params: np.ndarray) -> None:
        """
        Build conductivity tensor D as an (n_cells x 3 x 3) array.
        """
        logger.info("Initializing conductivity tensors (D)...")
        n_cells = self.cells.shape[0]
        D_local = np.zeros((n_cells, 3, 3), dtype=float)
        for i in range(n_cells):
            D_local[i] = np.diag(conductivity_params)
        self.D = D_local

    def _infer_write_method_from_extension(self, ext: str) -> WriteMethod:
        normalized = ext.lstrip(".").lower()
        if normalized in {"vtk", "vtp"}:
            return WriteMethod.VTK
        # For .vtu (unstructured grid), use PyVista which handles both mesh types
        if normalized == "vtu":
            return WriteMethod.PYVISTA
        if normalized == "pyvista":
            return WriteMethod.PYVISTA
        if normalized == "meshio":
            return WriteMethod.MESHIO
        logger.warning(
            "Unrecognized extension '%s', defaulting to VTK writer", normalized
        )
        return WriteMethod.VTK

    def assemble_stiffness_matrix(self) -> None:
        logger.info("Assembling global stiffness matrix…")
        B, J = compute_Bmatrix(self.xyz, self.cells)
        local_K = compute_local_stiffness_matrix(B, J, self.D)

        n_pts = self.xyz.shape[0]
        K_global = np.zeros((n_pts, n_pts), dtype=float)

        for e, ids in enumerate(self.cells):
            Ke = local_K[e]
            for local_i, gi in enumerate(ids):
                K_global[gi, ids] += Ke[local_i, :]

        self.K = K_global

    def compute_ecg(self, record_array: bool = True) -> np.ndarray:
        """
        Compute ECG from activation; auto-assembles stiffness matrix and injects it.
        """
        if self.K is None:
            self.assemble_stiffness_matrix()
        self.solver.K = self.K
        return self.solver.compute_ecg_from_activation(record_array=record_array)

    def compute_ecg_aux_field(self) -> Dict[str, np.ndarray]:
        return self.solver.compute_aux_integrals()

    def sample_activation_at(self, points: np.ndarray) -> np.ndarray:
        return VTKGeometryUtils.probe_activation(points, self.vtk_mesh)

    def activate_fim(
        self,
        points: np.ndarray,
        x0_vals: Optional[np.ndarray] = None,
        return_only_pmjs: bool = False,
    ) -> np.ndarray:
        """
        Legacy FIM interface for BO:
            - return_only_pmjs=True returns activation at 'points'
            - else returns full activation field array
        """
        if return_only_pmjs:
            return VTKGeometryUtils.probe_activation(self.vtk_mesh, points)
        return np.array(self.dd.PointData["activation"])

    def project_pmjs(self, pmjs: np.ndarray) -> np.ndarray:
        return VTKGeometryUtils.find_closest_pmjs(pmjs, self.vtk_locator)

    def save(self, path: str) -> None:
        logger.info(f"Saving mesh to {path}")
        path_obj = Path(path)
        method_enum = self._infer_write_method_from_extension(path_obj.suffix)
        success = MeshIO.write(self.vtk_mesh, path_obj, method_enum)
        if not success:
            logger.error(f"Failed to save mesh to {path} using method {method_enum}")

    def plot(self) -> None:
        DataPlotter.plot_mesh(self.vtk_mesh)

    def plot_ecg(
        self,
        arrays: np.ndarray,
        leads_names: list[str],
        t0: float,
        t1: float,
        n_times: int,
    ) -> None:
        DataPlotter.plot_ecg(arrays, leads_names, t0, t1, n_times)
