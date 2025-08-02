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

        self.vtk_locator = self._build_locator(self.vtk_mesh)

        self.dd = dsa.WrapDataObject(self.vtk_mesh)
        self.xyz = self.dd.Points
        self.cells = self.dd.Cells.reshape((-1, 5))[:, 1:]

        self.device = device
        self.fibers_path = fibers_path
        self.f0 = self._load_fibers(self.fibers_path)
        self.K = None  # Global stiffness matrix
        self.D = None  # Conductivity tensors
        self.Gi_nodal = None

        self.lead_field = None
        self.lead_fields_dict = lead_fields_dict
        self.electrode_pos = self._load_electrode_positions(electrodes_position)

        if conductivity_params is not None:
            self._initialize_conductivity(conductivity_params)

        self.fim = None
        self._reset_activation()

        self.solver = LeadFieldSolver(
            mesh=self.vtk_mesh,
            electrode_pos=self.electrode_pos,
            lead_fields_dict=self.lead_fields_dict,
        )

    def _build_locator(self, mesh):
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        return locator

    def _load_fibers(self, path: Optional[str]) -> Optional[vtk.vtkDataSet]:
        if path is None:
            return None

        reader = vtk.vtkDataSetReader()
        reader.SetFileName(path)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        return reader.GetOutput()

    def _load_electrode_positions(self, path: Optional[str]):
        if path is None:
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _reset_activation(self) -> None:
        activation = np.full(self.xyz.shape[0], np.inf)
        self.dd.PointData.append(activation, "activation")

    def _initialize_conductivity(self, conductivity_params: np.ndarray) -> None:
        logger.info("Initializing conductivity tensors (D)...")
        D_local = np.zeros((self.cells.shape[0], 3, 3))
        for i in range(self.cells.shape[0]):
            D_local[i] = np.diag(conductivity_params)
        self.D = D_local

    def assemble_stiffness_matrix(self) -> None:
        logger.info("Assembling global stiffness matrix...")
        B, J = compute_Bmatrix(self.xyz, self.cells)
        self.K = compute_local_stiffness_matrix(B, J, self.D)

    def compute_ecg(self, record_array: bool = True) -> np.ndarray:
        return self.solver.compute_ecg_from_activation(record_array=record_array)

    def compute_ecg_aux_field(self) -> dict[str, np.ndarray]:
        return self.solver.compute_aux_integrals()

    def sample_activation_at(self, points: np.ndarray) -> np.ndarray:
        return VTKGeometryUtils.probe_activation(points, self.vtk_mesh)

    def project_pmjs(self, pmjs: np.ndarray) -> np.ndarray:
        return VTKGeometryUtils.find_closest_pmjs(pmjs, self.vtk_locator)

    def _infer_write_method_from_extension(self, ext: str) -> WriteMethod:
        normalized = ext.lstrip(".").lower()
        if normalized in {"vtk", "vtp"}:
            return WriteMethod.VTK
        if normalized == "pyvista":
            return WriteMethod.PYVISTA
        if normalized == "meshio":
            return WriteMethod.MESHIO
        logger.warning(
            "Unrecognized extension '%s', defaulting to WriteMethod.VTK", normalized
        )
        return WriteMethod.VTK

    def save(self, path: str) -> None:
        logger.info(f"Saving mesh to {path}")
        path_obj = Path(path)
        method_enum = self._infer_write_method_from_extension(path_obj.suffix)
        MeshIO.write(self.vtk_mesh, path_obj, method_enum)

    def plot(self) -> None:
        DataPlotter.plot_mesh(self.vtk_mesh)

    def plot_ecg(self, arrays, leads_names, t0, t1, n_times) -> None:
        DataPlotter.plot_ecg(arrays, leads_names, t0, t1, n_times)
