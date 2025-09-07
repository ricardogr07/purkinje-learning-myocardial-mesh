import logging
import time
import pickle

import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa
import pyvista as pv

from fimpy.solver import FIMPY

from .core.fem_utils import assemble_K
from .io.fiber_processor import process_fibers
from .io.mesh_io import load_unstructured, build_cell_locator
from .core.activation import (
    legacy_seed_projection,
    fim_solve_from_seeds,
    sample_activation_at_points,
)
from .lead_field.lead_field_solver import LeadFieldSolver

_LOGGER = logging.getLogger(__name__)


class MyocardialMesh:
    """
    Endocardial myocardium mesh + ECG synthesis utilities.
    """

    def __init__(
        self,
        myo_mesh: str,
        electrodes_position: str,
        fibers: str,
        device: str | None = "cpu",
        conductivity_params: dict | None = None,
        lead_fields_dict: dict | None = None,
    ):
        """
        Myocardial mesh wrapper.

        - Preserves PointData['activation'] if present (e.g., True_endo.vtu).
        - Uses mesh_io.load_unstructured + mesh_io.build_cell_locator.
        - Loads fibers via io.fiber_processor.process_fibers (legacy-equivalent math).
        - Builds tensors D, Laplacian K, lead fields, and FIM solver.
        """
        # ---- device
        self.device = "cpu" if device is None else device

        # ---- read myocardium endocardium (XML/legacy handled in mesh_io)
        mesh_endo = load_unstructured(myo_mesh)
        if mesh_endo is None:
            raise RuntimeError(f"Failed to read mesh: {myo_mesh}")
        self.vtk_mesh = mesh_endo

        # ---- VTK adapter
        dd = dsa.WrapDataObject(self.vtk_mesh)
        self.xyz = dd.Points
        self.cells = dd.Cells.reshape((-1, 5))[:, 1:]  # tetra indices (T,4)

        # ---- activation: preserve if present
        npts = self.xyz.shape[0]
        if "activation" in dd.PointData.keys():
            act = np.asarray(dd.PointData["activation"], dtype=float)
            if act.shape[0] != npts:
                raise ValueError(
                    f"'activation' length {act.shape[0]} != number of points {npts} in {myo_mesh}"
                )
            dd.PointData["activation"][:] = act
        else:
            dd.PointData.append(np.full(npts, np.inf, dtype=float), "activation")

        # ---- locator (for PMJ snapping/probing)
        self.vtk_locator = build_cell_locator(self.vtk_mesh)

        # ---- electrodes
        with open(electrodes_position, "rb") as f:
            self.electrode_pos = pickle.load(f)  # dict: name -> (x,y,z)

        # ---- fibers → l_nodes/l_cell, Gi/Gm/D (delegated to processor)
        fr = process_fibers(
            fibers_path=fibers,
            xyz=np.asarray(self.xyz, float),
            cells=self.cells,
            conductivity_params=conductivity_params,
        )
        self.l_nodes = fr.l_nodes
        self.l_cell = fr.l_cell
        self.Gi_nodal = fr.Gi_nodal
        self.Gi_cell = fr.Gi_cell
        self.D = fr.D
        print(
            f"Conduction velocity in the direction of the fibers: {fr.cv_fiber_m_per_s} m/s"
        )

        # ---- Laplacian
        print("assembling Laplacian")
        self.K = assemble_K(np.asarray(self.xyz, float), self.cells, self.Gi_cell)

        # ---- Lead-field helper (composition; behavior unchanged)
        self.lead = LeadFieldSolver(
            xyz=np.asarray(self.xyz, float),
            electrode_pos=self.electrode_pos,
            Gi_nodal=self.Gi_nodal,
            K=self.K,
        )
        # keep cached fields for parity with legacy API
        self.aux_int_Vl = self.lead.compute_aux_Vl()
        self.lead_field = (
            lead_fields_dict
            if lead_fields_dict is not None
            else self.lead.get_lead_field()
        )

        # ---- FIM solver
        print("initializing FIM solver")
        t0 = time.time()
        self.fim = FIMPY.create_fim_solver(
            self.xyz, self.cells, self.D, device=self.device
        )
        print(time.time() - t0)

    def activate_fim(self, x0, x0_vals, return_only_pmjs=False):
        """
        Exact legacy behavior, now via small, testable helpers:
          1) legacy per-PMJ projection onto closest cell's 4 nodes
          2) full FIM solve
          3) optional sampling back at PMJ coordinates
        """
        dd = dsa.WrapDataObject(self.vtk_mesh)
        xyz = np.asarray(dd.Points, float)

        print('computing closest nodes to PMJs')
        t0 = time.time()
        act_init = legacy_seed_projection(
            x0=np.asarray(x0, float),
            x0_vals=np.asarray(x0_vals, float),
            vtk_locator=self.vtk_locator,
            cells=self.cells,
            D_cell=self.D,
            xyz=xyz,
        )
        print(time.time() - t0)

        print('solving')
        t0 = time.time()
        sol = fim_solve_from_seeds(act_init=act_init, fim_solver=self.fim)
        print(time.time() - t0)

        # update vtk field
        dd.PointData['activation'][:] = sol

        if return_only_pmjs:
            return sample_activation_at_points(
                vtk_mesh=self.vtk_mesh, points_xyz=np.asarray(x0, float)
            )
        else:
            return sol

    def new_get_ecg_aux_Vl(self):
        return self.lead.compute_aux_Vl()

    def get_lead_field(self):
        return self.lead.get_lead_field()

    def get_ecg(self, *args, **kwargs):
        return self.new_get_ecg(*args, **kwargs)

    def new_get_ecg(self, record_array=True):
        dd = dsa.WrapDataObject(self.vtk_mesh)
        u = np.asarray(dd.PointData["activation"], dtype=float)
        return self.lead.ecg_from_activation(
            u, record_array=record_array, lead_field=self.lead_field
        )

    def save_pv(self, fname):
        pv.UnstructuredGrid(self.vtk_mesh).save(fname)
