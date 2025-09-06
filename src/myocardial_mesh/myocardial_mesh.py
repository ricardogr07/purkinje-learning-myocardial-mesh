import logging
import time
import pickle
import os

import numpy as np
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
import pyvista as pv
from scipy import sparse

from fimpy.solver import FIMPY

from .geometry_utils import Bmatrix, localStiffnessMatrix

logger = logging.getLogger(__name__)


class MyocardialMesh:
    """
    Endocardial myocardium mesh + ECG synthesis utilities.
    """

    SIDE_LV = 1
    SIDE_RV = 2

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
        - Supports legacy .vtk and XML .vtu readers (safe feature gating).
        - Accepts fibers in PointData['fiber'] or CellData['fiber'].
        - Builds tensors D, Laplacian K, lead fields, and FIM solver.
        """

        # ---- device
        self.device = "cpu" if device is None else device

        # ---- helpers to read meshes safely (XML vs legacy)
        def _read_unstructured(path: str):
            ext = os.path.splitext(path)[1].lower()
            if ext == ".vtu":
                rdr = vtk.vtkXMLUnstructuredGridReader()
            elif ext == ".vtk":
                rdr = vtk.vtkDataSetReader()
            else:
                # try XML then fall back
                rdr = vtk.vtkXMLUnstructuredGridReader()
                try:
                    rdr.SetFileName(path)
                    if hasattr(rdr, "ReadAllVectorsOn"):  # XML readers usually don't
                        rdr.ReadAllVectorsOn()
                    if hasattr(rdr, "ReadAllScalarsOn"):
                        rdr.ReadAllScalarsOn()
                    rdr.Update()
                    if rdr.GetOutput() is None:
                        raise RuntimeError
                except Exception:
                    rdr = vtk.vtkDataSetReader()

            rdr.SetFileName(path)
            if hasattr(rdr, "ReadAllVectorsOn"):
                rdr.ReadAllVectorsOn()
            if hasattr(rdr, "ReadAllScalarsOn"):
                rdr.ReadAllScalarsOn()
            rdr.Update()
            return rdr.GetOutput()

        # ---- read myocardium endocardium
        mesh_endo = _read_unstructured(myo_mesh)
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
        loc = vtk.vtkCellLocator()
        loc.SetDataSet(self.vtk_mesh)
        loc.BuildLocator()
        self.vtk_locator = loc

        # ---- electrodes
        with open(electrodes_position, "rb") as f:
            self.electrode_pos = pickle.load(f)  # dict: name -> (x,y,z)

        # ---- fibers (point or cell) → l_cell (T,3) and l_nodes (N,3)
        f0 = _read_unstructured(fibers)
        if f0 is None:
            raise RuntimeError(f"Failed to read fibers mesh: {fibers}")
        dd_f0 = dsa.WrapDataObject(f0)

        if "fiber" in dd_f0.PointData.keys():
            xyz_f0 = np.asarray(dd_f0.Points, dtype=float)
            fibers_pt = np.asarray(dd_f0.PointData["fiber"], dtype=float)

            from scipy.spatial.distance import cdist

            xyz_this = np.asarray(self.xyz, dtype=float)
            ci = np.argmin(cdist(xyz_this, xyz_f0), axis=1)
            if len(ci) != len(set(ci)):
                raise AssertionError(
                    "Fiber point mapping has duplicates; meshes may be mismatched."
                )
            if np.max(np.linalg.norm(xyz_this - xyz_f0[ci], axis=1)) > 1e-3:
                raise AssertionError(
                    "Fiber point mapping error: nearest distance > 1e-3."
                )

            l_nodes = fibers_pt[ci]  # (N,3)
            l_cell = np.mean(l_nodes[self.cells], axis=1)  # (T,3)
        elif "fiber" in dd_f0.CellData.keys():
            l_cell = np.asarray(dd_f0.CellData["fiber"], dtype=float)  # (T,3)
            tmp = pv.UnstructuredGrid(mesh_endo)
            tmp["l"] = l_cell
            l_nodes = np.asarray(tmp.cell_data_to_point_data()["l"], dtype=float)
        else:
            raise ValueError(
                "Fibers directions should be named 'fiber' in PointData or CellData."
            )

        # normalize
        l_cell = (
            l_cell / (np.linalg.norm(l_cell, axis=1, keepdims=True) + 1e-12)
        ).astype(np.float64)
        l_nodes = (
            l_nodes / (np.linalg.norm(l_nodes, axis=1, keepdims=True) + 1e-12)
        ).astype(np.float64)

        # ---- conductivity tensors
        if conductivity_params is None:
            sigma_il = 3.0
            sigma_el = 3.0
            sigma_it = 0.3
            sigma_et = 1.2
            alpha = 2.0
            beta = 800.0
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

        Gm = (
            sigma_mt * I_[None, :, :]
            + (sigma_ml - sigma_mt) * l_cell[:, :, None] @ l_cell[:, None, :]
        )
        self.D = (alpha**2 / beta) * Gm * 100.0  # mm^2/ms^2

        self.Gi_nodal = (
            sigma_it * I_[None, :, :]
            + (sigma_il - sigma_it) * l_nodes[:, :, None] @ l_nodes[:, None, :]
        )
        self.Gi_cell = (
            sigma_it * I_[None, :, :]
            + (sigma_il - sigma_it) * l_cell[:, :, None] @ l_cell[:, None, :]
        )

        cv_fiber_m_per_s = np.sqrt((alpha**2 / beta) * sigma_ml * 100.0)
        print(
            f"Conduction velocity in the direction of the fibers: {cv_fiber_m_per_s} m/s"
        )

        # ---- Laplacian
        print("assembling Laplacian")
        self.K = self.assemble_K(np.asarray(self.xyz, float), self.cells, self.Gi_cell)

        # ---- ECG auxiliaries
        self.aux_int_Vl = self.new_get_ecg_aux_Vl()
        self.lead_field = (
            lead_fields_dict if lead_fields_dict is not None else self.get_lead_field()
        )

        # ---- FIM solver
        print("initializing FIM solver")
        t0 = time.time()
        self.fim = FIMPY.create_fim_solver(
            self.xyz, self.cells, self.D, device=self.device
        )
        print(time.time() - t0)

    def assemble_K(self, pts, elm, Gi):
        B, J = Bmatrix(pts, elm)
        K = localStiffnessMatrix(B, J, Gi)
        N = pts.shape[0]
        Kvals = K.ravel('C')
        II = np.repeat(elm, 4, axis=1).ravel()
        JJ = np.tile(elm, 4).ravel()
        return sparse.coo_matrix((Kvals, (II, JJ)), shape=(N, N)).tocsr()

    def activate_fim(self, x0, x0_vals, return_only_pmjs=False):
        import pyvista as pv

        dd = dsa.WrapDataObject(self.vtk_mesh)
        cells = self.cells
        xyz = dd.Points

        act = np.empty(xyz.shape[0])
        act.fill(np.inf)
        cellId = vtk.reference(0)
        subId = vtk.reference(0)
        x_proj = [0.0, 0.0, 0.0]
        dist = vtk.reference(0.0)

        print('computing closest nodes to PMJs')
        t0 = time.time()
        for k in range(x0.shape[0]):
            x_orig = x0[k, :]
            self.vtk_locator.FindClosestPoint(x_orig, x_proj, cellId, subId, dist)
            Gcell = np.linalg.inv(self.D[cellId, ...])
            cell_pts = cells[cellId, :]
            v = xyz[cell_pts, :] - x0[[k], :]
            new_act = x0_vals[k] + np.sqrt(np.einsum('ij,kj,ki->k', Gcell, v, v))
            act[cell_pts] = np.minimum(new_act, act[cell_pts])
        print(time.time() - t0)

        x0_mask = np.isfinite(act)  # boolean seeds
        seed_vals = act[x0_mask]

        print('solving')
        t0 = time.time()
        sol = self.fim.comp_fim(x0_mask, seed_vals)
        print(time.time() - t0)

        if self.device == 'gpu':
            sol = sol.get()
        dd.PointData['activation'][:] = sol

        if return_only_pmjs:
            x0_pv = pv.PolyData(x0.copy())
            result = x0_pv.sample(
                pv.UnstructuredGrid(self.vtk_mesh),
                tolerance=1e-6,
                snap_to_closest_point=True,
            )
            assert (
                np.sum(result['vtkValidPointMask'] == 0) == 0
            ), 'Error while sampling to x0_purkinje'
            return result['activation']
        else:
            return sol

    def new_get_ecg_aux_Vl(self):
        dd = dsa.WrapDataObject(self.vtk_mesh)
        xyz = dd.Points
        aux_int_l = {}
        for electrode_name, electrode_coords in self.electrode_pos.items():
            r = xyz - np.array(electrode_coords)
            r_norm3 = r / np.linalg.norm(r, axis=1) ** 3
            Gi_nodal_T = np.transpose(self.Gi_nodal, axes=(0, 2, 1))
            aux_int_l[electrode_name] = np.sum(Gi_nodal_T * r_norm3, axis=1)
        return aux_int_l

    def get_lead_field(self):
        aux_int_l = {}
        for electrode_name, electrode_coords in self.electrode_pos.items():
            r = self.xyz - np.array(electrode_coords)
            aux_int_l[electrode_name] = 1 / np.linalg.norm(r, axis=1)
        return aux_int_l

    def new_get_ecg(self, record_array=True):
        dd = dsa.WrapDataObject(self.vtk_mesh)
        u = dd.PointData['activation']
        V0, V1 = -80, 20
        eps = 1.0
        umin, _ = np.min(u), np.max(u)
        uu = u - umin
        req_time_ini = -5.0
        req_time_fin = 200.0
        n_times = int(req_time_fin - req_time_ini + 1)
        V_l_dict = {key: np.empty(n_times) for key in self.electrode_pos.keys()}
        for n_t, req_time in enumerate(
            np.linspace(req_time_ini, req_time_fin, n_times)
        ):
            Vm = V0 + (V1 - V0) / 2 * (1 + np.tanh((req_time - uu) / eps))
            for electrode_name in self.electrode_pos.keys():
                V_l_dict[electrode_name][n_t] = np.dot(
                    self.lead_field[electrode_name], self.K.dot(Vm)
                )
        leads_names = [
            "E1",
            "E2",
            "E3",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        V_W = 1.0 / 3.0 * (V_l_dict["RA"] + V_l_dict["LA"] + V_l_dict["LL"])
        arrays = []
        for l_name in leads_names:
            if l_name == "E1":
                arrays.append(V_l_dict["LA"] - V_l_dict["RA"])
            elif l_name == "E2":
                arrays.append(V_l_dict["LL"] - V_l_dict["RA"])
            elif l_name == "E3":
                arrays.append(V_l_dict["LL"] - V_l_dict["LA"])
            elif l_name == "aVR":
                arrays.append(3.0 / 2.0 * (V_l_dict["RA"] - V_W))
            elif l_name == "aVL":
                arrays.append(3.0 / 2.0 * (V_l_dict["LA"] - V_W))
            elif l_name == "aVF":
                arrays.append(3.0 / 2.0 * (V_l_dict["LL"] - V_W))
            else:
                arrays.append(V_l_dict[l_name] - V_W)
        return (
            np.rec.fromarrays(arrays, names=leads_names)
            if record_array
            else np.asarray(arrays)
        )

    def save_pv(self, fname):
        pv.UnstructuredGrid(self.vtk_mesh).save(fname)
