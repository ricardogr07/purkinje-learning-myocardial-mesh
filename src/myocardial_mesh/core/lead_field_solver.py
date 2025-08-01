import logging
from typing import Optional, Any

import numpy as np
from pyvista import DataSet
from vtkmodules.numpy_interface import dataset_adapter as dsa

logger = logging.getLogger(__name__)


class LeadFieldSolver:
    """Computes ECG signals from activation times and lead fields."""

    def __init__(
        self,
        mesh: DataSet,
        electrode_pos: Optional[dict[str, Any]] = None,
        lead_fields_dict: Optional[dict[str, np.ndarray]] = None,
        stiffness_matrix: Optional[np.ndarray] = None,
        gi_nodal: Optional[np.ndarray] = None,
    ) -> None:
        self.mesh = mesh
        self.electrode_pos = electrode_pos
        self.lead_fields_dict = lead_fields_dict
        self.K = stiffness_matrix
        self.Gi_nodal = gi_nodal
        self.lead_field: dict[str, np.ndarray] = self._get_lead_field()

    def _get_lead_field(self) -> dict[str, np.ndarray]:
        """
        Retrieve precomputed lead field or calculate it from electrode positions.
        """
        if self.lead_fields_dict is not None:
            return self.lead_fields_dict
        if self.electrode_pos is None:
            raise RuntimeError("No electrodes or lead fields provided.")

        logger.info("Computing lead field from electrode positions.")
        points = dsa.WrapDataObject(self.mesh).Points
        return {
            name: 1 / np.linalg.norm(points - np.array(coord), axis=1)
            for name, coord in self.electrode_pos.items()
        }

    def compute_aux_integrals(self) -> dict[str, np.ndarray]:
        """
        Compute auxiliary integral terms: Gi.T * ∇Z_l
        """
        if self.electrode_pos is None or self.Gi_nodal is None:
            raise RuntimeError("Missing electrode positions or Gi_nodal.")

        logger.info("Computing auxiliary integrals for ECG.")
        points = dsa.WrapDataObject(self.mesh).Points
        Gi_T = np.transpose(self.Gi_nodal, axes=(0, 2, 1))

        return {
            name: np.sum(
                Gi_T
                * (
                    (points - np.array(coord))
                    / np.linalg.norm(points - np.array(coord), axis=1)[:, None] ** 3
                ),
                axis=1,
            )
            for name, coord in self.electrode_pos.items()
        }

    def compute_ecg_from_activation(self, record_array: bool = False) -> np.ndarray:
        """
        Compute ECG signal from activation times using lead field and stiffness matrix.
        """
        if self.lead_field is None or self.K is None:
            raise RuntimeError("Missing lead field or stiffness matrix.")

        dd = dsa.WrapDataObject(self.mesh)
        u = dd.PointData["activation"]

        V0, V1, eps = -80, 20, 1.0
        req_time_ini, req_time_fin = -5.0, 200.0
        n_times = int(req_time_fin - req_time_ini + 1)
        time_vec = np.linspace(req_time_ini, req_time_fin, n_times)

        # Use lead_field keys (safe regardless of whether electrode_pos was provided)
        V_l_dict: dict[str, np.ndarray] = {
            k: np.empty(n_times) for k in self.lead_field
        }
        uu = u - np.min(u)

        logger.info("Computing ECG signals from Vm projection.")
        for i, t in enumerate(time_vec):
            Vm = V0 + (V1 - V0) / 2 * (1 + np.tanh((t - uu) / eps))
            for name in self.lead_field:
                V_l_dict[name][i] = np.dot(self.lead_field[name], self.K @ Vm)

        # Compute Wilson central terminal (requires these specific leads to exist)
        try:
            V_W = (V_l_dict["RA"] + V_l_dict["LA"] + V_l_dict["LL"]) / 3
        except KeyError as e:
            raise RuntimeError(
                f"Missing required electrode for Wilson terminal: {e}"
            ) from e

        leads = [
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
        ecg_signals: list[np.ndarray] = []

        for lead in leads:
            if lead == "E1":
                ecg_signals.append(V_l_dict["LA"] - V_l_dict["RA"])
            elif lead == "E2":
                ecg_signals.append(V_l_dict["LL"] - V_l_dict["RA"])
            elif lead == "E3":
                ecg_signals.append(V_l_dict["LL"] - V_l_dict["LA"])
            elif lead == "aVR":
                ecg_signals.append(1.5 * (V_l_dict["RA"] - V_W))
            elif lead == "aVL":
                ecg_signals.append(1.5 * (V_l_dict["LA"] - V_W))
            elif lead == "aVF":
                ecg_signals.append(1.5 * (V_l_dict["LL"] - V_W))
            else:
                ecg_signals.append(V_l_dict[lead] - V_W)

        if record_array:
            return np.rec.fromarrays(ecg_signals, names=leads)
        return np.array(ecg_signals)
