from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import numpy as np
from scipy import sparse


@dataclass
class LeadFieldSolver:
    """
    Minimal ECG synthesis helper.

    Responsibilities (identical math to your legacy implementation):
    - Build simple 1/|r| lead fields per electrode (node-wise weights).
    - Given an activation field, synthesize 12-lead ECG using the same
      Vm(t) and Laplacian/K pipeline you had in MyocardialMesh.new_get_ecg.
    """

    xyz: np.ndarray  # (N, 3)
    electrode_pos: Dict[str, Tuple[float, float, float]]
    Gi_nodal: np.ndarray  # (N, 3, 3)  (not used in synthesis; kept for parity/optionals)
    K: np.ndarray | sparse.spmatrix

    def get_lead_field(self) -> Dict[str, np.ndarray]:
        """
        Classic 1/|r| node-wise weights per electrode (unchanged).
        """
        lf: Dict[str, np.ndarray] = {}
        for name, coords in self.electrode_pos.items():
            r = self.xyz - np.asarray(coords, dtype=float)
            lf[name] = 1.0 / np.linalg.norm(r, axis=1)
        return lf

    def compute_aux_Vl(self) -> Dict[str, np.ndarray]:
        """
        Gi^T * r/|r|^3 node-wise quantity (kept so callers can cache it).
        Not used in the current ECG synthesis path, but matches the legacy API.
        """
        aux: Dict[str, np.ndarray] = {}
        Gi_T = np.transpose(self.Gi_nodal, axes=(0, 2, 1))  # (N,3,3)
        for name, coords in self.electrode_pos.items():
            r = self.xyz - np.asarray(coords, dtype=float)  # (N,3)
            r_norm3 = r / (np.linalg.norm(r, axis=1, keepdims=True) ** 3)  # (N,3)
            aux[name] = np.sum(Gi_T * r_norm3[:, None, :], axis=(1, 2))  # (N,)
        return aux

    # ---- ECG synthesis ----

    @staticmethod
    def _twelve_leads_from_limb_and_precordials(
        V_l_dict: Dict[str, np.ndarray]
    ) -> np.recarray:
        """
        Assemble 12-lead outputs (E1/E2/E3, aVR/aVL/aVF, V1..V6) from node-integrated
        limb potentials RA/LA/LL and precordials V1..V6. Matches legacy exact formulas.
        """
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

        V_W = (V_l_dict["RA"] + V_l_dict["LA"] + V_l_dict["LL"]) / 3.0

        arrays = []
        arrays.append(V_l_dict["LA"] - V_l_dict["RA"])  # E1
        arrays.append(V_l_dict["LL"] - V_l_dict["RA"])  # E2
        arrays.append(V_l_dict["LL"] - V_l_dict["LA"])  # E3
        arrays.append(1.5 * (V_l_dict["RA"] - V_W))  # aVR
        arrays.append(1.5 * (V_l_dict["LA"] - V_W))  # aVL
        arrays.append(1.5 * (V_l_dict["LL"] - V_W))  # aVF
        for name in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            arrays.append(V_l_dict[name] - V_W)

        return np.rec.fromarrays(arrays, names=leads)

    def ecg_from_activation(
        self,
        activation: np.ndarray,  # (N,)
        *,
        record_array: bool = True,
        req_time_ini: float = -5.0,
        req_time_fin: float = 200.0,
        V0: float = -80.0,
        V1: float = 20.0,
        eps: float = 1.0,
        electrodes_order: Iterable[str] | None = None,
        lead_field: Dict[str, np.ndarray] | None = None,
    ):
        """
        Reproduce MyocardialMesh.new_get_ecg() bit-for-bit.

        - Vm(t) = V0 + (V1-V0)/2 * (1 + tanh((t - (u - umin))/eps))
        - Integrate K·Vm against 1/|r| weights for each electrode.
        - Derive the 12-lead set.

        Returns a structured array (default) or a float ndarray (12, T)
        if record_array=False (matching your tests).
        """
        u = np.asarray(activation, dtype=float)
        umin = float(np.min(u))
        uu = u - umin

        n_times = int(req_time_fin - req_time_ini + 1)
        times = np.linspace(req_time_ini, req_time_fin, n_times)

        # Use caller-supplied lead_field if present (parity), else compute.
        lf = lead_field if lead_field is not None else self.get_lead_field()

        # default electrode iteration order = whatever is in the dict
        if electrodes_order is None:
            electrodes_order = list(lf.keys())

        V_l_dict: Dict[str, np.ndarray] = {
            name: np.empty(n_times, dtype=float) for name in lf.keys()
        }

        # main loop identical to legacy new_get_ecg
        for i, t in enumerate(times):
            Vm = V0 + (V1 - V0) * 0.5 * (1.0 + np.tanh((t - uu) / eps))
            KVm = self.K.dot(Vm)  # (N,)
            for name in electrodes_order:
                V_l_dict[name][i] = float(np.dot(lf[name], KVm))

        rec = self._twelve_leads_from_limb_and_precordials(V_l_dict)
        if record_array:
            return rec
        else:
            # return as (12, T) to match your “vector” usage in the orchestrator’s early-stop
            return np.vstack([rec[name] for name in rec.dtype.names])
