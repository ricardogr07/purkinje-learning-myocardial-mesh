import logging
from typing import Any, Union, Sequence

import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class DataPlotter:
    """Provides methods for visualizing mesh and ECG data."""

    @staticmethod
    def plot_mesh(
        meshes: Union[Any, Sequence[Any]],
        show: bool = True,
    ) -> None:
        """
        Plot one or more 3D meshes using PyVista.

        Parameters
        ----------
        meshes : Any or list[Any]
            A single mesh or a sequence of meshes (e.g., PolyData or UnstructuredGrid).
        show : bool, optional
            Whether to immediately display the plot window.
        """
        logger.info("Plotting mesh(es)...")

        if not isinstance(meshes, (list, tuple)):
            meshes = [meshes]

        plotter = pv.Plotter()
        for idx, mesh in enumerate(meshes):
            plotter.add_mesh(mesh, show_edges=True, opacity=0.6)
            logger.debug(
                f"Added mesh {idx} with {mesh.n_points} points and {mesh.n_cells} cells."
            )

        plotter.add_axes()
        if show:
            plotter.show()

    @staticmethod
    def plot_ecg(
        arrays: np.ndarray,
        leads_names: list[str],
        req_time_ini: float,
        req_time_fin: float,
        n_times: int,
    ) -> None:
        """
        Plot ECG signals from multiple leads in a 3x4 grid layout.

        Parameters
        ----------
        arrays : np.ndarray
            2D array of shape (n_leads, n_samples) with ECG signal values.
        leads_names : list[str]
            Names of each lead for labeling the subplots.
        req_time_ini : float
            Initial time (in seconds or ms).
        req_time_fin : float
            Final time.
        n_times : int
            Number of time samples.
        """
        logger.info("Plotting ECG signals...")

        if len(arrays) != len(leads_names):
            raise ValueError(
                f"Number of ECG traces ({len(arrays)}) does not match number of leads ({len(leads_names)})."
            )

        ecg_plot = np.rec.fromarrays(arrays, names=leads_names)
        fig, axs = plt.subplots(
            3, 4, figsize=(10, 13), dpi=120, sharex=True, sharey=True
        )

        time_vector = np.linspace(req_time_ini, req_time_fin, n_times)
        for ax, lead in zip(axs.ravel(), ecg_plot.dtype.names):
            ax.plot(time_vector, ecg_plot[lead], "b", alpha=0.6)
            ax.grid(linestyle="--", alpha=0.4)
            ax.set_title(lead)
            if lead == "V2":
                ax.legend(["Ground truth"], fontsize="8")

        fig.tight_layout()
        plt.show()
