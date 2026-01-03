"""Plotting utilities for ECG visualization.

This module provides helper functions to visualize 12-lead ECG recordings
returned by ``MyocardialMesh.new_get_ecg(record_array=True)``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

_LOGGER = logging.getLogger(__name__)


def ecg_12lead_plot(
    ecg_rec: NDArray[np.void],
    *,
    figsize: Tuple[float, float] = (10, 13),
    dpi: int = 120,
    suptitle: Optional[str] = None,
) -> Tuple[Figure, NDArray[np.object_]]:
    """Plot a 12-lead ECG recording in a 3×4 grid of subplots.

    Args:
        ecg_rec: Structured NumPy array returned by
            ``MyocardialMesh.new_get_ecg(record_array=True)``.
        figsize: Figure size in inches. Defaults to ``(10, 13)``.
        dpi: Matplotlib DPI for the figure. Defaults to ``120``.
        suptitle: Optional figure title.

    Returns:
        Tuple[Figure, NDArray[Axes]]: The created Matplotlib figure and a
        3×4 array of axes.
    """
    if ecg_rec.dtype.names is None:
        raise TypeError(
            "Expected a structured ECG array with named fields; "
            "call new_get_ecg(record_array=True)."
        )

    names = ecg_rec.dtype.names
    _LOGGER.info("Plotting 12-lead ECG with %d leads", len(names))

    fig, axs = plt.subplots(3, 4, figsize=figsize, dpi=dpi, sharex=True, sharey=True)
    axs = np.asarray(axs)

    for ax, lead in zip(axs.ravel(), names):
        y = np.asarray(ecg_rec[lead], dtype=float)
        ax.plot(y, linewidth=1.0)
        ax.grid(True)
        ax.set_title(lead)
        _LOGGER.debug("Plotted lead %s with %d samples", lead, y.shape[0])

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig, axs
