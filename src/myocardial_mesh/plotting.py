# myocardial_mesh/plotting.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def ecg_12lead_plot(
    ecg_rec: np.ndarray,
    *,
    figsize: Tuple[float, float] = (10, 13),
    dpi: int = 120,
    suptitle: str | None = None,
):
    """
    Parameters
    ----------
    ecg_rec : structured np.ndarray
        Use MyocardialMesh.new_get_ecg(record_array=True).
    figsize : tuple
        Figure size in inches (default (10, 13) to match the notebook).
    dpi : int
        Matplotlib DPI for the figure (default 120).
    suptitle : str | None
        Optional figure title.

    Returns
    -------
    fig, axs : (matplotlib.figure.Figure, np.ndarray[Axes] of shape (3,4))
    """
    if ecg_rec.dtype.names is None:
        raise TypeError(
            "Expected a structured ECG array with named fields; "
            "call new_get_ecg(record_array=True)."
        )

    names = ecg_rec.dtype.names  # keep raw order
    fig, axs = plt.subplots(3, 4, figsize=figsize, dpi=dpi, sharex=True, sharey=True)
    axs = np.asarray(axs)

    for ax, lead in zip(axs.ravel(), names):
        y = np.asarray(ecg_rec[lead], dtype=float)
        ax.plot(y, linewidth=1.0)
        ax.grid(True)
        ax.set_title(lead)

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig, axs
