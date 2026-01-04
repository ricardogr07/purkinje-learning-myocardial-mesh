"""Utilities for loading unstructured VTK datasets and building cell locators.

This module provides helper functions to load VTK unstructured meshes
(``.vtk`` legacy or ``.vtu`` XML) and to build ``vtkCellLocator`` objects
for spatial queries.
"""

from __future__ import annotations

from typing import cast

import logging
import os
import vtk
from vtkmodules.vtkCommonDataModel import vtkDataSet

_LOGGER = logging.getLogger(__name__)


def load_unstructured(path: str) -> vtkDataSet:
    """Load a legacy ``.vtk`` or XML ``.vtu`` unstructured dataset.

    Attempts to read the file as XML first when the extension is ambiguous,
    then falls back to the legacy reader if needed.

    Args:
        path: Path to the unstructured mesh file.

    Returns:
        vtkDataSet: The loaded unstructured VTK dataset.

    Raises:
        RuntimeError: If the file cannot be read or the reader returns no output.
    """
    _LOGGER.info("Loading unstructured mesh from %s", path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtu":
        _LOGGER.debug("Using vtkXMLUnstructuredGridReader for .vtu file")
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif ext == ".vtk":
        _LOGGER.debug("Using vtkDataSetReader for .vtk file")
        reader = vtk.vtkDataSetReader()
    else:
        _LOGGER.debug("Unknown extension %s; trying XML reader first", ext)
        reader = vtk.vtkXMLUnstructuredGridReader()
        try:
            reader.SetFileName(path)
            reader.Update()
            if reader.GetOutput() is None:
                raise RuntimeError
        except Exception:
            _LOGGER.debug("XML reader failed; falling back to legacy reader")
            reader = vtk.vtkDataSetReader()

    reader.SetFileName(path)
    # Legacy readers only expose these; harmless if absent
    if hasattr(reader, "ReadAllVectorsOn"):
        _LOGGER.debug("Enabling ReadAllVectorsOn")
        reader.ReadAllVectorsOn()
    if hasattr(reader, "ReadAllScalarsOn"):
        _LOGGER.debug("Enabling ReadAllScalarsOn")
        reader.ReadAllScalarsOn()
    reader.Update()

    out_obj = reader.GetOutput()
    if out_obj is None:
        _LOGGER.error("Failed to read unstructured mesh: %s", path)
        raise RuntimeError(f"Failed to read unstructured mesh: {path}")

    _LOGGER.info(
        "Successfully loaded unstructured mesh with %d points",
        out_obj.GetNumberOfPoints(),
    )
    out = cast(vtkDataSet, out_obj)
    return out


def build_cell_locator(dataset: vtkDataSet) -> vtk.vtkCellLocator:
    """Build a ``vtkCellLocator`` for closest-point queries on a dataset.

    Args:
        dataset: Input VTK dataset on which the locator will operate.

    Returns:
        vtk.vtkCellLocator: A built cell locator ready for spatial queries.
    """
    _LOGGER.info("Building vtkCellLocator for dataset")
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(dataset)
    loc.BuildLocator()
    _LOGGER.debug(
        "vtkCellLocator built with dataset containing %d points",
        dataset.GetNumberOfPoints(),
    )
    return loc
