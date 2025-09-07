from __future__ import annotations
import os
import vtk


def load_unstructured(path: str):
    """
    Load a legacy .vtk or XML .vtu unstructured dataset, with a safe
    fallback (try XML, then legacy). Returns a vtkDataSet.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif ext == ".vtk":
        reader = vtk.vtkDataSetReader()
    else:
        # try XML first, then fall back to legacy
        reader = vtk.vtkXMLUnstructuredGridReader()
        try:
            reader.SetFileName(path)
            reader.Update()
            if reader.GetOutput() is None:
                raise RuntimeError
        except Exception:
            reader = vtk.vtkDataSetReader()

    reader.SetFileName(path)
    # legacy readers only expose these; harmless if absent
    if hasattr(reader, "ReadAllVectorsOn"):
        reader.ReadAllVectorsOn()
    if hasattr(reader, "ReadAllScalarsOn"):
        reader.ReadAllScalarsOn()
    reader.Update()

    out = reader.GetOutput()
    if out is None:
        raise RuntimeError(f"Failed to read unstructured mesh: {path}")
    return out


def build_cell_locator(dataset) -> vtk.vtkCellLocator:
    """
    Build a vtkCellLocator for closest-point queries on the dataset.
    """
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(dataset)
    loc.BuildLocator()
    return loc
