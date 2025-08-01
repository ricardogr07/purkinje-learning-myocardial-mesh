import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import warnings

import meshio
import pyvista as pv
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa

logger = logging.getLogger(__name__)


class WriteMethod(str, Enum):
    """Available mesh writing backends."""

    VTK = "vtk"
    PYVISTA = "pyvista"
    MESHIO = "meshio"


class MeshIO:
    """Unified interface to read and write VTK-compatible meshes."""

    @staticmethod
    def write(
        vtk_mesh: vtk.vtkDataSet,
        file_path: Path,
        method: WriteMethod,
        point_data: Optional[dict[str, Any]] = None,
        cell_data: Optional[dict[str, Any]] = None,
        create_dirs: bool = False,
    ) -> bool:
        """Unified mesh writer interface."""
        file_path = Path(file_path)
        logger.debug(f"Preparing to write mesh to {file_path} using method: {method}")

        # Validate file extension
        if not MeshIO._validate_extension(file_path, method):
            logger.error(
                f"File extension does not match write method {method}: {file_path.suffix}"
            )
            return False

        # Validate or create directory
        if not file_path.parent.exists():
            if create_dirs:
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {file_path.parent}")
                except Exception as e:
                    logger.error(f"Failed to create directory {file_path.parent}: {e}")
                    return False
            else:
                logger.warning(f"Directory does not exist: {file_path.parent}")
                return False

        # Dispatch
        try:
            match method:
                case WriteMethod.VTK:
                    return MeshIO._write_vtk(vtk_mesh, file_path)
                case WriteMethod.PYVISTA:
                    return MeshIO._write_pyvista(vtk_mesh, file_path)
                case WriteMethod.MESHIO:
                    return MeshIO._write_meshio(
                        vtk_mesh, file_path, point_data, cell_data
                    )
                case _:
                    logger.error(f"Unknown write method: {method}")
                    return False
        except Exception as e:
            logger.error(f"Exception while writing mesh with {method}: {e}")
            return False

    @staticmethod
    def _validate_extension(file_path: Path, method: WriteMethod) -> bool:
        ext = file_path.suffix.lower()
        logger.debug(f"Validating extension '{ext}' for method {method}")
        if method == WriteMethod.VTK and ext != ".vtu":
            return False
        if method == WriteMethod.PYVISTA and ext not in {".vtu", ".vtk", ".vtp"}:
            return False
        if method == WriteMethod.MESHIO and ext not in {
            ".vtk",
            ".vtu",
            ".xml",
            ".xdmf",
        }:
            return False
        return True

    @staticmethod
    def _write_vtk(vtk_mesh: vtk.vtkDataSet, file_path: Path) -> bool:
        logger.debug(f"Calling VTK writer for file: {file_path}")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(vtk_mesh)
        result = writer.Write()
        if result == 1:
            logger.info(f"Mesh written successfully to {file_path} using VTK.")
            return True
        else:
            logger.warning(f"VTK writer failed to write to {file_path}")
            return False

    @staticmethod
    def _write_pyvista(vtk_mesh: vtk.vtkDataSet, file_path: Path) -> bool:
        logger.debug(f"Calling PyVista writer for file: {file_path}")
        try:
            mesh = pv.wrap(vtk_mesh)
            mesh.save(file_path)
            logger.info(f"Mesh saved to {file_path} using PyVista.")
            return True
        except Exception as e:
            logger.error(f"Failed to save with PyVista to {file_path}: {e}")
            return False

    @staticmethod
    def _write_meshio(
        vtk_mesh: vtk.vtkDataSet,
        file_path: Path,
        point_data: Optional[dict[str, Any]] = None,
        cell_data: Optional[dict[str, Any]] = None,
    ) -> bool:
        logger.debug(f"Calling meshio writer for file: {file_path}")
        try:
            dd = dsa.WrapDataObject(vtk_mesh)

            if not hasattr(dd, "Polygons"):
                logger.error("Wrapped mesh does not have 'Polygons' attribute.")
                return False

            points = dd.Points  # ndarray
            polygons = (
                dd.Polygons
            )  # flat array like [3, i0, i1, i2, 3, i3, i4, i5, ...]

            if polygons.ndim != 1 or len(polygons) % 4 != 0:
                logger.error(f"Polygons array shape invalid: {polygons.shape}")
                return False

            cells = polygons.reshape((-1, 4))[:, 1:]  # Remove the leading '3'

            logger.debug(
                f"MeshIO - Points shape: {points.shape}, Cells shape: {cells.shape}"
            )

            mesh = meshio.Mesh(
                points=points,
                cells={"triangle": cells},
                point_data=point_data or {},
                cell_data=cell_data or {},
            )

            mesh.write(file_path)
            logger.info(f"Mesh written to {file_path} using meshio.")
            return True
        except Exception as e:
            logger.error(f"Failed to write mesh with meshio to {file_path}: {e}")
            return False

    @staticmethod
    def read_mesh(
        file_path: Path,
        expected_type: type[pv.DataSet] = pv.DataSet,
    ) -> pv.DataSet:
        """
        Read a mesh file and return it as a PyVista object.

        Parameters
        ----------
        file_path : Path
            Path to the mesh file.
        expected_type : type, optional
            Expected PyVista type (e.g., pv.PolyData or pv.UnstructuredGrid).
            Defaults to pv.DataSet (no type check).

        Returns
        -------
        pv.DataSet
            The mesh object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        TypeError
            If the mesh is not of the expected type.
        """
        file_path = Path(file_path)
        logger.info(f"Reading mesh file: {file_path}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"{file_path} does not exist")

        try:
            mesh = pv.read(file_path)
            if not isinstance(mesh, expected_type):
                logger.error(
                    f"Expected {expected_type.__name__}, but got {type(mesh).__name__}"
                )
                raise TypeError(
                    f"Expected {expected_type.__name__}, but got {type(mesh).__name__}"
                )
            logger.info(
                f"Mesh loaded from {file_path} as {type(mesh).__name__} with {mesh.n_points} points"
            )
            return mesh
        except Exception as e:
            logger.exception(f"Error reading mesh file {file_path}: {e}")
            raise

    @staticmethod
    def load_legacy_vtk(file_path: str) -> pv.UnstructuredGrid:
        """
        Load a legacy VTK file using the VTK DataSetReader.

        .. deprecated:: 1.0.0
        This method uses `vtk.vtkDataSetReader` and should only be used if `pyvista.read()` fails.
        Use `pyvista.read()` or `MeshIO.read_mesh()` instead for automatic format detection.

        Parameters
        ----------
        file_path : str
            Path to the `.vtk` file in legacy format.

        Returns
        -------
        pv.UnstructuredGrid
            The loaded VTK mesh as a PyVista UnstructuredGrid.
        """
        warnings.warn(
            "load_legacy_vtk() is deprecated and should only be used for legacy VTK files. "
            "Use MeshIO.read_mesh() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        reader = vtk.vtkDataSetReader()
        reader.SetFileName(file_path)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        vtk_output = reader.GetOutput()
        return pv.wrap(vtk_output)
