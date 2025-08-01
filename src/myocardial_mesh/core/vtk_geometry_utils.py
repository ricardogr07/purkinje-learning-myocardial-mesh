import numpy as np
import logging
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa


logger = logging.getLogger(__name__)


class VTKGeometryUtils:
    @staticmethod
    def find_closest_pmjs(pmjs: np.ndarray, locator: vtk.vtkCellLocator) -> np.ndarray:
        """
        Project PMJs onto the mesh using a VTK spatial locator.

        Parameters
        ----------
        pmjs : np.ndarray
            Array of shape (N, 3) with PMJ coordinates.
        locator : vtk.vtkCellLocator
            Locator built on the mesh.

        Returns
        -------
        np.ndarray
            Array of projected PMJ coordinates.
        """
        cellId = vtk.reference(0)
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        ppmjs = np.zeros_like(pmjs)

        for k in range(pmjs.shape[0]):
            locator.FindClosestPoint(pmjs[k, :], ppmjs[k, :], cellId, subId, d)

        return ppmjs

    @staticmethod
    def probe_activation(mesh: vtk.vtkDataSet, query_points: np.ndarray) -> np.ndarray:
        """
        Interpolate the 'activation' field at selected query points on a VTK mesh.

        Parameters
        ----------
        mesh : vtk.vtkDataSet
            Mesh containing the 'activation' field.
        query_points : np.ndarray
            N×3 array of spatial locations.

        Returns
        -------
        np.ndarray
            Interpolated activation values at the query points.
        """
        logger.debug("Probing activation at selected locations...")

        # Create vtkPolyData from points
        vtk_points = vtk.vtkPoints()
        for p in query_points:
            vtk_points.InsertNextPoint(p)
        vtk_poly = vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_points)

        # Probe activation field from source mesh
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(mesh)
        probe.SetInputData(vtk_poly)
        probe.Update()

        output = dsa.WrapDataObject(probe.GetOutput())
        activation = output.PointData["activation"]

        return np.array(activation)
