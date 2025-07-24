import numpy as np
import logging


logger = logging.getLogger(__name__)


def Bmatrix(pts: np.ndarray, elm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the strain-displacement matrix (B-matrix) and the determinant of the Jacobian for a tetrahedral finite element.

    Args:
        pts (np.ndarray): Array of nodal coordinates with shape (num_nodes, 3).
        elm (np.ndarray): Array of element node indices (typically shape (4,) for a tetrahedron).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - B (np.ndarray): The strain-displacement matrix of shape (3, 4, ...), where ... denotes possible batch dimensions.
            - detJ (np.ndarray): The determinant of the Jacobian for the element(s), shape (...).
    """
    logger.debug("Computing B-matrix and determinant of Jacobian.")
    nodeCoords = np.moveaxis(pts[elm, :], 0, -1)

    # NOTE: The definition of the parent tetrahedron by Hughes - "The Finite Element Method" (p. 170) is different from the
    # local order given by VTK (http://victorsndvg.github.io/FEconv/formats/vtk.xhtml)
    # Here we follow the VTK convention

    x1 = nodeCoords[0, 0, :]
    x2 = nodeCoords[1, 0, :]
    x3 = nodeCoords[2, 0, :]
    x4 = nodeCoords[3, 0, :]
    y1 = nodeCoords[0, 1, :]
    y2 = nodeCoords[1, 1, :]
    y3 = nodeCoords[2, 1, :]
    y4 = nodeCoords[3, 1, :]
    z1 = nodeCoords[0, 2, :]
    z2 = nodeCoords[1, 2, :]
    z3 = nodeCoords[2, 2, :]
    z4 = nodeCoords[3, 2, :]
    x14 = x1 - x4
    x34 = x3 - x4
    x24 = x2 - x4
    y14 = y1 - y4
    y34 = y3 - y4
    y24 = y2 - y4
    z14 = z1 - z4
    z34 = z3 - z4
    z24 = z2 - z4
    detJ = (
        x14 * (y34 * z24 - z34 * y24)
        - y14 * (x34 * z24 - z34 * x24)
        + z14 * (x34 * y24 - y34 * x24)
    )
    Jinv_11 = y34 * z24 - y24 * z34
    Jinv_12 = -1.0 * (x34 * z24 - x24 * z34)
    Jinv_13 = x34 * y24 - x24 * y34
    Jinv_21 = -1.0 * (y14 * z24 - y24 * z14)
    Jinv_22 = x14 * z24 - x24 * z14
    Jinv_23 = -1.0 * (x14 * y24 - x24 * y14)
    Jinv_31 = y14 * z34 - y34 * z14
    Jinv_32 = -1.0 * (x14 * z34 - x34 * z14)
    Jinv_33 = x14 * y34 - x34 * y14
    Jinv = np.array(
        [
            [Jinv_11, Jinv_12, Jinv_13],
            [Jinv_21, Jinv_22, Jinv_23],
            [Jinv_31, Jinv_32, Jinv_33],
        ]
    )
    B_def = np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0], [0.0, 1.0, 0.0, -1.0]]
    )
    B = Jinv.T @ B_def[..., :]
    logger.debug("B-matrix and detJ computed successfully.")
    return B, detJ


def localStiffnessMatrix(B: np.ndarray, J: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Computes the local stiffness matrix for a finite element mesh.

    Args:
        B (np.ndarray): The strain-displacement matrix of shape (n_elements, n_nodes, n_dim).
        J (np.ndarray): The Jacobian determinants for each element, shape (n_elements,).
        G (np.ndarray): The material property matrix (e.g., elasticity tensor), shape (n_elements, n_dim, n_dim).

    Returns:
        np.ndarray: The local stiffness matrices for each element, shape (n_elements, n_nodes, n_nodes).
    """
    logger.debug("Computing local stiffness matrix.")
    K = np.einsum("nji,njk,nkl->nil", B, G, B) / 6.0
    K = 1 / J[:, None, None] * K
    logger.debug("Local stiffness matrix computed successfully.")
    return K
