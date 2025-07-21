import numpy as np
from myocardial_mesh.geometry_utils import Bmatrix, localStiffnessMatrix


def test_Bmatrix_output_shape_and_detJ_sign():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    elm = np.array([[0, 1, 2, 3]])

    B, detJ = Bmatrix(pts, elm)

    assert B.shape == (1, 3, 4)
    assert detJ.shape == (1,)
    assert detJ[0] > 0


def test_localStiffnessMatrix_row_sum_near_zero():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    elm = np.array([[0, 1, 2, 3]])
    B, detJ = Bmatrix(pts, elm)
    G = np.tile(np.eye(3), (1, 1, 1))

    K = localStiffnessMatrix(B, detJ, G)

    row_sums = K[0].sum(axis=1)
    assert np.allclose(row_sums, np.zeros(4), atol=1e-12)


def test_Bmatrix_and_localStiffnessMatrix():
    # Define a single tetrahedral element with known coordinates
    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # Node 0
            [1.0, 0.0, 0.0],  # Node 1
            [0.0, 1.0, 0.0],  # Node 2
            [0.0, 0.0, 1.0],  # Node 3
        ]
    )
    elm = np.array([[0, 1, 2, 3]])  # One element with 4 nodes

    # Compute B and detJ
    B, detJ = Bmatrix(pts, elm)

    # Check expected shape and types
    assert B.shape == (1, 3, 4)
    assert detJ.shape == (1,)
    assert detJ[0] > 0  # Valid tetrahedron should have positive volume

    # Use identity tensor for G to test numerical stability
    G = np.tile(np.eye(3), (1, 1, 1))  # Shape (1, 3, 3)

    # Compute local stiffness matrix
    K = localStiffnessMatrix(B, detJ, G)

    # Check shape
    assert K.shape == (1, 4, 4)

    # Stiffness matrix should be symmetric
    assert np.allclose(K[0], K[0].T, atol=1e-12)

    # Row sums should be near zero (physical consistency)
    assert np.allclose(K[0].sum(axis=1), np.zeros(4), atol=1e-12)
