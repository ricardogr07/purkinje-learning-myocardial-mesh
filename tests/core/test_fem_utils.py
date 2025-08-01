import numpy as np
from scipy import sparse

from myocardial_mesh.core.fem_utils import (
    compute_Bmatrix,
    compute_local_stiffness_matrix,
    assemble_stiffness_matrix,
)


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

    B, detJ = compute_Bmatrix(pts, elm)

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
    B, detJ = compute_Bmatrix(pts, elm)
    G = np.tile(np.eye(3), (1, 1, 1))

    K = compute_local_stiffness_matrix(B, detJ, G)

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
    B, detJ = compute_Bmatrix(pts, elm)

    # Check expected shape and types
    assert B.shape == (1, 3, 4)
    assert detJ.shape == (1,)
    assert detJ[0] > 0  # Valid tetrahedron should have positive volume

    # Use identity tensor for G to test numerical stability
    G = np.tile(np.eye(3), (1, 1, 1))  # Shape (1, 3, 3)

    # Compute local stiffness matrix
    K = compute_local_stiffness_matrix(B, detJ, G)

    # Check shape
    assert K.shape == (1, 4, 4)

    # Stiffness matrix should be symmetric
    assert np.allclose(K[0], K[0].T, atol=1e-12)

    # Row sums should be near zero (physical consistency)
    assert np.allclose(K[0].sum(axis=1), np.zeros(4), atol=1e-12)


def test_assemble_stiffness_matrix_single_element():
    # Define a single tetrahedral element
    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # Node 0
            [1.0, 0.0, 0.0],  # Node 1
            [0.0, 1.0, 0.0],  # Node 2
            [0.0, 0.0, 1.0],  # Node 3
        ]
    )
    elm = np.array([[0, 1, 2, 3]])
    # Use identity elasticity (one element)
    G = np.tile(np.eye(3), (1, 1, 1))  # shape (1,3,3)

    K_global = assemble_stiffness_matrix(pts, elm, G)

    # Should be a CSR sparse matrix of size (4,4)
    assert isinstance(K_global, sparse.csr_matrix)
    assert K_global.shape == (4, 4)

    # For a single tet, the global stiffness is symmetric
    A = K_global.toarray()
    assert np.allclose(A, A.T, atol=1e-12)

    # And each row should sum (almost) to zero
    row_sums = A.sum(axis=1)
    assert np.allclose(row_sums, np.zeros(4), atol=1e-12)


def test_compute_Bmatrix_batch_elements_and_zero_detJ():
    # Create two identical tetrahedral elements and one degenerate (zero volume)
    base_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Element 0 and 1: proper tet; Element 2: all nodes co-planar → detJ=0
    pts = base_pts
    elm = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 0],
        ]
    )

    B, detJ = compute_Bmatrix(pts, elm)

    # Expect three entries in batch dimension
    assert B.shape == (3, 3, 4)
    assert detJ.shape == (3,)

    # First two have positive volume, last is degenerate
    assert detJ[0] > 0
    assert detJ[1] > 0
    assert np.isclose(detJ[2], 0.0, atol=1e-12)

    # Now pass through local stiffness: should not blow up on zero-det element
    # Use identity G for all three
    G = np.tile(np.eye(3), (3, 1, 1))
    K = compute_local_stiffness_matrix(B, detJ, G)

    # Should produce a (3,4,4) array
    assert K.shape == (3, 4, 4)

    # For degenerate element, values may be inf or nan – at least ensure code ran
    assert not np.all(np.isnan(K[2])) or np.any(np.isinf(K[2]))
