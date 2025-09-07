from pathlib import Path
import numpy as np
from scipy import sparse

from myocardial_mesh import MyocardialMesh
from myocardial_mesh.core.fem_utils import assemble_K

DATA = Path("data/crtdemo")
NB = DATA / "nb"


def _build_myo():
    return MyocardialMesh(
        myo_mesh=str(NB / "True_endo.vtu"),  # small and has activation
        electrodes_position=str(DATA / "electrode_pos.pkl"),
        fibers=str(DATA / "crtdemo_f0_oriented.vtk"),
        device="cpu",
    )


def test_assemble_K_shape_and_symmetry():
    myo = _build_myo()
    K = assemble_K(np.asarray(myo.xyz, float), myo.cells, myo.Gi_cell)
    assert isinstance(K, sparse.csr_matrix)
    assert K.shape == (len(myo.xyz), len(myo.xyz))

    # sparse symmetry check without densifying
    diff = (K - K.T).tocsr()
    max_abs = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
    assert max_abs < 1e-10


def test_assemble_K_determinism():
    myo = _build_myo()
    K1 = assemble_K(np.asarray(myo.xyz, float), myo.cells, myo.Gi_cell)
    K2 = assemble_K(np.asarray(myo.xyz, float), myo.cells, myo.Gi_cell)

    # exact structural equality
    assert K1.nnz == K2.nnz
    assert np.array_equal(K1.indices, K2.indices)
    assert np.array_equal(K1.indptr, K2.indptr)
    # numerical equality
    assert np.allclose(K1.data, K2.data)
