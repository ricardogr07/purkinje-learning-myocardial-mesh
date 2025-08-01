from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
import vtk
from enum import Enum
from unittest.mock import patch, MagicMock
from vtkmodules.numpy_interface import dataset_adapter as dsa
import warnings
import meshio
from myocardial_mesh.io.mesh_io import MeshIO, WriteMethod


@pytest.fixture
def simple_polydata() -> vtk.vtkPolyData:
    """Create a minimal vtkPolyData with polygonal faces."""
    sphere = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
    return sphere.extract_geometry()


@pytest.mark.parametrize(
    "ext,method,expected",
    [
        (".vtu", WriteMethod.VTK, True),
        (".vtp", WriteMethod.PYVISTA, True),
        (".vtk", WriteMethod.PYVISTA, True),
        (".xml", WriteMethod.MESHIO, True),
        (".xdmf", WriteMethod.MESHIO, True),
        (".txt", WriteMethod.VTK, False),
        (".txt", WriteMethod.PYVISTA, False),
        (".txt", WriteMethod.MESHIO, False),
    ],
)
def test_validate_extension(ext, method, expected):
    path = Path(f"mesh{ext}")
    assert MeshIO._validate_extension(path, method) is expected


def test_write_pyvista(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    path = tmp_path / "test.vtp"
    result = MeshIO.write(simple_polydata, path, WriteMethod.PYVISTA)
    assert result is True
    assert path.exists()


def test_write_meshio(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    path = tmp_path / "test.vtu"
    triangulated = pv.wrap(simple_polydata).triangulate()

    point_data = {"dummy": np.random.rand(triangulated.GetNumberOfPoints())}
    cell_data = {"dummy": [np.random.rand(triangulated.GetNumberOfCells())]}

    result = MeshIO.write(triangulated, path, WriteMethod.MESHIO, point_data, cell_data)
    assert result is True
    assert path.exists()


def test_read_mesh(tmp_path: Path):
    """Create and read an OBJ file using PyVista."""
    mesh = pv.Sphere()
    obj_path = tmp_path / "sphere.obj"
    mesh.save(obj_path)

    loaded = MeshIO.read_mesh(obj_path)
    assert isinstance(loaded, pv.PolyData)
    assert loaded.n_points > 0


def test_integration_obj_to_all_writes(tmp_path: Path):
    """Read an OBJ and save it into all supported formats."""
    sphere = pv.Sphere()
    obj_path = tmp_path / "sphere.obj"
    sphere.save(obj_path)

    mesh = MeshIO.read_mesh(obj_path)

    for method, ext in [
        (WriteMethod.VTK, "vtu"),
        (WriteMethod.PYVISTA, "vtp"),
        (WriteMethod.MESHIO, "vtu"),
    ]:
        path = tmp_path / f"converted.{ext}"
        result = MeshIO.write(mesh, path, method)
        assert result is True
        assert path.exists()


@pytest.mark.parametrize("filename", ["nope.vtp", "also_missing.vtk"])
def test_read_mesh_file_not_found(filename, tmp_path):
    with pytest.raises(FileNotFoundError):
        MeshIO.read_mesh(tmp_path / filename)


def test_write_with_invalid_extension(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    """Should fail due to wrong extension for WriteMethod."""
    invalid_path = tmp_path / "bad.txt"
    result = MeshIO.write(simple_polydata, invalid_path, WriteMethod.VTK)
    assert result is False
    assert not invalid_path.exists()


def test_read_mesh_with_invalid_file(tmp_path: Path):
    """Should log error or return an empty mesh for corrupted content."""
    bad_obj = tmp_path / "bad.obj"
    bad_obj.write_text("this is not a valid obj")

    mesh = MeshIO.read_mesh(bad_obj)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 0 or mesh.n_cells == 0


def test_write_unknown_method(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    class FakeMethod(str, Enum):
        INVALID = "invalid"

    result = MeshIO.write(simple_polydata, tmp_path / "mesh.bad", FakeMethod.INVALID)  # type: ignore
    assert result is False


def test_write_creates_missing_dir(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    missing_dir = tmp_path / "missing_subdir"
    path = missing_dir / "mesh.vtp"
    result = MeshIO.write(simple_polydata, path, WriteMethod.PYVISTA, create_dirs=True)
    assert result is True
    assert path.exists()


def test_write_with_wrong_extension(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    path = tmp_path / "bad.txt"
    result = MeshIO.write(simple_polydata, path, WriteMethod.PYVISTA)
    assert result is False


def test_write_meshio_exception_handling(tmp_path: Path):
    class FakeMesh:
        pass

    result = MeshIO.write(FakeMesh(), tmp_path / "fail.vtu", WriteMethod.MESHIO)  # type: ignore
    assert result is False


def test_write_fails_on_missing_dir_without_create(
    tmp_path: Path, simple_polydata: vtk.vtkPolyData
):
    missing_dir = tmp_path / "nope"
    path = missing_dir / "mesh.vtp"
    result = MeshIO.write(simple_polydata, path, WriteMethod.PYVISTA, create_dirs=False)
    assert result is False


def test_write_with_unknown_method_enum(
    tmp_path: Path, simple_polydata: vtk.vtkPolyData
):
    class FakeMethod(str, Enum):
        BAD = "bad"

    result = MeshIO.write(simple_polydata, tmp_path / "file.bad", FakeMethod.BAD)  # type: ignore
    assert result is False


def test_write_vtk_writer_failure(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    path = tmp_path / "fail.vtu"

    fake_writer = MagicMock()
    fake_writer.Write.return_value = 0

    with patch(
        "myocardial_mesh.io.mesh_io.vtk.vtkXMLPolyDataWriter", return_value=fake_writer
    ):
        result = MeshIO.write(simple_polydata, path, WriteMethod.VTK)
        assert result is False


def test_write_pyvista_exception(tmp_path: Path):
    class Fake:
        pass

    path = tmp_path / "fail.vtp"
    result = MeshIO.write(Fake(), path, WriteMethod.PYVISTA)  # type: ignore
    assert result is False


def test_write_meshio_failure(tmp_path: Path):
    class Fake:
        pass

    result = MeshIO.write(Fake(), tmp_path / "fail.vtu", WriteMethod.MESHIO)  # type: ignore
    assert result is False


def test_read_mesh_with_invalid_file_should_raise(tmp_path: Path):
    bad_obj = tmp_path / "bad.obj"
    bad_obj.write_text("INVALID OBJ CONTENT")

    mesh = MeshIO.read_mesh(bad_obj)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 0 or mesh.n_cells == 0


def test__write_vtk_success(monkeypatch, tmp_path):
    fake_writer = MagicMock()
    fake_writer.Write.return_value = 1
    monkeypatch.setattr(vtk, "vtkXMLPolyDataWriter", lambda: fake_writer)

    # We only need a dummy vtkDataSet here
    mesh = vtk.vtkPolyData()
    target = tmp_path / "out.vtu"
    assert MeshIO._write_vtk(mesh, target) is True


def test__write_vtk_failure(monkeypatch, tmp_path):
    fake_writer = MagicMock()
    fake_writer.Write.return_value = 0
    monkeypatch.setattr(vtk, "vtkXMLPolyDataWriter", lambda: fake_writer)

    mesh = vtk.vtkPolyData()
    target = tmp_path / "out.vtu"
    assert MeshIO._write_vtk(mesh, target) is False


def test__write_pyvista_exception(monkeypatch, tmp_path):
    class BadMesh:
        pass

    # Force pv.wrap to raise when called
    monkeypatch.setattr(
        pv, "wrap", lambda mesh: (_ for _ in ()).throw(RuntimeError("oops"))
    )

    bad = BadMesh()
    target = tmp_path / "bad.vtp"
    assert MeshIO._write_pyvista(bad, target) is False


def make_fake_vtk(ds_has_polygons: bool, poly_array: np.ndarray = None):
    fake = MagicMock()
    wrapped = MagicMock()
    if ds_has_polygons:
        wrapped.Polygons = (
            poly_array if poly_array is not None else np.array([3, 0, 1, 2])
        )
    else:
        del wrapped.Polygons
    wrapped.Points = np.zeros((4, 3))
    fake.dd = wrapped
    return fake, wrapped


def test__write_meshio_missing_polygons(monkeypatch, tmp_path):
    # wrap returns object without Polygons
    monkeypatch.setattr(
        dsa,
        "WrapDataObject",
        lambda mesh: MagicMock(
            **{"Polygons": None}, PointData={}, Points=np.zeros((0, 3))
        ),
    )

    # delete attribute to simulate missing
    def wrap_no_poly(mesh):
        o = MagicMock()
        del o.Polygons
        return o

    monkeypatch.setattr(dsa, "WrapDataObject", wrap_no_poly)

    dummy = MagicMock()
    target = tmp_path / "out.vtu"
    assert MeshIO._write_meshio(dummy, target) is False


def test__write_meshio_bad_shape(monkeypatch, tmp_path):
    # Polygons length not multiple of 4
    arr = np.array([3, 0, 1])  # wrong length

    def wrap_bad(mesh):
        return MagicMock(Polygons=arr, Points=np.zeros((3, 3)))

    monkeypatch.setattr(dsa, "WrapDataObject", wrap_bad)

    dummy = MagicMock()
    target = tmp_path / "out.vtu"
    assert MeshIO._write_meshio(dummy, target) is False


def test__write_meshio_success(monkeypatch, tmp_path):
    # Proper polygons: two triangles [3,0,1,2, 3,2,3,0]
    arr = np.array([3, 0, 1, 2, 3, 2, 3, 0])

    def wrap_good(mesh):
        return MagicMock(Polygons=arr, Points=np.zeros((4, 3)))

    monkeypatch.setattr(dsa, "WrapDataObject", wrap_good)

    # Spy on meshio.Mesh.write
    written = {}

    class DummyMesh:
        def __init__(self, points, cells, point_data, cell_data):
            written["cells"] = cells

        def write(self, path):
            written["path"] = path

    monkeypatch.setattr(meshio, "Mesh", DummyMesh)

    dummy = MagicMock()
    target = tmp_path / "out.vtu"
    assert (
        MeshIO._write_meshio(dummy, target, point_data={"a": 1}, cell_data={"b": 2})
        is True
    )
    assert written["path"] == target


def test_read_mesh_type_mismatch(tmp_path):
    # Write a valid .vtp but force expected_type mismatch
    mesh = pv.Sphere()
    path = tmp_path / "ok.vtp"
    mesh.save(path)
    with pytest.raises(TypeError):
        MeshIO.read_mesh(path, expected_type=pv.UnstructuredGrid)


def test_read_mesh_success(tmp_path):
    mesh = pv.Sphere()
    path = tmp_path / "ok.vtp"
    mesh.save(path)
    loaded = MeshIO.read_mesh(path, expected_type=pv.PolyData)
    assert isinstance(loaded, pv.PolyData)
    assert loaded.n_points == mesh.n_points


def test_load_legacy_vtk_warn_and_return(tmp_path):
    # First, create a simple legacy VTK file
    # Use a PyVista mesh and write in legacy .vtk
    mesh = pv.Sphere()
    vtk_path = tmp_path / "legacy.vtk"
    mesh.save(vtk_path, binary=False)  # default writes XML, but assume legacy for test

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        result = MeshIO.load_legacy_vtk(str(vtk_path))
        # We expect a deprecation warning
        assert any(item.category is DeprecationWarning for item in w)
    assert isinstance(result, pv.UnstructuredGrid) or isinstance(result, pv.PolyData)
    assert result.n_points > 0
