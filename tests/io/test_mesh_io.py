from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
import vtk
from enum import Enum
from unittest.mock import patch, MagicMock

from myocardial_mesh.io.mesh_io import MeshIO, WriteMethod


@pytest.fixture
def simple_polydata() -> vtk.vtkPolyData:
    """Create a minimal vtkPolyData with polygonal faces."""
    sphere = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
    return sphere.extract_geometry()


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


def test_read_obj(tmp_path: Path):
    """Create and read an OBJ file using PyVista."""
    mesh = pv.Sphere()
    obj_path = tmp_path / "sphere.obj"
    mesh.save(obj_path)

    loaded = MeshIO.read_obj(obj_path)
    assert isinstance(loaded, pv.PolyData)
    assert loaded.n_points > 0


def test_integration_obj_to_all_writes(tmp_path: Path):
    """Read an OBJ and save it into all supported formats."""
    sphere = pv.Sphere()
    obj_path = tmp_path / "sphere.obj"
    sphere.save(obj_path)

    mesh = MeshIO.read_obj(obj_path)

    for method, ext in [
        (WriteMethod.VTK, "vtu"),
        (WriteMethod.PYVISTA, "vtp"),
        (WriteMethod.MESHIO, "vtu"),
    ]:
        path = tmp_path / f"converted.{ext}"
        result = MeshIO.write(mesh, path, method)
        assert result is True
        assert path.exists()


def test_read_obj_file_not_found():
    with pytest.raises(FileNotFoundError):
        MeshIO.read_obj(Path("non_existent.obj"))


def test_write_with_invalid_extension(tmp_path: Path, simple_polydata: vtk.vtkPolyData):
    """Should fail due to wrong extension for WriteMethod."""
    invalid_path = tmp_path / "bad.txt"
    result = MeshIO.write(simple_polydata, invalid_path, WriteMethod.VTK)
    assert result is False
    assert not invalid_path.exists()


def test_read_obj_with_invalid_file(tmp_path: Path):
    """Should log error or return an empty mesh for corrupted content."""
    bad_obj = tmp_path / "bad.obj"
    bad_obj.write_text("this is not a valid obj")

    mesh = MeshIO.read_obj(bad_obj)
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


def test_read_obj_with_invalid_file_should_raise(tmp_path: Path):
    bad_obj = tmp_path / "bad.obj"
    bad_obj.write_text("INVALID OBJ CONTENT")

    mesh = MeshIO.read_obj(bad_obj)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 0 or mesh.n_cells == 0
