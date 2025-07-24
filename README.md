# purkinje-learning-myocardial-mesh

A Python library for modeling, manipulating, and analyzing Purkinje fiber networks embedded in myocardial meshes — useful in computational cardiology, electrophysiology modeling, and bioengineering research.

---

## Features

- Parse and manipulate myocardial mesh data  
- Integrate Purkinje network geometries  
- Designed for computational simulation environments  
- Structured for maintainability and versioned releases  
- Automated testing with `pytest` and CI-friendly config  

---

## Installation

```bash
pip install myocardial-mesh
```

For development:

```bash
pip install -e ".[dev]"
```

Dependencies and development requirements are managed via `pyproject.toml`.

---

## Usage

Here’s a basic example (assumes functional API in `myocardial_mesh` or similar module):

```python
from myocardial_mesh import load_mesh, attach_purkinje_network

mesh = load_mesh("data/mesh.vtk")
network = attach_purkinje_network(mesh)
network.simulate()
```

---

## Versioning & Release

This project uses `release-please` to automate versioning and changelog generation.  
Follow [Conventional Commits](https://www.conventionalcommits.org/) to trigger releases.

Example commit:

```csharp
feat: add new Purkinje-mesh attachment algorithm
```

---

## License

This project is licensed under the terms of the MIT license.