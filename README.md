
# purkinje-learning-myocardial-mesh

A Python library for modeling, manipulating, and analyzing Purkinje‑fiber networks embedded in myocardial meshes—useful in computational cardiology, electrophysiology modelling, and bio‑engineering research.

---

## Features

- Parse and manipulate myocardial‑mesh data  
- Integrate Purkinje‑network geometries  
- Designed for computational‑simulation environments  
- Structured for maintainability and versioned releases  
- Automated testing with `pytest` and CI‑friendly config  

---

## Installation

```bash
pip install myocardial-mesh
```

For development:

```bash
pip install -e ".[dev]"
```

All dependencies are declared in `pyproject.toml`.

---

## Usage

```python
from myocardial_mesh import MyocardialMesh

patient_path = "data/patient_01"
device = "cpu"

# Initialise the mesh with LV geometry, electrodes, and fibre directions
myocardial_mesh = MyocardialMesh(
    myo_mesh=f"{patient_path}/crtdemo_mesh_oriented.vtk",
    electrodes_position=f"{patient_path}/electrode_pos.pkl",
    fibers=f"{patient_path}/crtdemo_f0_oriented.vtk",
    device=device,
)

# ----------------------------------------------------------------------
# Example 1: Activate a stimulus and retrieve PMJ activation values
# ----------------------------------------------------------------------
x0_xyz, x0_vals = ...  # coordinates and initial activation values
pmj_values = myocardial_mesh.activate_fim(
    x0_xyz,                # (N, 3) array of stimuli coordinates
    x0_vals,               # (N,) array of stimulus strengths / times
    return_only_pmjs=True, # restrict output to Purkinje–myocardium junctions
)

# pmj_values is now a NumPy/CuPy array on the selected device
print("PMJ activation snapshot:", pmj_values[:5])

# ----------------------------------------------------------------------
# Example 2: Forward-compute the ECG based on the current state
# ----------------------------------------------------------------------
ecg_leads = myocardial_mesh.new_get_ecg(record_array=False).copy()

# ecg_leads is an (time, 12) array if 12-lead configuration was used
print("Lead I peak amplitude:", ecg_leads[:, 0].max())

```

---

## Versioning & Release

This project uses **release-please** to automate versioning and changelog
generation. Follow
[Conventional Commits](https://www.conventionalcommits.org/) to trigger
releases.

Example commit:

```text
feat: add new Purkinje-mesh attachment algorithm
```

---

## Contributing

We welcome pull requests and new ideas.  
Please read the [CONTRIBUTING.md](CONTRIBUTING.md) guide before opening an issue or submitting a PR. It covers code style, branch policy, and the protected‑branch rules enforced for the repository.

---

## License

This project is licensed under the MIT License.
