
# Contributing to purkinje-learning-myocardial-mesh

Thank you for considering a contribution.
This project aims to provide a reliable, well‑tested toolkit for myocardial mesh generation, lead‑field computation, and related Bayesian optimisation workflows. The guidelines below help maintain consistency, quality, and reproducibility.

---

## 1. Code of conduct

All participants are expected to adhere to the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Please report unacceptable behaviour to the maintainers.

---

## 2. Getting started

### 2.1 Clone and set up a virtual environment

```bash
git clone https://github.com/ricardogr07/purkinje-learning-myocardial-mesh.git
cd purkinje-learning-myocardial-mesh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2.2 Install and run pre‑commit hooks

The repository uses **pre‑commit** to format code (Black), sort imports (ruff `I` rules), and lint (ruff).

```bash
pip install pre-commit     # run once per machine/environment
pre-commit install         # installs the git hook
pre-commit run --all-files # optional first-time full run
```

If a hook fails, address the reported issues and recommit.

---

## 3. Branching and pull‑request policy

`main` is a protected branch. **Direct pushes are prohibited.**
To contribute:

1. Create a feature branch
   ```bash
   git checkout -b feature/my-change
   ```
2. Commit your work (hooks run automatically).
3. Push the branch and open a Pull Request (PR) against `main`.
4. Ensure all required checks pass (CI, tests, code scanning, coverage).
5. Request review from a maintainer.
6. After approval, merge using the **“Create a merge commit”** button.
   Squash‑and‑merge or rebase‑and‑merge is disabled to preserve linear history.

### 3.1 Repository rules enforced by GitHub

| Rule | Requirement |
|------|-------------|
| **PRs only** | All changes must arrive via PR; no direct pushes to `main`. |
| **Linear history** | Merge commits only; no merge‑conflict commits pushed. |
| **Signed commits** | Every commit must have a verified GPG or SSH signature. |
| **Status checks** | CI tests, linting, coverage, code scanning, and deployments must succeed. |
| **No force push / delete** | `main` cannot be force‑pushed or deleted. |

---

## 4. Coding standards

| Aspect       | Tool(s)                                    | Standard / Rule                                          | Example |
|--------------|--------------------------------------------|----------------------------------------------------------|---------|
| Formatter    | [Black]                                    | PEP 8 auto-formatting                                    | `black src/` |
| Import order | [ruff] (`I` rules)                         | *isort*-compatible grouping and alphabetical sorting     | After `ruff --select I --fix`:<br>```python<br>import os<br>import numpy as np<br>from .core import Mesh<br>``` |
| Linting      | [ruff] (all rules enabled)                 | Zero warnings permitted                                  | `ruff check src/` must exit 0 |
| Typing       | [mypy]             | Python 3.10 type hints;<br>`mypy --strict` in CI         | `mypy --strict src/` exits 0 |
| Docstrings   | [pydocstyle] / ruff `D` rules              | Google style                                             | ```python <br> def add(a: int, b: int) -> int:<br>    \"\"\"Add two integers.\"\"\" <br>``` |
| Tests        | [pytest] + `pytest-cov`                    | ≥ 90 % line coverage                                     | `pytest --cov=myocardial_mesh` |
| Commit msgs  | Conventional Commits (used by *release-please*) | Imperative, `<type>(scope): description`                | `feat(mesh): add CuPy backend for lead-field solver` |

[Black]: https://black.readthedocs.io/en/stable/
[ruff]: https://github.com/astral-sh/ruff
[mypy]: https://mypy.readthedocs.io/en/stable/
[pydocstyle]: https://www.pydocstyle.org/en/stable/

---

## 5. Running the test suite

```bash
pytest --cov=myocardial_mesh --cov-report=term-missing
```

All tests must pass locally before opening a PR. CI will reject coverage below the project threshold.

---

## 6. Updating documentation

Documentation is built with **Sphinx** and the **furo** theme.

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The `-W` flag treats warnings as errors; the CI docs workflow uses the same
setting.

---

## 7. Filing issues and feature requests

* Search existing issues before opening a new one.
* Provide minimal reproduction steps or code snippets.
* For feature requests, describe the use‑case and desired behaviour.

---

## 8. Contact

Questions or requests can be raised via GitHub Issues or by contacting the maintainers listed in `pyproject.toml`.
