"""Sphinx configuration for the myocardial mesh documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "purkinje-learning-myocardial-mesh"
author = "purkinje-learning-myocardial-mesh contributors"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build"]

html_theme = "furo"
html_static_path = ["_static"]
