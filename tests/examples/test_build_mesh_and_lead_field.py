from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_mesh_and_lead_field_cli_smoke(tmp_path):
    script = REPO_ROOT / "examples" / "build_mesh_and_lead_field.py"
    screenshot = tmp_path / "plot.png"
    cmd = [
        sys.executable,
        str(script),
        "--sample-pmjs",
        "16",
        "--skip-plot",
        "--screenshot",
        str(screenshot),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed with {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    assert (
        not screenshot.exists()
    ), "Screenshot should not be created when --skip-plot is set."


def test_build_mesh_and_lead_field_with_plot(tmp_path):
    """Test that the plotting functionality works with a temporary output file."""
    script = REPO_ROOT / "examples" / "build_mesh_and_lead_field.py"
    screenshot = tmp_path / "plot_with_rendering.png"
    cmd = [
        sys.executable,
        str(script),
        "--sample-pmjs",
        "16",
        "--screenshot",
        str(screenshot),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed with {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    assert screenshot.exists(), "Screenshot should be created when plotting is enabled."
    assert screenshot.stat().st_size > 0, "Screenshot file should not be empty."
