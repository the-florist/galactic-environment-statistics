"""
    End-to-end smoke tests. Each CLI subcommand is run as a fresh subprocess and
    must complete successfully and write at least one plot file. Plots are
    written under a temporary working directory so the repo is left untouched.
"""

import os
import subprocess
import sys

import pytest

COMMANDS = ["growth-factor", "density-profile", "double-distribution"]


def _run(args, tmp_path):
    """Run `python -m gal_env_stats <args>` in a temp cwd, headless."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, "-m", "gal_env_stats", *args],
        cwd=tmp_path, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_runs_and_produces_a_plot(command, tmp_path):
    result = _run([command], tmp_path)
    assert result.returncode == 0, result.stderr
    plots = list((tmp_path / "plots").glob("*.pdf"))
    assert plots, f"{command!r} produced no plot files"


@pytest.mark.parametrize("cosmology", ["concordance", "eds"])
def test_cosmology_override(cosmology, tmp_path):
    result = _run(["--cosmology", cosmology, "growth-factor"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "plots").glob("*.pdf")
