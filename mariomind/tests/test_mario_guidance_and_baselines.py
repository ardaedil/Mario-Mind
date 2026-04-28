import subprocess
import sys
from pathlib import Path

from mariomind.envs.mario_dependency import INSTALL_COMMAND, INSTALL_GUIDANCE


def test_install_guidance_contains_command():
    assert INSTALL_COMMAND in INSTALL_GUIDANCE
    assert "gym-super-mario-bros" in INSTALL_GUIDANCE
    assert "nes-py" in INSTALL_GUIDANCE


def test_run_baselines_dummy_writes_artifacts(tmp_path: Path):
    out = tmp_path / "baseline_dummy"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_baselines.py",
            "--env",
            "dummy",
            "--episodes",
            "2",
            "--output-dir",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in ["metrics.csv", "summary.csv", "steps.csv", "config.json", "baseline_report.md"]:
        assert (out / name).exists()


def test_check_mario_setup_graceful_when_missing_deps():
    proc = subprocess.run(
        [sys.executable, "scripts/check_mario_setup.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        text = proc.stdout + proc.stderr
        assert "Missing Mario dependencies" in text
        assert INSTALL_COMMAND in text
