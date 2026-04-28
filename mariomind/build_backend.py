"""Minimal local PEP 517 backend to support offline editable installs in this environment."""

from __future__ import annotations

import base64
import csv
import hashlib
import os
from pathlib import Path
import zipfile

NAME = "mariomind"
VERSION = "0.1.0"


def _dist_info_dir() -> str:
    return f"{NAME}-{VERSION}.dist-info"


def _wheel_name() -> str:
    return f"{NAME}-{VERSION}-py3-none-any.whl"


def _hash_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _build_wheel(wheel_directory: str) -> str:
    wheel_directory_path = Path(wheel_directory)
    wheel_directory_path.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_directory_path / _wheel_name()

    entries: list[tuple[str, bytes]] = []

    src_root = Path("src") / NAME
    for file_path in src_root.rglob("*.py"):
        arcname = str(Path(NAME) / file_path.relative_to(src_root))
        entries.append((arcname, file_path.read_bytes()))

    dist = _dist_info_dir()
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: Experimental RL lab for Mario World 1-1\n"
    ).encode("utf-8")
    wheel = (
        "Wheel-Version: 1.0\n"
        "Generator: custom-backend\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode("utf-8")
    entries.append((f"{dist}/METADATA", metadata))
    entries.append((f"{dist}/WHEEL", wheel))

    record_rows = []
    for path, data in entries:
        record_rows.append((path, _hash_bytes(data), str(len(data))))

    record_content = []
    for row in record_rows:
        record_content.append(",".join(row))
    record_content.append(f"{dist}/RECORD,,")
    record_bytes = ("\n".join(record_content) + "\n").encode("utf-8")
    entries.append((f"{dist}/RECORD", record_bytes))

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, data in entries:
            zf.writestr(path, data)

    return wheel_path.name


def build_wheel(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    return _build_wheel(wheel_directory)


def build_editable(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    return _build_wheel(wheel_directory)


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings=None):
    dist = _dist_info_dir()
    out = Path(metadata_directory) / dist
    out.mkdir(parents=True, exist_ok=True)
    (out / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n",
        encoding="utf-8",
    )
    return dist


def prepare_metadata_for_build_editable(metadata_directory: str, config_settings=None):
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)
