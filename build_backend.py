"""Minimal stdlib-only PEP 517/660 backend for this dependency-free project.

Keeping the backend local makes ``pip install -e .`` work in a fresh Python
installation without downloading a build tool just to package pure Python.
It is intentionally small: the application code remains under ``src/`` and
the backend only assembles the wheel metadata and package files.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tarfile
import zipfile
from pathlib import Path

NAME = "edd-eval"
VERSION = "0.1.0"
DIST_INFO = "edd_eval-0.1.0.dist-info"
ROOT = Path(__file__).parent
SOURCE = ROOT / "src"


def _metadata() -> bytes:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: A deterministic evaluation harness for AI systems\n"
        "Requires-Python: >=3.10\n"
        "License: MIT\n"
        "Provides-Extra: langchain\n"
        "Requires-Dist: langchain-core>=0.3; extra == 'langchain'\n"
        "Requires-Dist: langsmith>=0.3.13; extra == 'langchain'\n"
        "Provides-Extra: deepeval\n"
        "Requires-Dist: deepeval>=4.2; extra == 'deepeval'\n"
        "Provides-Extra: all\n"
        "Requires-Dist: langchain-core>=0.3; extra == 'all'\n"
        "Requires-Dist: langsmith>=0.3.13; extra == 'all'\n"
        "Requires-Dist: deepeval>=4.2; extra == 'all'\n"
        "\n"
    ).encode()


def _wheel_metadata() -> bytes:
    return b"Wheel-Version: 1.0\nGenerator: edd-eval-build-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n"


def _entry_points() -> bytes:
    return b"[console_scripts]\nedd-eval = edd_eval.cli:main\n"


def _file_hash(payload: bytes) -> tuple[str, int]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    return f"sha256={digest}", len(payload)


def _wheel_files(*, editable: bool) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if editable:
        files["edd_eval.pth"] = (str(SOURCE.resolve()) + os.linesep).encode()
    else:
        for path in sorted((SOURCE / "edd_eval").rglob("*.py")):
            files[path.relative_to(SOURCE).as_posix()] = path.read_bytes()
    files[f"{DIST_INFO}/METADATA"] = _metadata()
    files[f"{DIST_INFO}/WHEEL"] = _wheel_metadata()
    files[f"{DIST_INFO}/entry_points.txt"] = _entry_points()
    record_rows = []
    for path, payload in files.items():
        digest, size = _file_hash(payload)
        record_rows.append(f"{path},{digest},{size}")
    record_rows.append(f"{DIST_INFO}/RECORD,,")
    files[f"{DIST_INFO}/RECORD"] = ("\n".join(record_rows) + "\n").encode()
    return files


def _write_wheel(directory: str, *, editable: bool) -> str:
    filename = f"edd_eval-0.1.0-py3-none-any.whl"
    output = Path(directory) / filename
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, payload in _wheel_files(editable=editable).items():
            archive.writestr(path, payload)
    return filename


def _write_metadata(directory: str) -> str:
    target = Path(directory) / DIST_INFO
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_bytes(_metadata())
    return DIST_INFO


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: dict | None = None) -> str:
    del config_settings
    return _write_metadata(metadata_directory)


def prepare_metadata_for_build_editable(metadata_directory: str, config_settings: dict | None = None) -> str:
    del config_settings
    return _write_metadata(metadata_directory)


def build_wheel(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    return _write_wheel(wheel_directory, editable=False)


def build_editable(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    return _write_wheel(wheel_directory, editable=True)


def get_requires_for_build_wheel(config_settings: dict | None = None) -> list[str]:
    del config_settings
    return []


def get_requires_for_build_editable(config_settings: dict | None = None) -> list[str]:
    del config_settings
    return []


def build_sdist(sdist_directory: str, config_settings: dict | None = None) -> str:
    del config_settings
    filename = f"{NAME}-{VERSION}.tar.gz"
    output = Path(sdist_directory) / filename
    prefix = f"{NAME}-{VERSION}"
    include = [Path("pyproject.toml"), Path("README.md"), Path("LICENSE"), Path("build_backend.py")]
    include.extend(path.relative_to(ROOT) for path in (SOURCE / "edd_eval").rglob("*.py"))
    with tarfile.open(output, "w:gz") as archive:
        for relative in include:
            source = ROOT / relative
            archive.add(source, arcname=f"{prefix}/{relative.as_posix()}")
    return filename


def get_requires_for_build_sdist(config_settings: dict | None = None) -> list[str]:
    del config_settings
    return []
