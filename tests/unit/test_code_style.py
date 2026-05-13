# Purpose: Enforce concise Python file comments across project-owned code.

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "scripts",
)


def test_python_files_start_with_single_purpose_comment() -> None:
    failures: list[str] = []
    for path in _python_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            failures.append(f"{_rel(path)} is empty")
            continue
        if not lines[0].startswith("# Purpose: ") or len(lines[0]) <= len("# Purpose: "):
            failures.append(f"{_rel(path)} must start with a single '# Purpose:' line")
            continue
        module = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(module) is not None:
            failures.append(f"{_rel(path)} must not use a module docstring")

    assert failures == []


def test_python_files_have_no_standalone_comments_after_header() -> None:
    failures: list[str] = []
    for path in _python_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines[1:], start=2):
            if line.lstrip().startswith("#"):
                failures.append(f"{_rel(path)}:{line_number}")

    assert failures == []


def _python_files() -> list[Path]:
    return sorted(path for root in PYTHON_ROOTS for path in root.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()
