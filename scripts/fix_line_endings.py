#!/usr/bin/env python3
"""Normalize Windows CRLF line endings to Unix LF for shell scripts.

Checkout on Windows can leave *.sh files with \\r\\n, which breaks Linux/macOS
with errors like ``/bin/bash^M: bad interpreter``. Run this before shell scripts:

    python3 scripts/fix_line_endings.py
    ./install.sh

Or use the bootstrap wrapper: ``python3 install.py``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_GLOBS = ("*.sh",)
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "graphify-out",
    ".cursor",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_target_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            files.append(path)
    return sorted(set(files))


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def normalize_crlf(data: bytes) -> tuple[bytes, bool]:
    if b"\r" not in data:
        return data, False
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return normalized, normalized != data


def fix_file(path: Path, *, dry_run: bool = False) -> bool:
    """Return True if the file had CRLF (or stray CR) line endings."""
    original = read_bytes(path)
    normalized, changed = normalize_crlf(original)
    if not changed:
        return False
    if not dry_run:
        path.write_bytes(normalized)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert CRLF/CR line endings to LF in shell scripts.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to fix (default: repo *.sh files)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report files that need fixing and exit 1 if any are found",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print output when files are changed or --check finds issues",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    if args.paths:
        targets: list[Path] = []
        for item in args.paths:
            item = item.resolve() if item.is_absolute() else (Path.cwd() / item).resolve()
            if item.is_dir():
                targets.extend(iter_target_files(item, DEFAULT_GLOBS))
            elif item.is_file():
                targets.append(item)
            else:
                print(f"skip: not found: {item}", file=sys.stderr)
        targets = sorted(set(targets))
    else:
        targets = iter_target_files(root, DEFAULT_GLOBS)

    needs_fix: list[Path] = []
    for path in targets:
        if fix_file(path, dry_run=True):
            needs_fix.append(path)

    if args.check:
        if needs_fix:
            if not args.quiet:
                print("CRLF line endings found:")
                for path in needs_fix:
                    print(f"  {path.relative_to(root)}")
            return 1
        if not args.quiet:
            print("All shell scripts use LF line endings.")
        return 0

    fixed: list[Path] = []
    for path in needs_fix:
        fix_file(path, dry_run=False)
        fixed.append(path)

    if fixed and not args.quiet:
        print(f"Fixed line endings in {len(fixed)} file(s):")
        for path in fixed:
            print(f"  {path.relative_to(root)}")
    elif not args.quiet and not fixed:
        print("No CRLF line endings found in shell scripts.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
