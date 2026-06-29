#!/usr/bin/env python3
"""Bootstrap installer for Linux/macOS when shell scripts have Windows CRLF endings.

Usage:
    python3 install.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIX_SCRIPT = ROOT / "scripts" / "fix_line_endings.py"
INSTALL_SH = ROOT / "install.sh"


def main() -> int:
    if not INSTALL_SH.is_file():
        print(f"error: missing {INSTALL_SH}", file=sys.stderr)
        return 1

    fix = subprocess.run(
        [sys.executable, str(FIX_SCRIPT), "--quiet"],
        cwd=ROOT,
        check=False,
    )
    if fix.returncode != 0:
        return fix.returncode

    install = subprocess.run(
        ["bash", str(INSTALL_SH)],
        cwd=ROOT,
        check=False,
    )
    return install.returncode


if __name__ == "__main__":
    raise SystemExit(main())
