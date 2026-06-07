#!/usr/bin/env python3
"""Run all module demos."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODULES = [
    ROOT / "01_sentiment" / "run.py",
    ROOT / "03_matrix_completion" / "run.py",
]


def main() -> int:
    for script in MODULES:
        print("\n" + "=" * 50)
        result = subprocess.run([sys.executable, str(script)], cwd=script.parent)
        if result.returncode != 0:
            return result.returncode
    print("\n" + "=" * 50)
    print("02_mnist requires download — run separately: python 02_mnist/run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
