"""Create an isolated evaluation environment and install its dependencies."""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
ENV_DIR = EVAL_DIR / ".venv"
REQUIREMENTS = EVAL_DIR / "requirements.txt"


def main() -> None:
    if not ENV_DIR.exists():
        print(f"Creating evaluation environment at {ENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(ENV_DIR)

    python = ENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print("Dependency installation failed. Enable package network access and rerun this script.", file=sys.stderr)
        raise SystemExit(error.returncode) from error
    print(f"Environment ready. Run: {python} eval/evaluate.py")


if __name__ == "__main__":
    main()
