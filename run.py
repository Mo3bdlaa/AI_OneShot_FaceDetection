#!/usr/bin/env python3
"""Convenience launcher so the project runs straight from a clone.

    python run.py --faces input_faces --source 0

It just puts ``src/`` on the path and hands over to the real CLI, which is
also available as ``python -m oneshot_fd`` after ``pip install -e .``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from oneshot_fd.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
