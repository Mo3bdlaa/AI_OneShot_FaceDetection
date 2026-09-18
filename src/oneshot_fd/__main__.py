"""Allows ``python -m oneshot_fd``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
