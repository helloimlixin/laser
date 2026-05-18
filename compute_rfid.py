import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: compute_rfid.py requires Python >= 3.10. "
        "Set PYTHON_BIN to a supported environment."
    )

from scripts.tools.compute_rfid import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.tools.compute_rfid import main

    main()
