import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: extract_token_cache.py requires Python >= 3.10. "
        "Set PYTHON_BIN to a supported environment or run cache.py through scripts/run.sh."
    )

from cache import *  # noqa: F401,F403
from cache import _token_cache_meta  # noqa: F401


if __name__ == "__main__":
    from cache import main

    main()
