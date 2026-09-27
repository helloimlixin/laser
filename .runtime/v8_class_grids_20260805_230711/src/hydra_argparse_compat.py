"""Work around Hydra + Python 3.14 argparse validation for --shell-completion.

Hydra 1.3.x uses a LazyCompletionHelp instance as ``help=`` for that flag. Python 3.14's
``ArgumentParser._check_help`` uses ``'%' not in help``, which raises on that object.
Fixed on Hydra main; call :func:`patch_argparse_for_hydra_on_py314` before ``import hydra``
until a released hydra-core includes the fix.
"""

import argparse
import sys


def patch_argparse_for_hydra_on_py314() -> None:
    if sys.version_info < (3, 14):
        return

    orig = argparse.ArgumentParser._check_help  # type: ignore[assignment]

    def _check_help(self, action):  # type: ignore[no-untyped-def]
        h = action.help
        if h is not None and type(h).__name__ == "LazyCompletionHelp":
            return None
        return orig(self, action)

    argparse.ArgumentParser._check_help = _check_help  # type: ignore[method-assign]
