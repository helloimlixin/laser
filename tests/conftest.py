"""Shared pytest fixtures and import-path setup for the LASER test suite."""

import sys
from pathlib import Path

# Make the repo root importable as ``src.*``. pytest.ini also sets
# ``pythonpath = .``; doing it here too is belt-and-suspenders (e.g. when a
# test module is run directly) and lets individual test modules drop their own
# duplicated sys.path bootstrapping.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest


@pytest.fixture
def recording_wandb_trainer():
    """Factory for a rank-zero trainer stub that records ``logger.experiment.log`` calls.

    Usage::

        trainer, calls = recording_wandb_trainer(global_step=7)
        model.__dict__["_trainer"] = trainer
        ...
        # calls is a list of (payload_dict, kwargs_dict) tuples.
    """

    def _make(global_step: int = 0):
        calls: list = []

        class _Experiment:
            def log(self, payload, **kwargs):
                calls.append((dict(payload), dict(kwargs)))

        trainer = type(
            "TrainerStub",
            (),
            {
                "is_global_zero": True,
                "global_step": int(global_step),
                "datamodule": None,
                "logger": type("LoggerStub", (), {"experiment": _Experiment()})(),
            },
        )()
        return trainer, calls

    return _make
