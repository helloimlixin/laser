"""
Deprecated compatibility wrapper for the old scratch `laser.py` entrypoint.

The canonical implementation now lives in `proto.py` for the training pipeline
and `spatial_prior.py` for the stage-2 visual autoregressive prior.
"""

from proto import *  # noqa: F401,F403

try:
    from spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig
except ModuleNotFoundError:  # pragma: no cover - compatibility import path
    from laser_transformer import SpatialDepthPrior, SpatialDepthPriorConfig


# Compatibility aliases for older scripts that imported these symbols from
# scratch/laser.py. The legacy flattened-token prior implementation is gone.
DictionaryLearning = DictionaryLearningTokenized
PatchDictionaryLearning = PatchDictionaryLearningTokenized
Prior = SpatialDepthPrior
PriorConfig = SpatialDepthPriorConfig
TransformerPrior = SpatialDepthPrior
TransformerConfig = SpatialDepthPriorConfig


def main() -> None:
    from proto import main as proto_main

    proto_main()


if __name__ == "__main__":
    main()
