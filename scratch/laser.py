"""
Lightweight compatibility frontend for scratch `laser.py`.

`proto.py` remains the canonical training implementation, but the public
`laser.py` entrypoint now defaults to a substantially cheaper autoencoder:
- base width `num_hiddens=64` instead of `128`
- `num_res_layers=1` instead of `2`
- no forced middle attention block
- no extra decoder residual block per level
- capped channel multiplier of `1` instead of `2`

The legacy `proto.py` defaults are still available via `LegacyLASER`.
"""

import proto as _proto
from proto import *  # noqa: F401,F403

try:
    from spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig
except ModuleNotFoundError:  # pragma: no cover - compatibility import path
    from laser_transformer import SpatialDepthPrior, SpatialDepthPriorConfig


LegacyEncoder = Encoder
LegacyDecoder = Decoder
LegacyLASER = LASER

LIGHT_NUM_HIDDENS = 64
LIGHT_NUM_RES_LAYERS = 1
LIGHT_MAX_CH_MULT = 1
LIGHT_DECODER_EXTRA_RESIDUAL_LAYERS = 0
LIGHT_USE_MID_ATTENTION = False


def _ensure_cli_default(flag: str, value: str) -> None:
    import sys

    if any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
        return
    sys.argv.extend([flag, value])


class Encoder(LegacyEncoder):
    def __init__(self, *args, use_mid_attention: bool = LIGHT_USE_MID_ATTENTION, **kwargs):
        kwargs.setdefault("use_mid_attention", use_mid_attention)
        super().__init__(*args, **kwargs)


class Decoder(LegacyDecoder):
    def __init__(
        self,
        *args,
        use_mid_attention: bool = LIGHT_USE_MID_ATTENTION,
        extra_res_blocks: int = LIGHT_DECODER_EXTRA_RESIDUAL_LAYERS,
        **kwargs,
    ):
        kwargs.setdefault("use_mid_attention", use_mid_attention)
        kwargs.setdefault("extra_res_blocks", extra_res_blocks)
        super().__init__(*args, **kwargs)


class LASER(LegacyLASER):
    def __init__(
        self,
        *args,
        num_hiddens: int = LIGHT_NUM_HIDDENS,
        num_residual_layers: int = LIGHT_NUM_RES_LAYERS,
        max_ch_mult: int = LIGHT_MAX_CH_MULT,
        decoder_extra_residual_layers: int = LIGHT_DECODER_EXTRA_RESIDUAL_LAYERS,
        use_mid_attention: bool = LIGHT_USE_MID_ATTENTION,
        **kwargs,
    ):
        kwargs.setdefault("num_hiddens", num_hiddens)
        kwargs.setdefault("num_residual_layers", num_residual_layers)
        kwargs.setdefault("max_ch_mult", max_ch_mult)
        kwargs.setdefault("decoder_extra_residual_layers", decoder_extra_residual_layers)
        kwargs.setdefault("use_mid_attention", use_mid_attention)
        super().__init__(*args, **kwargs)


# Compatibility aliases for older scripts that imported these symbols from
# scratch/laser.py. The legacy flattened-token prior implementation is gone.
DictionaryLearning = DictionaryLearningTokenized
PatchDictionaryLearning = PatchDictionaryLearningTokenized
Prior = SpatialDepthPrior
PriorConfig = SpatialDepthPriorConfig
TransformerPrior = SpatialDepthPrior
TransformerConfig = SpatialDepthPriorConfig
SparseDictAE = LASER


def main() -> None:
    _ensure_cli_default("--num_hiddens", str(LIGHT_NUM_HIDDENS))
    _ensure_cli_default("--num_res_layers", str(LIGHT_NUM_RES_LAYERS))

    _proto.Encoder = Encoder
    _proto.Decoder = Decoder
    _proto.LASER = LASER
    _proto.SparseDictAE = LASER
    _proto.main()


if __name__ == "__main__":
    main()
