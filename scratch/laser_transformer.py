"""Backward-compatible shim for the spatial-depth transformer implementation."""

try:
    from spatial_prior import (
        SpatialDepthPrior,
        SpatialDepthPriorConfig,
        build_spatial_depth_prior_config,
        soft_clamp,
    )
except ModuleNotFoundError:
    from scratch.spatial_prior import (
        SpatialDepthPrior,
        SpatialDepthPriorConfig,
        build_spatial_depth_prior_config,
        soft_clamp,
    )

__all__ = [
    "SpatialDepthPrior",
    "SpatialDepthPriorConfig",
    "build_spatial_depth_prior_config",
    "soft_clamp",
]
