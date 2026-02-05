"""
AstroLab Models
==============

Neural network models for astronomical data processing.
"""

# Core models
from .astro_model import AstroModel

# Autoencoders
from .base_model import AstroBaseModel

# Layers - Other
from .layers import (  # Point cloud layers; Pooling layers; Base layers; Convolution layers; Normalization layers; Graph layers; Heterogeneous layers
    AdaptivePointCloudLayer,
    AdaptivePooling,
    AstronomicalGraphConv,
    AstroPointCloudLayer,
    AttentivePooling,
    BaseGraphLayer,
    BaseLayer,
    BasePoolingLayer,
    BatchNorm,
    FlexibleGraphConv,
    GraphNorm,
    GraphPooling,
    HeteroGNNLayer,
    HierarchicalPooling,
    InstanceNorm,
    LambdaPooling,
    LayerNorm,
    MultiScalePointCloudEncoder,
    MultiScalePooling,
    StatisticalPooling,
)

# Layers - Heads
from .mixins.analysis import ModelAnalysisMixin

# Mixins
from .mixins.explainability import ExplainabilityMixin

# Encoders


# Layers - Decoders

# Layers - Encoders


__all__ = [
    "AstroModel",
    "AstroBaseModel",
    "ExplainabilityMixin",
    "ModelAnalysisMixin",
]

# For layers, encoders, decoders, heads: import from their respective submodules.
