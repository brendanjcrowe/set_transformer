from set_transformer.rl.feature_extractors.statistical import (
    CGFExtractor,
    GaussianExtractor,
    KMomentsExtractor,
)
from set_transformer.rl.feature_extractors.pooling import (
    DeepSetExtractor,
    PointNetExtractor,
)
from set_transformer.rl.feature_extractors.st import SetTransformerExtractor

# The three belief encoders the cross-domain experiment compares. They were
# moved out of experiments/ant_tag/4_train_rl_{cgf,st,gaussian}.py (which
# still re-export them, so saved SB3 checkpoints keep loading) because they
# contain no domain-specific logic and a second domain must not have to
# import an Ant-Tag script — that would pull in MuJoCo and collide on the
# numbered flat module names.
from set_transformer.rl.feature_extractors.cgf import (
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.gaussian import (
    WeightedGaussianFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.st import (
    STFeatureLoggingCallback,
    SetTransformerFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.pooled import (
    PointNetFeaturesExtractor,
    WeightedDeepSetFeaturesExtractor,
    WeightedKMomentsFeaturesExtractor,
    reload_pretrained_pooled,
)

__all__ = [
    "GaussianExtractor",
    "KMomentsExtractor",
    "CGFExtractor",
    "DeepSetExtractor",
    "PointNetExtractor",
    "SetTransformerExtractor",
    "STFeatureLoggingCallback",
    "SetTransformerFeaturesExtractor",
    "TNormLoggingCallback",
    "WeightedCGFFeaturesExtractor",
    "WeightedGaussianFeaturesExtractor",
    "WeightedDeepSetFeaturesExtractor",
    "PointNetFeaturesExtractor",
    "WeightedKMomentsFeaturesExtractor",
    "reload_pretrained_pooled",
]
