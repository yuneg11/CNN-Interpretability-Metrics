from .captum import get_captum_attribution
from .captum import (
    Saliency,
    GuidedBackprop,
    Deconvolution,
    Occlusion,
    LayerGradCam,
    GuidedGradCam,
    IntegratedGradients,
)


method_list = [
    "Saliency",
    "GuidedBackprop",
    "Deconvolution",
    "Occlusion",
    "LayerGradCam",
    "GuidedGradCam",
    "IntegratedGradients",
]

__all__ = [
    "get_captum_attribution",
    "method_list",
    *method_list
]
