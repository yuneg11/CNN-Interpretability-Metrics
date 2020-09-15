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


try:
    import sys
    import os

    ml = None

    def init_env(repository_path="cloned/p"):
        global ml

        sys.path.append(os.path.join(os.getcwd(), repository_path))

        from attrs import method_list as pml

        ml = pml

    init_env()

    methods = __import__("attrs", globals(), locals(), ml, 0)

    method_list.extend(ml)

    if ml:
        for method in ml:
            exec(f"{method} = methods.{method}")

except:
    pass


try:
    from .grad_cam_pytorch import BackPropagation as gBackPropagation
    from .grad_cam_pytorch import DeconvNet as gDeconvNet
    from .grad_cam_pytorch import GradCAM as gGradCAM
    from .grad_cam_pytorch import GuidedBackPropagation as gGuidedBackPropagation
    from .grad_cam_pytorch import GuidedGradCAM as gGuidedGradCAM

    method_list.extend([
        "gBackPropagation", "gDeconvNet", "gGradCAM", "gGuidedBackPropagation", "gGuidedGradCAM"
    ])

except:
    pass


try:
    from .iba_paper_code import Gradient as iGradient
    from .iba_paper_code import Saliency as iSaliency
    from .iba_paper_code import GuidedBackprop as iGuidedBackprop
    from .iba_paper_code import GradCAM as iGradCAM
    from .iba_paper_code import GuidedGradCAM as iGuidedGradCAM
    from .iba_paper_code import IntegratedGradients as iIntegratedGradients
    from .iba_paper_code import SmoothGrad as iSmoothGrad
    from .iba_paper_code import Occlusion as iOcclusion
    from .iba_paper_code import LRP as iLRP
    from .iba_paper_code import PerSampleBottleneckReader as iPSBR

    method_list.extend([
        "iGradient", "iSaliency", "iGuidedBackprop", "iGradCAM", "iGuidedGradCAM",
        "iIntegratedGradients", "iSmoothGrad", "iOcclusion", "iLRP", "iPSBR"
    ])

except:
    pass


__all__ = [
    "method_list",
    *method_list
]
