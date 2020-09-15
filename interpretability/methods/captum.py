from typing import Union, Callable

from captum import attr


def _get_attribute(
    attribute: Callable,
    default_kwargs: dict = {}
) -> Callable:
    def _attribute(*args, **kwargs):
        return attribute(*args, **{**default_kwargs, **kwargs})
    return _attribute


def get_captum_attribution(
    attribution_base: Union[attr.Attribution, str],
    *init_args: list,
    attribute_kwargs: dict = {},
    **init_kwargs: dict
) -> attr.Attribution:
    try:
        if type(attribution_base) is str:
            attribution_base = eval(attribution_base)
    except:
        raise ValueError("Invalid attribution name or class '{attribution_base}'")

    attribution = attribution_base(*init_args, **init_kwargs)
    attribution.attribute = _get_attribute(attribution.attribute, attribute_kwargs)

    return attribution


def Saliency(model, attribute_kwargs={}):
    return get_captum_attribution(attr.Saliency, model, attribute_kwargs=attribute_kwargs)


def GuidedBackprop(model, attribute_kwargs={}):
    return get_captum_attribution(attr.GuidedBackprop, model, attribute_kwargs=attribute_kwargs)


def Deconvolution(model, attribute_kwargs={}):
    return get_captum_attribution(attr.Deconvolution, model, attribute_kwargs=attribute_kwargs)


def Occlusion(model, attribute_kwargs={}):
    # Only for ImageNet dataset
    default_attribute_kwargs = {
        "strides": 14,
        "sliding_window_shapes": (3, 14, 14)
    }
    return get_captum_attribution(attr.Occlusion, model, attribute_kwargs={**default_attribute_kwargs, **attribute_kwargs})


def LayerGradCam(model, layer=None, device_ids=None, attribute_kwargs={}):
    if layer is None:
        layer = model.options["target_layer"]
    return get_captum_attribution(attr.LayerGradCam, model, layer, device_ids, attribute_kwargs=attribute_kwargs)


def GuidedGradCam(model, layer=None, device_ids=None, attribute_kwargs={}):
    if layer is None:
        layer = model.options["target_layer"]
    return get_captum_attribution(attr.GuidedGradCam, model, layer, device_ids, attribute_kwargs=attribute_kwargs)


def IntegratedGradients(model, attribute_kwargs={}):
    return get_captum_attribution(attr.IntegratedGradients, model, attribute_kwargs=attribute_kwargs)
