import sys
import os
import torch


Lib = None


def init_env(repository_path="cloned/grad-cam-pytorch"):
    global Lib

    sys.path.append(os.path.join(os.getcwd(), repository_path))

    import grad_cam as lib

    Lib = lib


init_env()


class Wrapper():
    def __init__(self, MethodClass, model, default_reduce, **generate_kwargs):
        self.MethodClass = MethodClass
        self.model = model
        self.default_reduce = default_reduce
        self.generate_kwargs = generate_kwargs

    def heatmap_postprocess(self, heatmaps, reduce_channel):
        if reduce_channel == "abs":
            heatmaps = heatmaps.abs().max(dim=1)[0]
        elif reduce_channel == "max":
            heatmaps = heatmaps.max(dim=1)[0]
        elif reduce_channel == "mean":
            heatmaps = heatmaps.mean(dim=1)

        view_shape = (-1, 1, 1, 1)[:len(heatmaps.shape)]
        mins = heatmaps.view(heatmaps.shape[0], -1).min(dim=1)[0].view(view_shape)
        maxs = heatmaps.view(heatmaps.shape[0], -1).max(dim=1)[0].view(view_shape)
        abs_maxs = torch.max(mins.abs(), maxs.abs())

        return heatmaps / abs_maxs

    def interpret(self, inputs, labels=None, reduce_channel=True):
        reduce_channel = self.default_reduce if reduce_channel == True else reduce_channel

        method = self.MethodClass(self.model.model)
        method._encode_one_hot = get_encode_one_hot(method)

        probs, ids = method.forward(inputs)

        class_scores = torch.empty_like(probs).scatter_(dim=1, index=ids, src=probs)

        if labels is None:
            labels = class_scores.argmax(dim=1, keepdim=True)

        method.backward(ids=labels)
        heatmaps = method.generate(**self.generate_kwargs)

        method.remove_hook()

        heatmaps = self.heatmap_postprocess(heatmaps, reduce_channel)

        return heatmaps, class_scores


class BackPropagation(Wrapper):
    def __init__(self, model):
        super().__init__(Lib.BackPropagation, model, "abs")


class DeconvNet(Wrapper):
    def __init__(self, model):
        super().__init__(Lib.Deconvnet, model, "abs")


class GradCAM(Wrapper):
    def __init__(self, model):
        target_layer = ""
        for name, module in model.model.named_modules():
            if id(module) == id(model.last_conv):
                target_layer = name
        super().__init__(Lib.GradCAM, model, "mean", target_layer=target_layer)


class GuidedBackPropagation(Wrapper):
    def __init__(self, model):
        super().__init__(Lib.GuidedBackPropagation, model, "abs")


class GuidedGradCAM(Wrapper):
    def __init__(self, model):
        self.GradCAM = GradCAM(model)
        self.GuidedBackPropagation = GuidedBackPropagation(model)

    def interpret(self, inputs, labels=None, reduce_channel=True):
        grad_cam_heatmaps, class_scores = self.GradCAM.interpret(inputs, labels, reduce_channel)
        gbp_heatmaps, _ = self.GuidedBackPropagation.interpret(inputs, labels, reduce_channel)

        heatmaps = self.heatmap_postprocess(grad_cam_heatmaps * gbp_heatmaps, False)

        return heatmaps, class_scores


class Occlusion(Wrapper):
    def __init__(self, model):
        raise NotImplementedError


def get_encode_one_hot(self):
    def _encode_one_hot(labels):
        src = torch.ones((labels.shape[0], 1), device=self.device)
        return torch.zeros_like(self.logits, device=self.device) \
                    .scatter_(dim=1, index=labels, src=src)
    return _encode_one_hot
