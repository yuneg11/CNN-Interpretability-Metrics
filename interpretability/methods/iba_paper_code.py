import sys
import os
import torch


IBA = None
Baselines = None
Estimator = None

repository = None


def init_env(repository_path="cloned/IBA-paper-code"):
    global IBA
    global Baselines
    global Estimator

    global repository

    repository = os.path.join(os.getcwd(), repository_path)
    sys.path.append(repository)

    from attribution_bottleneck import attribution
    from attribution_bottleneck.utils import baselines
    from attribution_bottleneck.bottleneck import estimator

    IBA = attribution
    Baselines = baselines
    Estimator = estimator


init_env()


class Wrapper():
    def __init__(self, model, MethodClass, *args, **kwargs):
        self.model = model
        self.method = MethodClass(*args, **kwargs)

    def interpret(self, inputs, labels=None, reduce_channel=True):
        class_scores = self.model(inputs).detach()

        if labels is None:
            labels = class_scores.argmax(dim=1, keepdim=True)

        if reduce_channel == True:
            shape = (inputs.shape[0], inputs.shape[2], inputs.shape[3])
        else:
            shape = inputs.shape

        heatmaps = torch.empty(shape, device=inputs.device)

        for idx, (input_t, label) in enumerate(zip(inputs, labels)):
            heatmap = torch.from_numpy(self.method.heatmap(input_t.unsqueeze(dim=0), label))
            heatmaps[idx] = heatmap if reduce_channel else heatmap.unsqueeze(dim=0).repeat(3, 1, 1)
        # heatmaps = torch.from_numpy(self.method.heatmap(inputs, labels))

        view_shape = (-1, 1, 1, 1)[:len(heatmaps.shape)]
        mins = heatmaps.view(heatmaps.shape[0], -1).min(dim=1)[0].view(view_shape)
        maxs = heatmaps.view(heatmaps.shape[0], -1).max(dim=1)[0].view(view_shape)
        abs_maxs = torch.max(mins.abs(), maxs.abs())

        return heatmaps / abs_maxs, class_scores


class Gradient(Wrapper):
    def __init__(self, model, **kwargs):
        init_kwargs = {}
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.Gradient, model.model, **init_kwargs)
        # self.method._calc_gradient = get_calc_gradient_hack(self.method)


class Saliency(Wrapper):
    def __init__(self, model, **kwargs):
        init_kwargs = {}
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.Saliency, model.model, **init_kwargs)
        # self.method._calc_gradient = get_calc_gradient_hack(self.method)


class GuidedBackprop(Wrapper):
    def __init__(self, model, **kwargs):
        init_kwargs = {}
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.GuidedBackprop, model.model, **init_kwargs)
        # self.method._calc_gradient = get_calc_gradient_hack(self.method)


class GradCAM(Wrapper):
    def __init__(self, model, **kwargs):
        init_kwargs = {
            "layer": model.last_conv,
            "interp": "bilinear"
        }
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.GradCAM, model.model, **init_kwargs)


class GuidedGradCAM(Wrapper):
    def __init__(self, model, **kwargs):
        init_kwargs = {
            "gradcam_layer": model.last_conv,
            "gradcam_interp": "bilinear"
        }
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.GuidedGradCAM, model.model, **init_kwargs)


class IntegratedGradients(Wrapper):
    def __init__(self, model, iba_gradient_base=IBA.Saliency, **kwargs):
        init_kwargs = {
            "steps": 50,
            "baseline": Baselines.Mean()
        }
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.IntegratedGradients, iba_gradient_base(model.model), **init_kwargs)
        self.method._backpropagate_multiple = get_backpropagate_multiple_hack(self.method)
        # self.mehtod.backprop._calc_gradient = get_calc_gradient_hack(self.mehtod.backprop)


class SmoothGrad(Wrapper):
    def __init__(self, model, iba_gradient_base=IBA.Saliency, **kwargs):
        init_kwargs = {
            "std": 0.15,
            "steps": 50
        }
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.SmoothGrad, iba_gradient_base(model.model), **init_kwargs)
        self.method._backpropagate_multiple = get_backpropagate_multiple_hack(self.method)
        # self.mehtod.backprop._calc_gradient = get_calc_gradient_hack(self.mehtod.backprop)


class Occlusion(Wrapper):
    def __init__(self, model, **kwargs):
        init_kwargs = {
            "size": 14,
            "baseline": Baselines.Mean()
        }
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.Occlusion, model.model, **init_kwargs)


class LRP(Wrapper):
    def __init__(self, model, **kwargs):
        init_kwargs = {
            "eps": -5,
            "beta": -1,
            "device": next(model.model.parameters()).device
        }
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.LRP, model.model, **init_kwargs)


class PerSampleBottleneckReader(Wrapper):
    def __init__(self, model, estimator_layer=None, estimator_weights_path=None, **kwargs):
        """
        This default estimator setting is only for resnet50.
        If you want to use other models, please change the estimator parameters.
        """
        if estimator_layer is None:
            estimator_layer = model.model.layer2
        if estimator_weights_path is None:
            estimator_weights_path = f"{repository}/weights/estimator_resnet50_2.torch"

        estim = Estimator.ReluEstimator(estimator_layer)
        estim.load(estimator_weights_path)

        init_kwargs = {
            "beta": 10,
        }
        init_kwargs.update(kwargs)

        super().__init__(model, IBA.PerSampleBottleneckReader, model.model, estim, **init_kwargs)


# Hack code for torch-based calculation instead of numpy-based calculation
def get_backpropagate_multiple_hack(self):
    def _backpropagate_multiple_hack(inputs: list, target_t: torch.Tensor):
        grads = torch.zeros((len(inputs), *inputs[0].shape))

        for i in range(len(inputs)):
            grad = self.backprop._calc_gradient(input_t=inputs[i], target_t=target_t)

            if len(grad.shape) == 3:
                grad = grad.unsqueeze(dim=0)

            grads[i, :, :, :, :] = grad

        return grads.cpu().numpy()
    return _backpropagate_multiple_hack


# Hack code for batch input support
def get_calc_gradient_hack(self):
    def _calc_gradient_hack(input_t: torch.Tensor, target_t: torch.Tensor):
        self.model.zero_grad()
        img_var = torch.autograd.Variable(input_t, requires_grad=True)
        logits = self.model(img_var)

        one_src = torch.ones((logits.shape[0], 1), device=logits.device)
        grad_eval_point = torch.zeros_like(logits).scatter_(dim=1, index=target_t, src=one_src)
        logits.backward(gradient=grad_eval_point)

        return img_var.grad
    return _calc_gradient_hack
