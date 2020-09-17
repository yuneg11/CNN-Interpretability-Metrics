import torch
from torchvision import models


default_device = "cuda" if torch.cuda.is_available() else "cpu"

models_list = ["vgg16bn", "resnet18", "resnet34", "resnet50", "resnet101", "googlenet", "alexnet"]


def get_model(model_name: str, device: str = default_device):
    lowered_model_name = model_name.lower()

    if lowered_model_name == "vgg16bn":
        model = models.vgg16_bn(pretrained=True).to(device)
        model.options = {
            "target_layer": model.features
        }
    elif lowered_model_name == "resnet18":
        model = models.resnet18(pretrained=True).to(device)
        model.options = {
            "target_layer": model.layer4
        }
    elif lowered_model_name == "resnet34":
        model = models.resnet34(pretrained=True).to(device)
        model.options = {
            "target_layer": model.layer4
        }
    elif lowered_model_name == "resnet50":
        model = models.resnet50(pretrained=True).to(device)
        model.options = {
            "target_layer": model.layer4
        }
    elif lowered_model_name == "resnet101":
        model = models.resnet101(pretrained=True).to(device)
        model.options = {
            "target_layer": model.layer4
        }
    elif lowered_model_name == "googlenet":
        model = models.googlenet(pretrained=True).to(device)
        model.options = {
            "target_layer": model.inception5b
        }
    elif lowered_model_name == "alexnet":
        model = models.alexnet(pretrained=True).to(device)
        model.options = {
            "target_layer": model.features
        }
    else:
        raise ValueError(f"Invalid model name '{model_name}' (Supported models: {models_list})")

    return model.eval()
