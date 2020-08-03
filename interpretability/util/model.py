import torch
from torchvision import models


default_device = "cuda" if torch.cuda.is_available() else "cpu"

models_list = ["vgg16", "resnet18", "resnet50", "googlenet"]


def get_model(model_name: str, device: str = default_device):
    # model_pool = pc_models if pc_model else models
    model_pool = models

    lowered_model_name = model_name.lower()

    if lowered_model_name == "vgg16":
        model = model_pool.vgg16(pretrained=True).to(device)
        model.options = {
            "target_layer": model.features
        }
    elif lowered_model_name == "resnet18":
        model = model_pool.resnet18(pretrained=True).to(device)
        model.options = {
            "target_layer": model.layer4
        }
    elif lowered_model_name == "resnet50":
        model = model_pool.resnet50(pretrained=True).to(device)
        model.options = {
            "target_layer": model.layer4
        }
    elif lowered_model_name == "googlenet":
        model = model_pool.googlenet(pretrained=True).to(device)
        model.options = {
            "target_layer": model.inception5b
        }
    else:
        raise ValueError(f"Invalid model name '{model_name}' (Supported models: {models_list})")

    return model.eval()
