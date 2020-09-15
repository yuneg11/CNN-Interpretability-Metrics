# Interpretability Metrics

## Setup

1. Put ImageNet validation dataset under "data/imagenet/val" folder
2. Install libraries
```bash
# For PIP
pip install -r requirements.txt

# For Conda
conda install --file requirements.txt -y
```


## Run

### Degradation

```bash
# Evaluation Example
python3 scripts/degradation.py Saliency GradCam -m resnet50 -s results/ -d cuda:1

# Draw Graph Example
python3 scripts/graph.py -s results -o ./result.png
```

#### Supported Methods

1. Captum
- Saliency
- GuidedBackprop
- Deconvolution
- Occlusion
- LayerGradCam
- GuidedGradCam
- IntegratedGradients

2. Custom Captum
- OnOffCam
- GuidedOnOffCam

#### Supported Models

- AlexNet
- VGG16
- ResNet18
- ResNet50
- GoogLeNet
