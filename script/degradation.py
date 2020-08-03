import os
import sys

import torch
import click

# To import interpretability packages
sys.path.append(os.getcwd())

from interpretability.method import *
from interpretability.metric import degradation

from interpretability.util.data import get_imagenet_loader
from interpretability.util.model import get_model, models_list


default_device = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("methods", metavar="METHODS", type=click.Choice(method_list), required=True, nargs=-1)
@click.option("-m", "--model-name", type=click.Choice(models_list), required=True, help="Name of model to evaluate")
@click.option("-i", "--imagenet-dir", type=click.STRING, required=False, default="data/sample", help="Path to imagenet directory")
@click.option("-k", "--batch-size", type=click.INT, required=False, default=16, help="Batch size")
@click.option("-d", "--device", type=click.STRING, required=False, default=default_device, help="Device to execute evaluation")
@click.option("-b", "--baseline", type=click.STRING, required=False, default="mean", help="Baseline of degradation")
@click.option("-l", "--load-checkpoint", type=click.BOOL, required=False, default=False, help="Load results from checkpoint directory")
@click.option("-c", "--checkpoint-dir", type=click.STRING, required=False, help="Path to load/save checkpoint")
def main(imagenet_dir, batch_size, device, model_name, methods, baseline, load_checkpoint, checkpoint_dir=None):
    print(f"Device: {device}")

    model = get_model(model_name, device=device)

    for method_name in methods:
        if checkpoint_dir is None:
            checkpointer, checkpoint_prefix = None, None
            print("Checkpointer is disabled!")
        else:
            checkpoint_prefix = f"{checkpoint_dir}/{model_name}-{method_name}"
            if load_checkpoint:
                filenames = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") and f"{model_name}-{method_name}-" in f]
                if len(filenames) == 0:
                    checkpointer = degradation.Checkpointer(checkpoint_prefix)
                    print(f"Cannot find checkpoint for {method_name}")
                else:
                    checkpointer = degradation.Checkpointer(checkpoint_prefix, init_file=f"{checkpoint_dir}/{filenames[0]}")
            else:
                checkpointer = degradation.Checkpointer(checkpoint_prefix)

        data_loader = get_imagenet_loader(imagenet_dir, batch_size=batch_size, normalize=True, shuffle=False)

        init_kwargs = {}
        attribution = eval(method_name)(model, **init_kwargs, attribute_kwargs={})

        morf, lerf = degradation.evaluate(attribution, data_loader, tile_size=14, baseline=baseline, perturb_stride=5,
                                          desc=method_name, checkpointer=checkpointer)

        if checkpointer is not None:
            checkpointer.save(morf, lerf, filename=f"{checkpoint_prefix}.pt")


if __name__ == "__main__":
    main()
