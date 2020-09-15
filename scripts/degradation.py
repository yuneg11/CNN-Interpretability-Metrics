import os
import sys
from argparse import ArgumentParser

import torch

# To import interpretability packages
sys.path.append(os.getcwd())

from interpretability.methods import *
from interpretability.metrics import degradation

from interpretability.utils.data import get_imagenet_loader
from interpretability.utils.models import get_model, models_list


default_device = "cuda" if torch.cuda.is_available() else "cpu"

def main(raw_args):
    parser = ArgumentParser(description="Degradation Graph Plot")
    parser.add_argument("methods", metavar="<methods>", choices=method_list, nargs="+")
    parser.add_argument("-m", "--model-name", metavar="<model-name>", choices=models_list, required=True, help="Name of model to evaluate")
    parser.add_argument("-s", "--save-dir", required=True, help="Path to load/save result")
    parser.add_argument("-i", "--imagenet-dir", default="data/imagenet/val", help="Path to imagenet directory")
    parser.add_argument("-k", "--batch-size", type=int, required=False, default=16, help="Batch size")
    parser.add_argument("-d", "--device", required=False, default=default_device, help="Device to execute evaluation")
    parser.add_argument("-b", "--baseline", required=False, default="mean", help="Baseline of degradation")
    parser.add_argument("-r", "--reduce", required=False, default="absmean", help="Reduce mode")
    parser.add_argument("-l", "--load-checkpoint", action="store_true", default=False, help="Load results from checkpoint directory")
    args = parser.parse_args(raw_args)

    print(f"Device: {args.device}")

    model = get_model(args.model_name, device=args.device)

    for method_name in args.methods:
        checkpoint_prefix = f"{args.save_dir}/{args.model_name}-{method_name}"
        if args.load_checkpoint:
            filenames = [f for f in os.listdir(args.save_dir) if f.endswith(".pt") and f"{args.model_name}-{method_name}-" in f]
            if len(filenames) == 0:
                checkpointer = degradation.Checkpointer(checkpoint_prefix)
                print(f"Cannot find checkpoint for {method_name}")
            else:
                checkpointer = degradation.Checkpointer(checkpoint_prefix, init_file=f"{args.save_dir}/{filenames[0]}")
        else:
            checkpointer = degradation.Checkpointer(checkpoint_prefix)

        # sampler_state = {"idx_cursor": 0, "idx_end": 16*10}
        sampler_state = None
        data_loader = get_imagenet_loader(args.imagenet_dir, batch_size=args.batch_size, normalize=True, shuffle=False, sampler_state=sampler_state)

        init_kwargs = {}
        attribution = eval(method_name)(model, **init_kwargs, attribute_kwargs={})

        morf, lerf = degradation.evaluate(attribution, data_loader, tile_size=14, baseline=args.baseline,
                                          perturb_stride=5, checkpointer=checkpointer, reduce_mode=args.reduce)

        if checkpointer is not None:
            checkpointer.save(morf, lerf, filename=f"{checkpoint_prefix}.pt")


if __name__ == "__main__":
    main(sys.argv[1:])
