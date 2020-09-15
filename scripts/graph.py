import os
import sys
from argparse import ArgumentParser

import torch

# To import interpretability packages
sys.path.append(os.getcwd())

from interpretability.metrics import degradation
from interpretability.utils.plot import plot_degradation_results


method_class = {
    # "DeconvNet": None,
    # "Saliency": None,
    # "GuidedBackprop": None,
    # "wPC": None,
    # "LRP": None,
    # "CLRP": None,
    # "GradCAM": None,
    # "PCCAM": None,
    # "GuidedGradCAM": None,
    # "GuidedPCCAM": None,
    # "Occlusion": None,
    # "iLRP": None,
    # "iSaliency": None,
    # "iGuidedBackprop": None,
    # "iIntegratedGradients": None,
    # "iSmoothGrad": None,
    # "iGradCAM": None,
    # "iGuidedGradCAM": None,
    # "iOcclusion": None,
    # "iPSBR": None,
}


def main(raw_args):
    parser = ArgumentParser(description="Degradation Graph Plot")
    parser.add_argument("-s", "--save-dir", required=True, help="Path to load result")
    parser.add_argument("-o", "--output", default=None, help="Name of result graph output file")
    parser.add_argument("--clamp", action="store_true", default=False,  help="Clamp degradation results from 0 to 1")
    args = parser.parse_args(raw_args)

    if args.output is None:
        args.output = f"{args.save_dir}/result.png"

    file_list = [f for f in os.listdir(args.save_dir) if f.endswith(".pt")]

    for filename in file_list:
        # method_name = filename.split("-")[1].split(".")[0]
        # if method_name in method_class:
        #     method_class[method_name] = filename
        full_name = filename.split("/")[-1].split(".")[0]
        method_class[full_name] = filename

    results = {}
    for method_name in sorted(list(method_class.keys())):
    # for method_name in method_class.keys():
        filename = method_class[method_name]
        if filename is not None:
            data = torch.load(f"{args.save_dir}/{filename}")
            raw_morf, raw_lerf = data["morf"], data["lerf"]

            morf = (raw_morf - raw_morf[-1]) / (raw_morf[0] - raw_morf[-1])
            lerf = (raw_lerf - raw_lerf[-1]) / (raw_lerf[0] - raw_lerf[-1])

            if args.clamp:
                morf.clamp_(min=0, max=1)
                lerf.clamp_(min=0, max=1)

            degradation_score = degradation.score(morf, lerf)
            results[method_name] = (morf.cpu(), lerf.cpu(), degradation_score)

            print(f"{method_name}: {degradation_score:.3f}")

    plot_degradation_results(results, filename=args.output)
    print(f"Result output: '{args.output}'")


if __name__ == "__main__":
    main(sys.argv[1:])
