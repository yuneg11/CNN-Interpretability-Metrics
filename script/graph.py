import os
from os import path
import sys

import torch
import click

# To import interpretability packages
sys.path.append(os.getcwd())

from interpretability.metric import Degradation

from interpretability.util.data import save_degradation_results


default_checkpoint_dir = "result/degradation"

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


@click.command()
@click.option("-t", "--clamp", type=click.BOOL, required=False, default=False, help="Clamp degradation results from 0 to 1")
@click.option("-c", "--checkpoint-dir", type=click.STRING, required=False, default=default_checkpoint_dir, help="Path to load/save checkpoint")
@click.option("-o", "--output", type=click.STRING, required=False, help="Name of result graph output file")
def main(clamp, checkpoint_dir, output=None):
    if output is None:
        output = f"{checkpoint_dir}/result.png"

    file_list = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

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
            data = torch.load(f"{checkpoint_dir}/{filename}")
            if type(data) is dict:
                raw_morf, raw_lerf = data["morf"], data["lerf"]
            else:
                # For legacy support
                raw_morf, raw_lerf = data[0], data[1]

            morf = (raw_morf - raw_morf[-1]) / (raw_morf[0] - raw_morf[-1])
            lerf = (raw_lerf - raw_lerf[-1]) / (raw_lerf[0] - raw_lerf[-1])

            if clamp:
                morf.clamp_(min=0, max=1)
                lerf.clamp_(min=0, max=1)

            degradation_score = Degradation.score(morf, lerf)
            results[method_name] = (morf.cpu(), lerf.cpu(), degradation_score)

            print(f"{method_name}: {degradation_score:.3f}")

    save_degradation_results(output, results)
    print(f"Result output: '{output}'")


if __name__ == "__main__":
    main()
