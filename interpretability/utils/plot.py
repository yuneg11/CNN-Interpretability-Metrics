import math

import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, SymLogNorm


def plot_images(images_dict, filename=None, nrows=None, ncols=None, figsize=None, cmap="jet"):
    fig_num = len(images_dict)
    if ncols is None and nrows is None:
        ncols = math.ceil(math.sqrt(fig_num))
        nrows = math.ceil(fig_num / ncols)
    elif ncols is not None and nrows is None:
        nrows = math.ceil(fig_num / ncols)
    elif ncols is None:
        ncols = math.ceil(fig_num / nrows)

    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = ax.ravel()

    for idx, (name, image) in enumerate(images_dict.items()):
        axes[idx].set_title(name)
        axes[idx].imshow(image, cmap=cmap)
        axes[idx].axis("off")

    plt.tight_layout()

    if filename == "inline" or filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close(fig)


def plot_degradation_results(results_dict, filename=None, ncols=None, nrows=None, title=None):
    r"""
    Args:
        results_dict: Dict[Tuple[Tensor, Tensor, Float]] -> Dict key = Heatmap Name
                                                            Tuple = (MoRF, LeRF, Score)
                                                            MoRF = 1D Tensor
                                                            LeRF = 1D Tensor
                                                            Score = Float
            EX) results_dict = {
                    "GradCam": (Tensor([1.0, 0.4, 0.3, 0.2, 0.1, 0.0]), Tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.0]), 0.5),
                    "Saliency": (Tensor([1.0, 0.5, 0.4, 0.3, 0.2, 0.0]), Tensor([1.0, 0.8, 0.7, 0.6, 0.5, 0.0]), 0.3),
                }
    """

    fig_num = len(results_dict)
    if ncols is None and nrows is None:
        ncols = math.ceil(math.sqrt(fig_num))
        nrows = math.ceil(fig_num / ncols)
    elif ncols is not None and nrows is None:
        nrows = math.ceil(fig_num / ncols)
    elif ncols is None:
        ncols = math.ceil(fig_num / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True, squeeze=False)
    if title is not None:
        fig.suptitle(title)

    for method_idx, (name, (morf, lerf, score)) in enumerate(results_dict.items()):
        x_axis = torch.linspace(0, 1, morf.shape[0]).numpy()
        morf, lerf = morf.cpu().numpy(), lerf.cpu().numpy()

        target_axes = axes[method_idx // ncols, method_idx % ncols]
        target_axes.set(title=name, xlabel="x̄ - degradation of x", ylabel="s(x̄) - scaled score")
        target_axes.plot(x_axis, lerf, label="LeRF")
        target_axes.plot(x_axis, morf, label="MoRF")
        target_axes.fill_between(x_axis, morf, lerf, facecolor="gray", alpha=0.5)
        target_axes.text(0.45, 0.48, f"{score:.3f}")
        target_axes.grid()
        target_axes.legend()

    for idx in range(len(results_dict), ncols * nrows):
        target_axes = axes[idx // ncols, idx % ncols]
        target_axes.axis("off")

    if filename == "inline" or filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close(fig)


def plot_heatmaps(heatmaps_dict, filename=None, vertical=False, cmap="BlWhRd", outlier_percent=0.02):
    r"""
    Args:
        heatmaps_dict: Dict[Tensor] -> Tensor shape = N x C x H x W
                                       Dict key = Heatmap Name

            EX) heatmaps_dict = {
                    "GradCam": Tensor(8 x 3 x 224 x 224),
                    "Saliency": Tensor(8 x 3 x 224 x 224),
                }
    """

    nrows = next(iter(heatmaps_dict.values())).shape[0]
    ncols = len(heatmaps_dict)

    if vertical:
        nrows, ncols = ncols, nrows

    figsize = (3 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for method_idx, (name, heatmaps) in enumerate(heatmaps_dict.items()):
        if vertical:
            axes[method_idx, 0].set_title(name, rotation="vertical", x=-0.05, y=0)
        else:
            axes[0, method_idx].set_title(name)

        heatmaps = heatmaps.cpu()
        if len(heatmaps.shape) == 4:
            if heatmaps.shape[1] <= 3:
                heatmaps = heatmaps.permute(0, 2, 3, 1)
        else:
            heatmaps = heatmaps.unsqueeze(dim=3)

        if cmap == "BlWhRd":
            cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
        elif cmap == "RdWhGr":
            cmap = LinearSegmentedColormap.from_list("", ["red", "white", "green"])

        for image_idx, heatmap in enumerate(heatmaps):
            target_axes = axes[image_idx, method_idx] if not vertical else axes[method_idx, image_idx]
            target_axes.axis("off")
            if name == "Input":
                target_axes.imshow(heatmap)
            else:
                heatmap_combined = heatmap.sum(dim=2)
                sorted_values = heatmap_combined.abs().flatten().sort()[0]
                cumulative_sums = sorted_values.cumsum(dim=0)
                threshold_idx = (cumulative_sums >= cumulative_sums[-1] * (1 - outlier_percent)).nonzero()[0, 0]
                heatmap_normed = (heatmap_combined / sorted_values[threshold_idx]).clamp(min=-1, max=1)

                # mean = heatmap_normed.mean()
                # if mean < 0.1:
                #     norm = Normalize(vmin=-1, vmax=1)
                # elif mean > 0.25:
                #     norm = PowerNorm(gamma=1.0, vmin=-1, vmax=1)
                # else:
                #     norm = SymLogNorm(linthresh=0.3, linscale=1.0, base=10, vmin=-1, vmax=1)
                norm = PowerNorm(gamma=1.0, vmin=-1, vmax=1)

                target_axes.imshow(heatmap_normed, cmap=cmap, norm=norm)

    fig.tight_layout()

    if filename == "inline" or filename is None:
        fig.show()
    else:
        fig.savefig(filename)
        plt.close(fig)
