import torch

from torch.utils.data import Sampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, SymLogNorm


class PartialSampler(Sampler):
    def __init__(self, data_source, idx_start=None, idx_end=None, verbose=True):
        self.data_source = data_source

        if idx_start is not None:
            assert idx_start < len(data_source), \
                    f"Start index '{idx_start}' is larger than the length of dataset '{len(data_source)}'"
            self.idx_cursor = idx_start
            if verbose:
                print(f"Start at index {self.idx_cursor} / {len(data_source)}")
        else:
            self.idx_cursor = 0

        if idx_end is not None:
            assert idx_end <= len(data_source), \
                    f"End index '{idx_end}' is larger than the length of dataset '{len(data_source)}'"
            self.idx_end = idx_end
            if verbose:
                print(f"End at index {self.idx_end} / {len(data_source)}")
        else:
            self.idx_end = len(data_source)

    def __iter__(self):
        idx_start = self.idx_cursor
        for idx in range(idx_start, self.idx_end):
            self.idx_cursor = idx
            yield idx

    def __len__(self):
        return self.idx_end - self.idx_cursor

    def state_dict(self):
        return {
            "idx_cursor": self.idx_cursor,
            "idx_end": self.idx_end
        }

    def load_state_dict(self, state, verbose=True):
        self.idx_cursor = state["idx_cursor"] + 1
        self.idx_end = state["idx_end"]

        if verbose:
            print(f"Start at index {self.idx_cursor} / {len(self.data_source)}")
            print(f"End at index {self.idx_end} / {len(self.data_source)}")

    @staticmethod
    def from_state_dict(data_source, state, verbose=True):

        return PartialSampler(data_source, idx_start=state["idx_cursor"]+1, idx_end=state["idx_end"], verbose=verbose)


def get_imagenet_loader(imagenet_dir, batch_size, normalize=True, shuffle=False, sampler_state=None, verbose=True):
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    image_data = ImageFolder(imagenet_dir, transform=transforms.Compose(transform_list))

    if verbose:
        print(f"{len(image_data.classes)} classes / {len(image_data.samples)} samples")

    if sampler_state is None:
        sampler = PartialSampler(image_data, verbose=verbose)
    else:
        sampler = PartialSampler.from_state_dict(image_data, sampler_state, verbose=verbose)

    return DataLoader(image_data, batch_size=batch_size, shuffle=shuffle, sampler=sampler)


def save_images(filename, images_dict, nrows, ncols, figsize=None, cmap="jet"):
    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = ax.ravel()

    for idx, (name, image) in enumerate(images_dict.items()):
        axes[idx].set_title(name)
        axes[idx].imshow(image, cmap=cmap)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(filename)


def save_degradation_results(filename, results_dict, title=None):
    fig_num = len(results_dict)
    ncols = math.ceil(math.sqrt(fig_num))
    nrows = math.ceil(fig_num / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True, squeeze=False)
    if title is not None:
        fig.suptitle(title)

    for method_idx, (name, (morf, lerf, score)) in enumerate(results_dict.items()):
        x_axis = torch.linspace(0, 1, morf.shape[0]).numpy()
        morf, lerf = morf.cpu().numpy(), lerf.cpu().numpy()

        target_axes = axes[method_idx // ncols, method_idx % ncols]
        target_axes.set(title=name, xlabel="x̄ - degradation of x", ylabel="s(x̄) - normalized score")
        target_axes.plot(x_axis, lerf, label="LeRF")
        target_axes.plot(x_axis, morf, label="MoRF")
        target_axes.fill_between(x_axis, morf, lerf, facecolor="gray", alpha=0.5)
        target_axes.text(0.45, 0.48, f"{score:.3f}")
        target_axes.grid()
        target_axes.legend()

    for idx in range(len(results_dict), ncols * nrows):
        target_axes = axes[idx // ncols, idx % ncols]
        target_axes.axis("off")

    if filename == "inline":
        plt.show()
    else:
        fig.savefig(filename)


def save_heatmaps(filename, heatmaps_dict, vertical=False, cmap="BlWhRd", outlier_percent=0.02):
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

    if filename == "inline":
        fig.show()
    else:
        fig.savefig(filename)
