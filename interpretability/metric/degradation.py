import torch
from torch.nn import Fold, Unfold
import torch.nn.functional as F

from tqdm import trange, tqdm

import os


class Degradation:
    @staticmethod
    def perturb(model, inputs_tile, covers_tile, targets, tile_rank, fold, stride, mode="MoRF", use_probs=True, progbar=True):
        if mode == "MoRF":
            get_target_idx = lambda idx: idx
        elif mode == "LeRF":
            get_target_idx = lambda idx: -(idx + 1)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported mode: ['MoRF', 'LeRF']")

        index0 = torch.arange(inputs_tile.shape[0], dtype=torch.long)
        history = torch.zeros((covers_tile.shape[2]), device=inputs_tile.device)

        prev_degraded_scores = None
        skipped_cnt = 0

        for perturb_cnt in trange(covers_tile.shape[2], desc=mode, disable=not progbar, ncols=70):
            index1 = tile_rank[:, get_target_idx(perturb_cnt)]
            inputs_tile[index0, :, index1] = covers_tile[index0, :, index1]

            if perturb_cnt % stride == 0 or perturb_cnt == covers_tile.shape[2] - 1:
                perturbed_inputs = fold(inputs_tile.view(inputs_tile.shape[0], -1, inputs_tile.shape[3]))

                logits = model(perturbed_inputs)
                probs = F.softmax(logits, dim=1)

                scores = probs if use_probs else logits
                degraded_scores = scores.gather(dim=1, index=targets.view(targets.shape[0], 1)).flatten().detach()
                history[perturb_cnt] += degraded_scores.sum()

                for idx in range(skipped_cnt):
                    current_ratio = (idx + 1) / (skipped_cnt + 1)
                    prev_ratio = 1 - current_ratio
                    history[perturb_cnt - skipped_cnt + idx] += (prev_degraded_scores * prev_ratio + degraded_scores * current_ratio).sum()

                prev_degraded_scores = degraded_scores
                skipped_cnt = 0
            else:
                skipped_cnt += 1

        return history

    @staticmethod
    def baseline(inputs_tile, baseline):
        if baseline == "tile_mean":
            return inputs_tile.mean(dim=3, keepdim=True)
        elif baseline == "mean":
            return inputs_tile.mean(dim=3, keepdim=True) \
                              .mean(dim=2, keepdim=True) \
                              .repeat(1, 1, inputs_tile.shape[2], 1)
        elif baseline == "zero":
            return torch.zeros((*inputs_tile.shape[:3], 1), device=inputs_tile.device)
        elif baseline == "uniform":
            return torch.rand_like(inputs_tile)
        else:
            raise NotImplementedError

    @staticmethod
    def evaluate(attribution, data_loader, tile_size=14, baseline="mean", perturb_stride=5,
                 initial_state=None, desc="Eval", progbar=True, checkpoint_prefix=None):
        child_progbar = progbar

        model, attribute = attribution.forward_func, attribution.attribute
        device = next(model.parameters()).device

        prev_idx_cur = 0
        if initial_state is None:
            morf, lerf = None, None
        else:
            morf, lerf = initial_state["morf"], initial_state["lerf"]
            if morf is not None and lerf is not None:
                morf, lerf = morf.to(device=device), lerf.to(device=device)
            if "sampler_state" in initial_state:
                sampler_state = initial_state["sampler_state"]
                if sampler_state is not None:
                    prev_idx_cur = sampler_state["idx_cursor"]

        for inputs, targets in tqdm(data_loader, desc=f"{desc:15s}", disable=not progbar, ncols=70):
            assert inputs.shape[2] % tile_size == 0 and inputs.shape[3] % tile_size == 0, "Size mismatch"

            inputs, targets = inputs.to(device=device), targets.to(device=device)
            inputs.requires_grad = True
            heatmaps = F.interpolate(attribute(inputs, target=targets), size=inputs.shape[2:4], mode="bilinear", align_corners=False)

            with torch.no_grad():
                tile_num = int(inputs.shape[2] / tile_size)

                unfold = Unfold(kernel_size=tile_num, dilation=tile_size)
                fold = Fold(output_size=inputs.shape[2:4], kernel_size=tile_num, dilation=tile_size)

                heatmaps_tile = unfold(heatmaps.sum(dim=1, keepdim=True))
                tile_rank = heatmaps_tile.sum(dim=2).argsort(dim=1, descending=True)
                inputs_tile = unfold(inputs).view(inputs.shape[0], inputs.shape[1], tile_num * tile_num, -1)
                covers_tile = Degradation.baseline(inputs_tile, baseline)

                morf_b = Degradation.perturb(model, inputs_tile.clone(), covers_tile, targets,
                                             tile_rank, fold, stride=perturb_stride, mode="MoRF", progbar=child_progbar)
                morf = morf_b if morf is None else morf + morf_b

                lerf_b = Degradation.perturb(model, inputs_tile, covers_tile, targets,
                                             tile_rank, fold, stride=perturb_stride, mode="LeRF", progbar=child_progbar)
                lerf = lerf_b if lerf is None else lerf + lerf_b

            # Checkpoint
            idx_cur = data_loader.sampler.state_dict()["idx_cursor"]

            torch.save({
                "morf": morf.cpu(),
                "lerf": lerf.cpu(),
                "sampler_state": data_loader.sampler.state_dict()
            }, f"{checkpoint_prefix}-{idx_cur}.pt")

            if os.path.exists(f"{checkpoint_prefix}-{prev_idx_cur}.pt"):
                os.remove(f"{checkpoint_prefix}-{prev_idx_cur}.pt")

            prev_idx_cur = idx_cur

        norm_morf = (morf - morf[-1]) / (morf[0] - morf[-1])
        norm_lerf = (lerf - lerf[-1]) / (lerf[0] - lerf[-1])

        return norm_morf.cpu(), norm_lerf.cpu()

    @staticmethod
    def score(norm_morf, norm_lerf):
        assert norm_morf.shape == norm_lerf.shape, "History length doesn't match"

        return (norm_lerf[1:-1] - norm_morf[1:-1]).mean().item()
