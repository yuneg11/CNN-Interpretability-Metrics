from torch.utils.data import Sampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


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
