import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def prepare_pair(image_path, mask_path):
    with Image.open(image_path) as source:
        image = source.resize((256, 256), Image.Resampling.BILINEAR)

    with Image.open(mask_path) as source:
        mask = source.resize((256, 256), Image.Resampling.NEAREST)

    image = np.array(image, dtype=np.float32) / 255
    image = image.transpose(2, 0, 1)

    labels = np.array(mask, dtype=np.int64)
    mask = labels - 1
    mask[labels == 0] = 255

    return image, mask


class LandCoverDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs.iloc[index]
        image, mask = prepare_pair(pair["image"], pair["mask"])
        return torch.from_numpy(image), torch.from_numpy(mask)
