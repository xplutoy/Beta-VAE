import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv
from PIL import Image


def _file_paths(dir, ext):
    paths = []
    for entry in os.scandir(dir):
        if entry.is_file() and entry.path.endswith(ext):
            paths.append(entry.path)
        elif entry.is_dir():
            paths.extend(_file_paths(entry.path, ext))
    return paths


class single_tensor_dataset(data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


class single_class_image_folder(data.Dataset):
    def __init__(self, root, ext='.png', transform=None):
        self.img_paths = _file_paths(root, ext)
        self.L = len(self.img_paths)
        self.transform = transform

    def __len__(self):
        return self.L

    def __getitem__(self, item):
        f_path = self.img_paths[random.choice(range(self.L))]
        img = Image.open(f_path)
        if self.transform is not None:
            img = self.transform(img)
        return img


_transformer = tv.transforms.Compose([
    tv.transforms.Resize([64, 64]),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

chairs_3d_iter = torch.utils.data.DataLoader(
    dataset=single_class_image_folder('../../Datasets/rendered_chairs/', transform=_transformer),
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=4,
)

# load dsprites datasets
_dsprites_array = np.load('../../Datasets/dsprites_ndarray.npz', encoding='bytes')
_dsprites_tensor = torch.from_numpy(_dsprites_array['imgs']).unsqueeze(1).float()
dsprites_iter = torch.utils.data.DataLoader(
    dataset=single_tensor_dataset(_dsprites_tensor),
    batch_size=64,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=4
)
