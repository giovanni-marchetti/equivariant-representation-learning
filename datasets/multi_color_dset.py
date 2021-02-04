import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np

import torch
import torch.utils.data
from torch import nn, optim, save, load
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt

from datasets.equiv_dset import EquivDataset

import argparse
import pickle

from tqdm import tqdm

def color_image(img1, color1):
    img1 = img1.repeat(3, 1, 1) # (3, 64, 64)
    img1 = img1 * color1.unsqueeze(-1).unsqueeze(-1)
    return img1

def get_color(class_idx):
    pass

def generate_dataset(N=100000, num_classes = 250):

    assert N > num_classes, "N > num_classes"

    base_path = "data/multi_color_data/"
    save_file = base_path + 'data.pkl'

    cmap = plt.get_cmap('viridis', num_classes)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    dset = EquivDataset('data/sprites_data/', greyscale = True)
    pairs = []

    img_pairs = []
    actions = []
    classes = []

    N_per_class = N // num_classes
    for class_idx in tqdm(range(num_classes)):

        for i in range(N_per_class):

            while True:
                color = torch.FloatTensor(cmap(class_idx)[:-1]) * 255
                color = color.int()

                idx1 = np.random.randint(0, 100000)
                img1, img1_next, action1, class1 = dset.__getitem__(idx1)
                if int(class1) != 1:
                    continue

                img1 = color_image(img1, color)
                img1_next = color_image(img1_next, color)

                pair = [img1, img1_next, action1, torch.LongTensor([class_idx])]
                pair = [x.cpu().data.numpy() for x in pair]
                pair[0] = pair[0].astype(np.uint8)
                pair[1] = pair[1].astype(np.uint8)

                pairs.append(pair)
                break

    with open(save_file, 'wb') as f:
        pickle.dump(pairs, f)


class EquivMultiColorDataset(torch.utils.data.Dataset):
    def __init__(self):
        path = 'data/multi_color_data/data.pkl'
        if not os.path.exists(path):
            generate_dataset(100000, 250)

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        img1, img2, action, _class = self.data[index]
        return torch.from_numpy(img1).float() / 255., torch.from_numpy(img2).float() / 255., torch.from_numpy(action).float(), torch.Tensor([_class]).long().squeeze(-1)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    generate_dataset(100000, 250)
