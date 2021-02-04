import torch
import torchvision
import random
from PIL import Image
import numpy as np
import pickle
import os
from tqdm import tqdm


def generate_rotated_mnist(n = 10000):
    base = 'data/rotated_mnist'
    if not os.path.exists(base):
        os.makedirs(base)
    save_file = base + f'/data_diff_{n}.pkl'

    dataset = torchvision.datasets.MNIST('data', train=True, download=True)
    pairs = []

    data_labels = list(range(10))

    for label in data_labels:
        dataset = torchvision.datasets.MNIST('data', train=True, download=True)

        idx = dataset.targets == label
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

        num_examples = len(dataset)

        i = 0
        with tqdm(total=n) as pbar:
            while True:
                idx1 = random.randint(0, num_examples-1)
                idx2 = random.randint(0, num_examples-1)

                x1, _ = dataset.__getitem__(idx1)
                x2, _ = dataset.__getitem__(idx2)

                rotation1 = random.random() * 360
                rotation2 = random.random() * 360
                x1_rotate = np.array(x1.rotate(rotation1))
                x2_rotate = np.array(x2.rotate(rotation2))

                delta_rotation = np.mod((rotation1 - rotation2) / 360 * 2 * np.pi, 2 * np.pi)
                pair = [x1_rotate[None, :, :], x2_rotate[None, :, :], delta_rotation, label]
                pairs.append(pair)

                pbar.update(1)
                i += 1
                if i > n-1:
                    break

        with open(save_file, 'wb') as f:
            pickle.dump(pairs, f)

        

class RotatedMNIST(torch.utils.data.Dataset):
    def __init__(self, n=10000):
        if not os.path.exists('data'):
            os.mkdir('data')
        save_file = f'data/rotated_mnist/data_diff_{n}.pkl'

        if not os.path.exists(save_file):
            print("Generating data...")
            generate_rotated_mnist(n)

        with open(save_file, 'rb') as f:
            self.data = pickle.load(f)
       
    def __getitem__(self, idx):
        x1, x2, rotation, label = self.data[idx]

        x1 = torch.from_numpy(x1) / 255.
        x2 = torch.from_numpy(x2) / 255.
        rotation = torch.Tensor([rotation]).float()
        label = torch.Tensor([label]).long()
        return x1, x2, rotation, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    generate_rotated_mnist(10000)