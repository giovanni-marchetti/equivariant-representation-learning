import torch
import numpy as np
import os

from datasets.equiv_dset import EquivDataset

import argparse
import pickle


def color_image(img1, color1):
    img1 = img1.repeat(3, 1, 1) # (3, 64, 64)
    img1 = img1 * color1.unsqueeze(-1).unsqueeze(-1)
    return img1


def generate(N=100000):

    dset = EquivDataset('data/sprites_data/', greyscale = True)
    train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=256, shuffle = True)

    color1 = torch.FloatTensor((255,99,71)) / 255
    color2 = torch.FloatTensor((100,149,237)) / 255
    color3 = torch.FloatTensor((50,205,50)) / 255

    final_data = []
    final_lbls = []
    final_classes = []

    for i in range(N):

        class1 = 1
        class2 = 2
        class3 = 3
        while (class1 != class2) or (class2 != class3):

            idx1 = np.random.randint(0, 100000)
            idx2 = np.random.randint(0, 100000)
            idx3 = np.random.randint(0, 100000)
            img1, img1_next, action1, class1 = dset.__getitem__(idx1)
            img2, img2_next, action2, class2 = dset.__getitem__(idx2)
            img3, img3_next, action3, class3 = dset.__getitem__(idx3)

        new = torch.zeros((3, 64, 64))

        img1 = color_image(img1, color1)
        img2 = color_image(img2, color2)
        img3 = color_image(img3, color3)
        img1_next = color_image(img1_next, color1)
        img2_next = color_image(img2_next, color2)
        img3_next = color_image(img3_next, color3)

        new = torch.clip((img1 + img2 + img3), 0, 1)
        new_next = torch.clip((img1_next + img2_next + img3_next), 0, 1)

        final_data.append([new.detach().numpy(), new_next.detach().numpy()])

        new = torch.zeros((6,))
        new[:2] = action1
        new[2:4] = action2
        new[4:6] = action3

        final_lbls.append([new.detach().numpy()])
        final_classes.append([class1.detach().numpy()[0]])

    if not os.path.exists('data/multisprites_data'):
        os.mkdir('data/multisprites_data')

    np.save('data/multisprites_data/equiv_data.npy', np.array(final_data))
    np.save('data/multisprites_data/equiv_lbls.npy', np.array(final_lbls))
    np.save('data/multisprites_data/equiv_classes.npy', np.array(final_classes))

class EquivMultiSpritesDataset(EquivDataset):
    def __init__(self):

        path = 'data/multisprites_data'
        if not os.path.exists(path):
            generate()

        super().__init__('data/multisprites_data/')


if __name__ == '__main__':
    generate(10)