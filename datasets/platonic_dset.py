import torch
import numpy as np
import pickle


def PlatonicMerged(N, big=True):
    pyra = PlatonicDataset('tetra', N=N, big=big)
    octa = PlatonicDataset('octa', N=N, big=big)
    cube = PlatonicDataset('cube', N=N, big=big)
    return torch.utils.data.ConcatDataset([cube, octa, pyra])

class PlatonicDataset(torch.utils.data.Dataset):
    def __init__(self, platonic, N, big=True, width=64):

        self.classes = {'cube':0, 'tetra':1, 'octa':2}
        self.platonic = platonic

        postfix = '-big' if big else ''
        path = f'data/platonic/{platonic}_uniform_black-{width}{postfix}.pkl'
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        img1, img2, action = self.data[idx]

        img1 = torch.from_numpy(img1).float() / 255.
        img2 = torch.from_numpy(img2).float() / 255.
        action = torch.from_numpy(action).float()

        return img1, img2, action, torch.Tensor([self.classes[self.platonic]]).long()

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    pass
