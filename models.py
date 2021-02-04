import torch
from torch import nn, load
from torch.nn import functional as F
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.models as torch_models
from torch.autograd import Variable
import torchvision.transforms as transforms
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, euler_angles_to_matrix

import numpy as np
from functools import reduce
import utils
import nn_utils

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, action_dim, encoding='tanh', device='cuda', softmax = True):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.action_dim = action_dim

        self.encoding = encoding
        self.device = device

        self.softmax = softmax

    def encode(self, x):
        encoded =  self.encoder(x)
        return encoded

    def decode(self, z):
        return self.decoder(z)

    def encode_pose(self, x):
        z = self.encode(x)
        return self.pose_activation(z[:, :self.action_dim])

    def pose_activation(self, z):
        if self.encoding == 'tanh':
            return torch.tanh(z)
        elif self.encoding == 'normalize':
            return nn_utils.normalize(z)
        elif self.encoding == 'non':
            return z
        elif self.encoding == 'euler':
            t = F.tanh(z)
            t = t * torch.Tensor([[np.pi, np.pi/2, np.pi]]).to(self.device)
            return t
        elif self.encoding == 'angle':
            return nn_utils.normalize(z)


    def forward(self, x):
        z = self.encode(x)

        pose = self.pose_activation(z[:, :self.action_dim])
        if self.encoding == 'normalize':
            random_euler = torch.randn(size = (z.size(0), 3)) * np.sqrt(0.1 * 2 * np.pi / 360)
            random_euler = random_euler.to(self.device)
            random_quaternion = matrix_to_quaternion(euler_angles_to_matrix(random_euler, 'XYZ'))
            pose = quaternion_multiply(pose, random_quaternion)
        if self.softmax:
            extra = F.softmax(z[:, self.action_dim :], dim=-1)
        else:
            extra = z[:, self.action_dim:]

        total = torch.cat((pose, extra), -1)
        return self.decode(total), z



class AE_MNIST(AE):
    def __init__(self, latent_dim, nc, action_dim, encoding, softmax=True):

        encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),          # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 7, 7
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View([-1, 256 * 5 * 5]),                 # B, 256
            nn.Linear(256 * 5 * 5, latent_dim),             # B, latent_dim*2
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 5 * 5),               # B, 256
            View((-1, 256, 5, 5)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 1),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, 1, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), # B,  32, 32, 32
            nn.Sigmoid()
            )
        super(AE_MNIST, self).__init__(encoder, decoder, latent_dim, action_dim, encoding, softmax=softmax)


class AE_CNN(AE):                           #64x64
    def __init__(self, latent_dim, nc, action_dim, encoding, softmax=True):

        encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View([-1, 256]),                 # B, 256
            nn.Linear(256, latent_dim),             # B, latent_dim*2
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),  # B, nc, 64, 64
            nn.Sigmoid()
            )
        super(AE_CNN, self).__init__(encoder, decoder, latent_dim, action_dim, encoding, softmax=softmax)


class BetaVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, beta):
        super(BetaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.use_vae = True
        self.beta = beta

    def encode(self, x):
        encoded =  self.encoder(x)
        return encoded[:, :self.latent_dim], encoded[:, self.latent_dim : ]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):                           #Monte Carlo (with one sample)
        mu, logvar = self.encode(x)
        if self.use_vae:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return self.decode(z), mu, logvar, z

    def traverse(self, sample, path):
        for i, latent  in enumerate(sample):
            img = self.decode(latent)
            save_image(img, path + '_' + str(i) + '.png', nrow = len(sample[0]))

    def recon_loss(self, recon_x, x):
        return F.binary_cross_entropy(recon_x, x, reduction = 'sum')

    def loss_function(self, recon_x, x, mu, logvar, beta):
        KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta_loss = beta * KL_loss
        return self.recon_loss(recon_x, x), KL_loss

    def full_loss(self, x):
        recon_x, mu, logvar, z = self.forward(x)
        return self.loss_function(recon_x, x, mu, logvar, self.beta)


class VAE_CNN(BetaVAE):                           #64x64
    def __init__(self, latent_dim, nc, beta):

        encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View([-1, 256]),                 # B, 256
            nn.Linear(256, 2*latent_dim),             # B, latent_dim*2
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),  # B, nc, 64, 64
            nn.Sigmoid()
            )
        super(VAE_CNN, self).__init__(encoder, decoder, latent_dim, beta)




if __name__ == "__main__":
    pass
