import torch
import torch.utils.data
from torch import nn, optim, save, load
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle

from models import AE_CNN, VAE_CNN, AE_MNIST
from resnet import ResnetV2

from nn_utils import entropy_loss, equivariance_loss, centroid_entropy, cross_entropy
from utils import plot_triangle, plot_points, make_dir

# Import datasets
from datasets.multi_color_dset import EquivMultiColorDataset
from datasets.platonic_dset import PlatonicMerged
from datasets.equiv_dset import EquivDataset
from datasets.rotated_mnist import RotatedMNIST
from datasets.multi_sprites import EquivMultiSpritesDataset

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help="Set seed for training")

# Training details
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--reg-lambda', default=10.0, type=float, help="Entropy regularization")
parser.add_argument('--ce-lambda', default=1.0, type=float, help="Class regularization")
parser.add_argument('--pose-lambda', default=10, type=float, help="Pose regularization")
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--val-interval', type=int, default=10)
parser.add_argument('--save-interval', default=10, type=int, help="Epoch to save model")
parser.add_argument('--no-softmax', action='store_true', default = False)

# Dataset
parser.add_argument('--dataset', default='sprites', type=str, help="Dataset")
parser.add_argument('--batch-size', type=int, default=64, help="Batch size")

# Model arguments
parser.add_argument('--model', default='cnn', type=str, help="Model to use")

parser.add_argument('--action-dim', default=2, type=int, help="Dimension of the group")
parser.add_argument('--extra-dim', default=3, type=int, help="Number of classes")
parser.add_argument('--model-name', required=True, type=str, help="Name of model")

# Rotation specific arguments
parser.add_argument('--encoding', type=str, default='normalize', help="Activation on z, tanh, normalize or non")
parser.add_argument('--method', type=str, default='quaternion', help="What loss to use for rotations")

# Beta VAE
parser.add_argument('--beta-vae', action='store_true', default=False, help="Use beta vae")
parser.add_argument('--beta', type=float, help="Beta for a beta vae")
parser.add_argument('--latent_dim', type=int, help="Latent dim for VAE")

# Optimization
parser.add_argument('--lr-scheduler', action='store_true', default=False, help="Use a lr scheduler")
parser.add_argument('--lr', default=1e-3, type=float)

args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save paths
MODEL_PATH = os.path.join('checkpoints', args.model_name)

figures_dir = os.path.join(MODEL_PATH, 'figures')
tf_log_dir = os.path.join(MODEL_PATH, 'summary')
model_file = os.path.join(MODEL_PATH, 'model.pt')
meta_file = os.path.join(MODEL_PATH, 'metadata.pkl')
log_file = os.path.join(MODEL_PATH, 'log.txt')

make_dir(MODEL_PATH)
make_dir(figures_dir)
make_dir(tf_log_dir)

pickle.dump({'args': args}, open(meta_file, 'wb'))

# Set dimensions
action_dim = args.action_dim
extra_dim = args.extra_dim           #Also number of classes
latent_dim = action_dim + extra_dim

if args.model == 'vae':
    latent_dim = args.latent_dim

# Set group action
if args.dataset == 'sprites' or args.dataset == 'shapes' or args.dataset == 'multi-color' or args.dataset == 'multi-sprites':
    ACTION_TYPE = 'translate'
    encoding = 'tanh'
if args.dataset == 'platonics':
    ACTION_TYPE = 'rotate'
    encoding = args.encoding
if args.dataset == 'mnist':
    ACTION_TYPE = 'rotate_onedim'
    encoding = 'angle'

# Allocate dataset
if args.dataset == 'color-shift':
    dset = EquivDataset('data/colorshift_data/')
if args.dataset == 'sprites':
    dset = EquivDataset('data/sprites_data/', greyscale = True)
elif args.dataset == 'multi-sprites':
    dset = EquivMultiSpritesDataset()
elif args.dataset == 'multi-color':
    dset = EquivMultiColorDataset()
elif args.dataset == 'platonics':
    dset = PlatonicMerged(args.n, big=True)
elif args.dataset == 'mnist':
    dset = RotatedMNIST(10000)
else:
    print("Invalid dataset")


train_data, valid_data = torch.utils.data.random_split(dset, [len(dset) - int(len(dset)/10), int(len(dset)/10)])
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=args.batch_size, shuffle = True)

# Sample data
img, _, _, _ = next(iter(train_loader))
img_shape = img.shape[1:]

use_softmax = not args.no_softmax

# Create Model
if args.model == 'cnn':
    model = AE_CNN(latent_dim, img_shape[0], args.action_dim, encoding, softmax = use_softmax).to(device)
if args.model == 'mnist-cnn':
    model = AE_MNIST(latent_dim, img_shape[0], args.action_dim, encoding, softmax = use_softmax).to(device)
if args.model == 'resnet':
    model = ResnetV2(latent_dim, img_shape[0], args.action_dim, encoding).to(device)
if args.beta_vae:
    model = VAE_CNN(latent_dim, img_shape[0], args.beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
writer = SummaryWriter(log_dir = tf_log_dir)

if args.lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

def train(epoch, data_loader, mode='train'):

    mu_loss = 0
    total_indices = []

    for batch_idx, (img, img_next, action, classes) in enumerate(data_loader):

        if mode == 'train':
            optimizer.zero_grad()
            model.train()
        elif mode == 'val':
            model.eval()

        img = img.to(device)
        img_next = img_next.to(device)
        action = action.to(device)
        if args.dataset == 'multi-sprites':
            action = action.squeeze(1)

        if not args.beta_vae:
            recon, z = model(img)
            recon_next, z_next = model(img_next)

            recon_loss = entropy_loss(recon, img)
            recon_loss_next = entropy_loss(recon_next, img_next)

            total_recon_loss = 0.5 * (recon_loss + recon_loss_next)

            equiv_loss = equivariance_loss(model, z, action, action_dim, img_next, action_type=ACTION_TYPE, device=device, method=args.method)

            if use_softmax:
                ce_loss = cross_entropy(z, z_next, action_dim)
                entropy_reg = 0.5 * (centroid_entropy(F.softmax(z[:, action_dim:], dim=-1)) + centroid_entropy(F.softmax(z_next[:, action_dim:], dim=-1)))
            else:
                ce_loss = torch.sum((z[:, action_dim : ] - z_next[:, action_dim:])**2, -1).mean(0)

                # Hinge-loss
                z2 = z_next[:, action_dim : ]
                z2_rand = z2[torch.randperm(len(z2))]
                distance = torch.norm(z2 - z2_rand, p=2, dim=-1)

                entropy_reg = torch.max(torch.zeros_like(distance).to(device), torch.ones_like(distance).to(device) - distance).mean()

            loss = total_recon_loss + args.pose_lambda * equiv_loss + args.ce_lambda * ce_loss + args.reg_lambda * entropy_reg

        elif args.beta_vae:
            recon, mu, logvar, z = model(img)
            recon_loss, kl_div = model.loss_function(recon, img, mu, logvar, args.beta)
            loss = recon_loss + model.beta * kl_div

        if mode == 'train':
            loss.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0 and mode == 'train':
            if not args.beta_vae:
                print(f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)} \
                        Recon Loss: {total_recon_loss.item():.3} Equiv: {equiv_loss.item():.3} Neg entropy {entropy_reg.item():.3}")
            elif args.beta_vae:
                print(f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)} \
                        Recon Loss: {recon_loss.item():.3} KL-DIV: {kl_div.item():.3}")

        if mode == 'val':
            _, indices = F.softmax(z[:, action_dim:], dim =-1).max(dim = -1)
            if use_softmax:
                points = F.softmax(z[:, action_dim:], dim = -1).cpu().detach().numpy()
            else:
                points = z[:, action_dim:].cpu().detach().numpy()

            colors = classes.detach().numpy()[:, 0]

            if batch_idx == 0:
                total_points = points
                total_colors = colors
            else:
                total_points = np.append(total_points, points, 0)
                total_colors = np.append(total_colors, colors, 0)

            total_indices = np.append(total_indices, indices.cpu().data.numpy(), 0)


    if mode == 'val':
        print(f"{mode.upper()} Epoch: {epoch}, Loss: {(mu_loss / len(data_loader)):.3}")

        # Plot reconstruction
        save_image(recon[:16], f'{figures_dir}/recon_{str(epoch)}.png')

        unique, counts = np.unique(total_indices, return_counts=True)
        print("Classes", list(unique))
        print("Counts", list(counts))

    if (epoch < 10) or (epoch % args.save_interval) == 0:
        save(model, model_file)

    if args.lr_scheduler and mode == 'val':
        scheduler.step(loss)

    if not args.beta_vae:
        writer.add_scalar(f"Loss_equiv/{mode}", equiv_loss, epoch)
        writer.add_scalar(f"Loss_recon/{mode}", total_recon_loss, epoch)
    else:
        writer.add_scalar(f"Loss_kl/{mode}", kl_div, epoch)
        writer.add_scalar(f"Loss_recon/{mode}", recon_loss, epoch)



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, 'train')
        train(epoch, val_loader, 'val')
