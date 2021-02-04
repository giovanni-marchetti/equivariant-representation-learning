import torch
import torch.nn.functional as F
import numpy as np
import os
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, euler_angles_to_matrix
import matplotlib.pyplot as plt

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_triangle(points, colors, path):
    points2d = np.zeros([len(points), 2])
    points2d[:, 0] = (points[:, 0] - points[:, 1])*0.577350
    points2d[:, 1] = points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.axis('off')
    plt.plot([-0.577350, 0.577350], [0,0], [0.577350, 0], [0,1], [0, -0.577350], [1,0], color = 'lightgray', zorder = 0)           #draw triangle
    ax.scatter(points2d[:,0], points2d[:,1], c = colors, zorder = 5)#cmap='Greens')
    plt.savefig(path)

def plot_points(points, colors, path):
    """
    points : 2d

    """
    cmap = plt.get_cmap('viridis', 250)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:,0], points[:, 1], c=colors, cmap = cmap)
    plt.savefig(path)