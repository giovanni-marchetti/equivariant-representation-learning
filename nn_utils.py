import torch
import torch.nn.functional as F
import numpy as np
import os
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, euler_angles_to_matrix
import matplotlib.pyplot as plt

def entropy_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction = 'mean')

def normalize(z, EPS=1e-8):
    norm = torch.norm(z, p=2, dim=-1, keepdim=True) + EPS
    return z.div(norm)

def quaternion_distance(q1, q2):
    dist1 = torch.norm(q1-q2, p=2, dim = -1)
    dist2 = torch.norm(q1 + q2, p=2, dim=-1)
    dist = torch.min(dist1, dist2).mean()

    return dist

def equivariance_loss(model, z, action, action_dim, img_next, action_type, method='quaternion', device='cuda'):
    true_pose = model.encode_pose(img_next)

    if action_type == 'translate':
        z_out = torch.tanh(z[:, :action_dim])
        pose = z_out + action
        equiv_loss = F.mse_loss(pose, true_pose)
    elif action_type == 'rotate':
        if method == 'naive':
            """
            Encode naively to R^3 and multiply by matrix
            """
            pose = (z[:, None, :action_dim] @ action).squeeze(1)
            equiv_loss = (1 - F.cosine_similarity(pose, true_pose, dim=-1)).mean()
        elif method == 'quaternion':
            quaternion_action = matrix_to_quaternion(action)
            z_quaternion = normalize(z[:, :action_dim])
            pose = quaternion_multiply(z_quaternion, quaternion_action)
            equiv_loss = quaternion_distance(pose, true_pose)

    elif action_type == 'rotate_onedim':
        cos = torch.cos(action)
        sin = torch.sin(action)

        rot_matrix = torch.zeros((action.size(0), 2, 2)).to(device)
        rot_matrix[:, 0, 0] = cos[:, 0]
        rot_matrix[:, 0, 1] = -sin[:, 0]
        rot_matrix[:, 1, 0] = sin[:, 0]
        rot_matrix[:, 1, 1] = cos[:, 0]

        z_out = model.pose_activation(z[:, :action_dim])

        pose = rot_matrix @ z_out.unsqueeze(-1)
        equiv_loss = F.mse_loss(pose.squeeze(-1), true_pose)

    return equiv_loss

def cross_entropy(z, z_next, action_dim):
    return - (F.log_softmax(z[:, action_dim:], dim = -1)* F.softmax(z_next[:, action_dim:], dim=-1) ).mean()   #CrossEntropy

def centroid_entropy(z, EPS=1e-7):
    centroid = z.mean(0)
    return  (centroid * torch.log(centroid + EPS)).mean()
