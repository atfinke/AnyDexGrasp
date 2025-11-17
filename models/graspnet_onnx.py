"""
ONNX-compatible GraspNet model
Replaces CUDA operators with PyTorch/ONNX-compatible operations
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'onnx_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from onnx_ops.pointnet2_onnx import (
    furthest_point_sample,
    cylinder_query_and_group,
    group_points
)
from pointnet2_backbone_onnx import PointNet2Backbone
from pt_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class SharedMLP(nn.Module):
    """Shared MLP for point cloud processing."""
    def __init__(self, mlps, bn=True):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList() if bn else None

        for i in range(len(mlps) - 1):
            self.mlps.append(nn.Conv2d(mlps[i], mlps[i+1], 1))
            if bn:
                self.bns.append(nn.BatchNorm2d(mlps[i+1]))

    def forward(self, x):
        """
        Args:
            x: (B, C, N, nsample)
        Returns:
            x: (B, mlps[-1], N, nsample)
        """
        for i, conv in enumerate(self.mlps):
            x = conv(x)
            if self.bns is not None:
                x = self.bns[i](x)
            x = F.relu(x, inplace=True)
        return x


class PointCylinderGroupONNX(nn.Module):
    """
    Point Cylinder Group module using ONNX-compatible operations.
    Replaces CUDA-based CylinderQueryAndGroup.
    """
    def __init__(self, nsample, mlps, cylinder_radius=0.05, hmin=-0.02, hmax=0.04, bn=True):
        super().__init__()
        self.cylinder_radius = cylinder_radius
        self.nsample = nsample
        self.hmin = hmin
        self.hmax = hmax

        mlps[0] += 3  # Add xyz coordinates
        self.mlps = SharedMLP(mlps, bn=bn)

    def forward(self, xyz, new_xyz, view_rot, features):
        """
        Args:
            xyz: (B, N, 3) input points
            new_xyz: (B, M, 3) cylinder centers
            view_rot: (B, M, 3, 3) rotation matrices
            features: (B, C, N) input features

        Returns:
            new_features: (B, mlps[-1], M) output features
        """
        B, M, _, _ = view_rot.size()

        # Perform cylinder query and grouping
        grouped_features, grouped_xyz = cylinder_query_and_group(
            xyz, new_xyz, view_rot, features,
            self.cylinder_radius, self.hmin, self.hmax, self.nsample
        )
        # grouped_features: (B, C, M, nsample)
        # grouped_xyz: (B, M, nsample, 3)

        # Normalize grouped xyz
        grouped_xyz_norm = grouped_xyz / self.cylinder_radius  # normalize

        # Rotate grouped xyz to cylinder frame
        # grouped_xyz_norm: (B, M, nsample, 3)
        # view_rot: (B, M, 3, 3)
        # Result: (B, M, nsample, 3)
        grouped_xyz_rotated = torch.einsum('bmni,bmij->bmnj',
                                           grouped_xyz_norm,
                                           view_rot.transpose(-2, -1))

        # Concatenate xyz with features
        # grouped_xyz_rotated: (B, M, nsample, 3) -> (B, 3, M, nsample)
        grouped_xyz_rotated = grouped_xyz_rotated.permute(0, 3, 1, 2)

        # Concatenate
        grouped_features = torch.cat([grouped_xyz_rotated, grouped_features], dim=1)

        # Apply MLP
        new_features = self.mlps(grouped_features)  # (B, mlps[-1], M, nsample)

        # Max pooling
        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        )  # (B, mlps[-1], M, 1)
        new_features = new_features.squeeze(3)  # (B, mlps[-1], M)

        return new_features


class ViewEstimatorONNX(nn.Module):
    """
    View Estimator using ONNX-compatible operations.
    """
    def __init__(self, in_channels=512, num_samples=1024, num_view=300,
                 sampling='fps', heatmap_th=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.num_view = num_view
        self.sampling = sampling
        self.heatmap_th = heatmap_th

        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        self.conv3 = nn.Conv1d(in_channels, num_view, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) input points
            features: (B, C, N) input features

        Returns:
            view_heatmap: (B, M, num_view) view predictions
            seed_xyz: (B, M, 3) sampled seed points
            seed_features: (B, C, M) features at seed points
        """
        B, N, _ = xyz.shape

        # Sample seed points using FPS
        seed_inds = furthest_point_sample(xyz, self.num_samples)  # (B, M)

        # Gather seed xyz
        seed_inds_expanded = seed_inds.unsqueeze(-1).expand(-1, -1, 3)  # (B, M, 3)
        seed_xyz = torch.gather(xyz, 1, seed_inds_expanded)  # (B, M, 3)

        # Gather seed features
        seed_features = group_points(features, seed_inds)  # (B, C, M)

        # Forward pass
        out = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        seed_features = out + seed_features  # residual
        out = self.conv3(out)  # (B, num_view, M)
        out = out.transpose(1, 2)  # (B, M, num_view)

        return out, seed_xyz, seed_features

    def sample_grasp_points_fps(self, points):
        """Simplified FPS-based sampling for ONNX export."""
        if self.num_samples < 0:
            return torch.arange(points.size(0), device=points.device)

        fps_inds = furthest_point_sample(points.unsqueeze(0), self.num_samples)
        return fps_inds.squeeze(0).long()


class GraspGeneratorONNX(nn.Module):
    """
    Grasp Generator using ONNX-compatible operations.
    """
    def __init__(self, feature_dim, num_sample=16, cylinder_radius=0.05,
                 hmin=-0.02, hmax=0.04, num_angle=48, num_depth=5, half_views=False):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.in_dim = feature_dim

        if half_views:
            mlps = [self.in_dim, 256, 128, 128]
        else:
            mlps = [self.in_dim, 512, 512, 512]

        self.pcg = PointCylinderGroupONNX(num_sample, mlps, cylinder_radius,
                                          hmin, hmax, bn=True)

        # Output: (score + width) * num_angle * num_depth
        self.conv1 = nn.Conv1d(mlps[-1], mlps[-1], 1)
        self.conv2 = nn.Conv1d(mlps[-1], mlps[-1], 1)
        self.conv3 = nn.Conv1d(mlps[-1], num_angle * num_depth * 2, 1)
        self.bn1 = nn.BatchNorm1d(mlps[-1])
        self.bn2 = nn.BatchNorm1d(mlps[-1])

    def forward(self, xyz, new_xyz, view_rot, features):
        """
        Args:
            xyz: (B, N, 3) input points (seed points)
            new_xyz: (B, M, 3) grasp centers
            view_rot: (B, M, 3, 3) approach directions
            features: (B, C, N) seed features

        Returns:
            grasp_preds: (B, M, num_angle*num_depth*2) grasp predictions
            view_features: (B, mlps[-1], M) intermediate features
        """
        B, num_seed, _ = new_xyz.size()

        view_features = self.pcg(xyz, new_xyz, view_rot, features)
        before_generator = view_features

        view_features = F.relu(self.bn1(self.conv1(view_features)), inplace=True)
        view_features = F.relu(self.bn2(self.conv2(view_features)), inplace=True)
        grasp_preds = self.conv3(view_features)  # (B, num_angle*num_depth*2, M)
        grasp_preds = grasp_preds.transpose(1, 2).contiguous()  # (B, M, num_angle*num_depth*2)

        return grasp_preds, view_features, before_generator


class GraspNetONNX(nn.Module):
    """
    ONNX-compatible GraspNet model.
    Replaces MinkowskiEngine and CUDA operators with PyTorch/ONNX-compatible operations.
    """
    def __init__(self, in_channels=3, num_seed=1024, num_view=300,
                 num_angle=48, num_depth=5, sampling='fps',
                 is_training=False, half_views=False):
        super().__init__()

        if half_views:
            num_view //= 2

        self.num_seed = num_seed
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.half_views = half_views

        # Backbone: PointNet++ instead of MinkowskiEngine
        if half_views:
            self.heatmap_generator = PointNet2Backbone(
                in_channels=in_channels, out_channels=3,
                feature_dim=128, half_views=True
            )
            feature_dim = 128
        else:
            self.heatmap_generator = PointNet2Backbone(
                in_channels=in_channels, out_channels=3,
                feature_dim=512, half_views=False
            )
            feature_dim = 512

        # View estimator
        self.view_estimator = ViewEstimatorONNX(
            in_channels=feature_dim, num_samples=num_seed,
            num_view=num_view, sampling=sampling
        )

        # Grasp generator
        self.grasp_generator = GraspGeneratorONNX(
            feature_dim, num_sample=16, cylinder_radius=0.05,
            hmin=-0.02, hmax=0.04, num_angle=num_angle,
            num_depth=num_depth, half_views=half_views
        )

    def forward(self, xyz, features=None):
        """
        Args:
            xyz: (B, N, 3) input point cloud
            features: (B, C, N) input features (optional)

        Returns:
            Dictionary with predictions:
                - objectness: (B, N, 2) object vs background
                - heatmap: (B, N) grasp heatmap
                - view_heatmap: (B, num_seed, num_view) view scores
                - seed_xyz: (B, num_seed, 3) sampled grasp centers
                - grasp_scores: (B, num_seed, num_angle, num_depth) grasp scores
                - grasp_widths: (B, num_seed, num_angle, num_depth) gripper widths
        """
        B, N, _ = xyz.shape

        # Stage 1: Heatmap generation
        heatmap_out, point_features = self.heatmap_generator(
            xyz, features, return_features=True
        )
        # heatmap_out: (B, 3, N) - [objectness_0, objectness_1, heatmap]
        # point_features: (B, feature_dim, N)

        objectness_pred = heatmap_out[:, 0:2, :].transpose(1, 2)  # (B, N, 2)
        heatmap_pred = torch.sigmoid(heatmap_out[:, 2, :])  # (B, N)

        # Stage 2: View estimation
        view_heatmap, seed_xyz, seed_features = self.view_estimator(
            xyz, point_features
        )
        # view_heatmap: (B, num_seed, num_view)
        # seed_xyz: (B, num_seed, 3)
        # seed_features: (B, feature_dim, num_seed)

        view_heatmap_pred = torch.sigmoid(view_heatmap)

        # Select best view for each seed point
        view_scores, view_inds = torch.max(view_heatmap_pred, dim=2)  # (B, num_seed)

        # Get view directions from template
        if self.half_views:
            template_views = generate_grasp_views(2 * self.num_view)[:self.num_view]
        else:
            template_views = generate_grasp_views(self.num_view)

        template_views = template_views.to(xyz.device)  # (num_view, 3)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(
            B, self.num_seed, -1, -1
        )  # (B, num_seed, num_view, 3)

        # Gather selected views
        view_inds_expanded = view_inds.view(B, self.num_seed, 1, 1).expand(
            -1, -1, -1, 3
        )  # (B, num_seed, 1, 3)
        view_xyz = torch.gather(template_views, 2, view_inds_expanded).squeeze(2)
        # view_xyz: (B, num_seed, 3)

        # Convert view directions to rotation matrices
        view_xyz_flat = view_xyz.view(B * self.num_seed, 3)
        batch_angle = torch.zeros(
            view_xyz_flat.size(0), dtype=view_xyz_flat.dtype, device=view_xyz_flat.device
        )
        view_rot = batch_viewpoint_params_to_matrix(-view_xyz_flat, batch_angle)
        view_rot = view_rot.view(B, self.num_seed, 3, 3)

        # Stage 3: Grasp generation
        grasp_preds, view_features, before_generator = self.grasp_generator(
            seed_xyz, seed_xyz, view_rot, seed_features
        )
        # grasp_preds: (B, num_seed, num_angle*num_depth*2)

        AD = self.num_angle * self.num_depth
        grasp_scores = torch.sigmoid(grasp_preds[:, :, 0:AD]).view(
            B, self.num_seed, self.num_angle, self.num_depth
        )
        grasp_widths = torch.sigmoid(grasp_preds[:, :, AD:AD*2]).view(
            B, self.num_seed, self.num_angle, self.num_depth
        )

        return {
            'objectness': objectness_pred,
            'heatmap': heatmap_pred,
            'view_heatmap': view_heatmap_pred,
            'view_xyz': view_xyz,
            'seed_xyz': seed_xyz,
            'grasp_scores': grasp_scores,
            'grasp_widths': grasp_widths,
            'view_features': view_features,
        }


class GraspNetONNXInference(nn.Module):
    """
    Simplified ONNX model for inference (no training components).
    """
    def __init__(self, num_seed=1024, num_view=300, num_angle=48,
                 num_depth=5, half_views=False):
        super().__init__()
        self.model = GraspNetONNX(
            in_channels=3, num_seed=num_seed, num_view=num_view,
            num_angle=num_angle, num_depth=num_depth,
            is_training=False, half_views=half_views
        )

    def forward(self, xyz):
        """
        Simplified forward for ONNX export.

        Args:
            xyz: (B, N, 3) input point cloud

        Returns:
            grasp_scores: (B, num_seed, num_angle, num_depth)
            grasp_widths: (B, num_seed, num_angle, num_depth)
            seed_xyz: (B, num_seed, 3)
        """
        outputs = self.model(xyz)
        return outputs['grasp_scores'], outputs['grasp_widths'], outputs['seed_xyz']
