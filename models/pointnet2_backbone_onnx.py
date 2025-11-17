"""
ONNX-compatible PointNet++ backbone
Alternative to MinkowskiEngine-based ResUNet14/Res4UNet14 for ONNX export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'onnx_ops'))

from onnx_ops.pointnet2_onnx import (
    furthest_point_sample,
    ball_query,
    group_points,
    three_nn,
    three_interpolate
)


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction module.
    Replaces sparse convolutions with point-based operations.
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # MLP layers
        # For group_all, in_channel doesn't include grouped xyz (not concatenated)
        # For normal SA, in_channel should already include the +3 from caller
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) input points
            points: (B, C, N) input features

        Returns:
            new_xyz: (B, npoint, 3) sampled points
            new_points: (B, mlp[-1], npoint) output features
        """
        B, N, _ = xyz.shape

        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            new_points = points.unsqueeze(2)  # (B, C, 1, N) - group all points together
        else:
            # Sample points using FPS
            fps_idx = furthest_point_sample(xyz, self.npoint)  # (B, npoint)

            # Gather new xyz
            fps_idx_expanded = fps_idx.unsqueeze(-1).expand(-1, -1, 3)  # (B, npoint, 3)
            new_xyz = torch.gather(xyz, 1, fps_idx_expanded)  # (B, npoint, 3)

            # Query ball
            idx = ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B, npoint, nsample)

            # Group points
            grouped_xyz = group_points(xyz.transpose(1, 2), idx)  # (B, 3, npoint, nsample)
            grouped_xyz_normalized = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)

            if points is not None:
                grouped_points = group_points(points, idx)  # (B, C, npoint, nsample)
                new_points = torch.cat([grouped_xyz_normalized, grouped_points], dim=1)
            else:
                new_points = grouped_xyz_normalized

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling
        new_points = torch.max(new_points, dim=-1)[0]  # (B, mlp[-1], npoint)

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation module.
    Upsamples features using interpolation.
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, N1, 3) points at lower resolution
            xyz2: (B, N2, 3) points at higher resolution
            points1: (B, C1, N1) features at lower resolution
            points2: (B, C2, N2) features at higher resolution

        Returns:
            new_points: (B, mlp[-1], N2) upsampled features
        """
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape

        if N1 == 1:
            # Global feature - repeat for all points
            interpolated_points = points1.repeat(1, 1, N2)
        else:
            # Find 3 nearest neighbors and interpolate
            dists, idx = three_nn(xyz2, xyz1)  # (B, N2, 3)

            # Compute interpolation weights (inverse distance weighting)
            dists = torch.clamp(dists, min=1e-10)
            weight = 1.0 / dists  # (B, N2, 3)
            weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # normalize

            # Interpolate features
            interpolated_points = three_interpolate(points1, idx, weight)  # (B, C1, N2)

        # Concatenate with skip connection features
        if points2 is not None:
            new_points = torch.cat([interpolated_points, points2], dim=1)
        else:
            new_points = interpolated_points

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class PointNet2Backbone(nn.Module):
    """
    PointNet++ backbone for grasp detection.
    ONNX-compatible alternative to ResUNet14/Res4UNet14.
    """
    def __init__(self, in_channels=3, out_channels=3, feature_dim=512, half_views=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.half_views = half_views

        if half_views:
            # Lightweight version
            # Set abstraction layers (encoder)
            self.sa1 = PointNetSetAbstraction(1024, 0.02, 32, in_channels + 3, [32, 32, 64])
            self.sa2 = PointNetSetAbstraction(512, 0.04, 32, 64 + 3, [64, 64, 128])
            self.sa3 = PointNetSetAbstraction(256, 0.08, 32, 128 + 3, [128, 128, 128])
            self.sa4 = PointNetSetAbstraction(None, None, None, 128, [128, 128, 128], group_all=True)

            # Feature propagation layers (decoder)
            self.fp4 = PointNetFeaturePropagation(128 + 128, [128, 128])
            self.fp3 = PointNetFeaturePropagation(128 + 128, [128, 128])
            self.fp2 = PointNetFeaturePropagation(64 + 128, [128, 64])
            self.fp1 = PointNetFeaturePropagation(in_channels + 64, [64, 64, 64])

            # Output heads
            self.conv1 = nn.Conv1d(64, 64, 1)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, out_channels, 1)

            self.feature_out_dim = 128
        else:
            # Full version
            # Set abstraction layers (encoder)
            # in_channels accounts for concat of grouped xyz (3) + input features
            self.sa1 = PointNetSetAbstraction(2048, 0.02, 32, in_channels + 3, [64, 64, 128])
            self.sa2 = PointNetSetAbstraction(1024, 0.04, 32, 128 + 3, [128, 128, 256])
            self.sa3 = PointNetSetAbstraction(512, 0.08, 32, 256 + 3, [256, 256, 512])
            self.sa4 = PointNetSetAbstraction(None, None, None, 512, [512, 512, 512], group_all=True)

            # Feature propagation layers (decoder)
            self.fp4 = PointNetFeaturePropagation(512 + 512, [512, 512])
            self.fp3 = PointNetFeaturePropagation(256 + 512, [512, 256])
            self.fp2 = PointNetFeaturePropagation(128 + 256, [256, 128])
            self.fp1 = PointNetFeaturePropagation(in_channels + 128, [128, 128, 128])

            # Output heads
            self.conv1 = nn.Conv1d(128, 128, 1)
            self.bn1 = nn.BatchNorm1d(128)
            self.conv2 = nn.Conv1d(128, out_channels, 1)

            # Feature dimension is the output of fp1 (last element of the MLP)
            self.feature_out_dim = 128

    def forward(self, xyz, features=None, return_features=False):
        """
        Args:
            xyz: (B, N, 3) input points
            features: (B, C, N) input features (optional, will use xyz if None)
            return_features: whether to return intermediate features

        Returns:
            output: (B, out_channels, N) output predictions
            features: (B, feature_dim, N) intermediate features (if return_features=True)
        """
        B, N, _ = xyz.shape

        # Initial features
        if features is None:
            l0_points = xyz.transpose(1, 2)  # (B, 3, N)
        else:
            l0_points = features
        l0_xyz = xyz

        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature propagation layers (upsample from low res to high res)
        l3_points = self.fp4(l4_xyz, l3_xyz, l4_points, l3_points)  # 1 -> 512 points
        l2_points = self.fp3(l3_xyz, l2_xyz, l3_points, l2_points)  # 512 -> 1024 points
        l1_points = self.fp2(l2_xyz, l1_xyz, l2_points, l1_points)  # 1024 -> 2048 points
        l0_points_new = self.fp1(l1_xyz, l0_xyz, l1_points, l0_points)  # 2048 -> N points

        # Feature for grasp detection
        feature_output = l0_points_new

        # Output head
        x = F.relu(self.bn1(self.conv1(l0_points_new)))
        x = self.conv2(x)  # (B, out_channels, N)

        if return_features:
            return x, feature_output
        else:
            return x


class PointNet2BackboneWrapper(nn.Module):
    """
    Wrapper to make PointNet2Backbone compatible with the original interface.
    Simulates MinkowskiEngine sparse tensor interface.
    """
    def __init__(self, in_channels=3, out_channels=3, feature_dim=512, half_views=False, **kwargs):
        super().__init__()
        self.backbone = PointNet2Backbone(in_channels, out_channels, feature_dim, half_views)
        self.feature_dim = self.backbone.feature_out_dim

    def forward(self, sparse_input, return_features=False):
        """
        Args:
            sparse_input: dict with 'coords' (B, N, 3) and 'features' (B, C, N)
                         OR MinkowskiEngine SparseTensor (for compatibility)

        Returns:
            output: dict with 'coords', 'features', and 'predictions'
        """
        # Handle both dict and MinkowskiEngine SparseTensor input
        if isinstance(sparse_input, dict):
            xyz = sparse_input['coords']
            features = sparse_input.get('features', None)
        else:
            # MinkowskiEngine SparseTensor
            # Convert to dense format
            coords = sparse_input.C  # (N, 4) [batch, x, y, z]
            features = sparse_input.F  # (N, C)

            # Group by batch
            B = coords[:, 0].max().item() + 1
            xyz_list = []
            feat_list = []
            for b in range(B):
                mask = coords[:, 0] == b
                # Note: coords are voxel indices, need to convert back to xyz
                # This is a simplified version
                xyz_list.append(coords[mask, 1:].float())
                feat_list.append(features[mask])

            # Pad to same length
            max_len = max(len(x) for x in xyz_list)
            xyz_padded = []
            feat_padded = []
            for b in range(B):
                n = len(xyz_list[b])
                if n < max_len:
                    # Pad by repeating first point
                    xyz_b = torch.cat([xyz_list[b], xyz_list[b][:1].repeat(max_len - n, 1)])
                    feat_b = torch.cat([feat_list[b], feat_list[b][:1].repeat(max_len - n, 1)])
                else:
                    xyz_b = xyz_list[b]
                    feat_b = feat_list[b]
                xyz_padded.append(xyz_b)
                feat_padded.append(feat_b)

            xyz = torch.stack(xyz_padded)  # (B, max_len, 3)
            features = torch.stack(feat_padded).transpose(1, 2)  # (B, C, max_len)

        # Forward pass
        if return_features:
            output, feat_output = self.backbone(xyz, features, return_features=True)
            return output, feat_output
        else:
            output = self.backbone(xyz, features, return_features=False)
            return output
