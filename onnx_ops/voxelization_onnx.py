"""
ONNX-compatible voxelization and sparse tensor operations
Provides alternatives to MinkowskiEngine for ONNX export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def voxelize_point_cloud(points, voxel_size, coord_range=None):
    """
    Voxelize point cloud by quantizing coordinates.

    Args:
        points: (N, 3) or (B, N, 3) point coordinates
        voxel_size: float, size of each voxel
        coord_range: optional tuple of (min_coords, max_coords) for each dimension

    Returns:
        voxel_coords: (M, 3) or (B, M, 3) unique voxel coordinates
        point_to_voxel: (N,) or (B, N) mapping from points to voxels
        voxel_features: aggregated features per voxel
    """
    batched = points.dim() == 3

    if not batched:
        points = points.unsqueeze(0)  # (1, N, 3)

    B, N, _ = points.shape
    device = points.device

    # Quantize coordinates to voxel grid
    voxel_coords_continuous = points / voxel_size
    voxel_coords = torch.floor(voxel_coords_continuous).long()  # (B, N, 3)

    # For each batch, find unique voxels
    all_voxel_coords = []
    all_point_to_voxel = []
    all_voxel_indices = []

    for b in range(B):
        batch_voxel_coords = voxel_coords[b]  # (N, 3)

        # Convert to unique representation
        # Use a hash-like approach: encode 3D coords to 1D
        # Assuming coords are reasonably bounded
        if coord_range is not None:
            min_coord = coord_range[0]
            max_coord = coord_range[1]
            grid_size = int((max_coord - min_coord) / voxel_size) + 1
        else:
            # Compute from data
            min_coord = batch_voxel_coords.min(dim=0)[0]
            max_coord = batch_voxel_coords.max(dim=0)[0]
            grid_size = (max_coord - min_coord).max().item() + 1

        # Shift coords to be non-negative
        shifted_coords = batch_voxel_coords - batch_voxel_coords.min(dim=0)[0]

        # Encode to 1D (assuming grid_size is reasonable)
        encoded = (shifted_coords[:, 0] * grid_size * grid_size +
                   shifted_coords[:, 1] * grid_size +
                   shifted_coords[:, 2])

        # Find unique voxels
        unique_encoded, inverse_indices = torch.unique(encoded, return_inverse=True)

        # Decode back to 3D
        unique_z = unique_encoded % grid_size
        unique_y = (unique_encoded // grid_size) % grid_size
        unique_x = unique_encoded // (grid_size * grid_size)
        unique_coords = torch.stack([unique_x, unique_y, unique_z], dim=1)  # (M, 3)

        # Restore original offset
        unique_coords = unique_coords + batch_voxel_coords.min(dim=0)[0]

        all_voxel_coords.append(unique_coords)
        all_point_to_voxel.append(inverse_indices)
        all_voxel_indices.append(torch.arange(len(unique_coords), device=device))

    if not batched:
        return all_voxel_coords[0], all_point_to_voxel[0]
    else:
        return all_voxel_coords, all_point_to_voxel


def aggregate_features_by_voxel(features, point_to_voxel, num_voxels, reduction='mean'):
    """
    Aggregate point features by voxel.

    Args:
        features: (N, C) point features
        point_to_voxel: (N,) mapping from points to voxels
        num_voxels: int, total number of voxels
        reduction: 'mean', 'max', or 'sum'

    Returns:
        voxel_features: (num_voxels, C) aggregated features
    """
    N, C = features.shape
    device = features.device

    # Initialize voxel features
    voxel_features = torch.zeros(num_voxels, C, device=device)

    if reduction == 'mean':
        # Sum features and count points per voxel
        voxel_features.scatter_add_(0, point_to_voxel.unsqueeze(1).expand(-1, C), features)
        counts = torch.bincount(point_to_voxel, minlength=num_voxels).float().unsqueeze(1)
        counts = torch.clamp(counts, min=1)  # Avoid division by zero
        voxel_features = voxel_features / counts

    elif reduction == 'max':
        # Max pooling - use scatter with max reduction
        voxel_features, _ = torch_scatter.scatter_max(
            features, point_to_voxel.unsqueeze(1).expand(-1, C), dim=0,
            dim_size=num_voxels
        )

    elif reduction == 'sum':
        voxel_features.scatter_add_(0, point_to_voxel.unsqueeze(1).expand(-1, C), features)

    return voxel_features


class SparseToDense(nn.Module):
    """
    Convert sparse tensor representation to dense for ONNX export.
    This is a workaround for MinkowskiEngine sparse tensors.
    """
    def __init__(self, voxel_size, grid_size, feature_dim):
        super().__init__()
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.feature_dim = feature_dim

    def forward(self, coords, features):
        """
        Args:
            coords: (N, 4) sparse coordinates [batch_idx, x, y, z]
            features: (N, C) sparse features

        Returns:
            dense_tensor: (B, C, D, H, W) dense tensor
        """
        B = coords[:, 0].max().item() + 1
        N, C = features.shape

        # Initialize dense tensor
        dense_tensor = torch.zeros(
            B, C, self.grid_size, self.grid_size, self.grid_size,
            device=features.device, dtype=features.dtype
        )

        # Fill in features at sparse locations
        batch_idx = coords[:, 0].long()
        x = coords[:, 1].long()
        y = coords[:, 2].long()
        z = coords[:, 3].long()

        # Clamp coordinates to grid bounds
        x = torch.clamp(x, 0, self.grid_size - 1)
        y = torch.clamp(y, 0, self.grid_size - 1)
        z = torch.clamp(z, 0, self.grid_size - 1)

        # Use advanced indexing to fill
        dense_tensor[batch_idx, :, x, y, z] = features

        return dense_tensor


class DenseToSparse(nn.Module):
    """
    Convert dense tensor back to sparse representation.
    """
    def __init__(self, threshold=1e-6):
        super().__init__()
        self.threshold = threshold

    def forward(self, dense_tensor):
        """
        Args:
            dense_tensor: (B, C, D, H, W) dense tensor

        Returns:
            coords: (N, 4) sparse coordinates [batch_idx, x, y, z]
            features: (N, C) sparse features
        """
        B, C, D, H, W = dense_tensor.shape

        # Find non-zero locations
        # Use magnitude across channels
        magnitude = torch.sum(dense_tensor ** 2, dim=1)  # (B, D, H, W)
        mask = magnitude > self.threshold

        # Get coordinates of non-zero elements
        batch_idx, x, y, z = torch.where(mask)

        # Extract features
        features = dense_tensor[batch_idx, :, x, y, z]  # (N, C)

        # Construct coordinate tensor
        coords = torch.stack([batch_idx.float(), x.float(), y.float(), z.float()], dim=1)

        return coords, features


class SparseConv3d(nn.Module):
    """
    Sparse 3D convolution using dense operations for ONNX export.
    This is a memory-intensive alternative to MinkowskiEngine.

    Note: This should only be used for ONNX export. For training,
    use MinkowskiEngine's native sparse convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=kernel_size // 2,
            bias=bias
        )

    def forward(self, x):
        """
        Args:
            x: (B, C_in, D, H, W) dense tensor

        Returns:
            output: (B, C_out, D', H', W') dense tensor
        """
        return self.conv(x)


class SparseBatchNorm(nn.Module):
    """
    Batch normalization for sparse tensors using dense operations.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)


class SparseReLU(nn.Module):
    """ReLU for sparse tensors (works on dense representation)."""
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)


class SparseConvolutionWrapper(nn.Module):
    """
    Wrapper to convert between sparse and dense for ONNX export.
    This allows using standard PyTorch conv3d operations.
    """
    def __init__(self, minkowski_conv, grid_size=128):
        super().__init__()
        self.grid_size = grid_size

        # Extract parameters from MinkowskiEngine conv
        self.in_channels = minkowski_conv.in_channels
        self.out_channels = minkowski_conv.out_channels
        self.kernel_size = minkowski_conv.kernel_size
        self.stride = minkowski_conv.stride
        self.dilation = minkowski_conv.dilation

        # Create equivalent dense convolution
        self.conv = nn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.kernel_size // 2,
            bias=minkowski_conv.bias is not None
        )

        # Copy weights from MinkowskiEngine conv
        with torch.no_grad():
            # Note: Weight layout may differ between MinkowskiEngine and PyTorch
            # This is a simplified conversion - may need adjustment
            if hasattr(minkowski_conv, 'kernel'):
                self.conv.weight.copy_(minkowski_conv.kernel)
            if minkowski_conv.bias is not None:
                self.conv.bias.copy_(minkowski_conv.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) dense tensor

        Returns:
            output: (B, C_out, D', H', W') dense tensor
        """
        return self.conv(x)


def convert_minkowski_model_to_dense(minkowski_model, grid_size=128):
    """
    Convert a MinkowskiEngine model to use dense convolutions for ONNX export.

    Args:
        minkowski_model: model using MinkowskiEngine
        grid_size: size of dense grid

    Returns:
        dense_model: equivalent model using dense convolutions
    """
    # This is a placeholder - actual implementation would need to:
    # 1. Traverse the model architecture
    # 2. Replace MinkowskiConvolution with SparseConvolutionWrapper
    # 3. Replace MinkowskiBatchNorm with SparseBatchNorm
    # 4. Replace other MinkowskiEngine modules
    # 5. Copy weights appropriately

    raise NotImplementedError(
        "Automatic conversion from MinkowskiEngine to dense is complex. "
        "Consider manually re-implementing the model architecture using dense operations."
    )
