"""
ONNX-compatible PointNet2 operations
Re-implements pointnet2/_ext_src/src/*.cu in PyTorch for ONNX export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ball_query(xyz, new_xyz, radius, nsample):
    """
    Ball query: find all points within a radius.
    Re-implements ball_query_gpu.cu

    Args:
        xyz: (B, N, 3) input points
        new_xyz: (B, M, 3) query centers
        radius: float, search radius
        nsample: int, maximum number of points in each ball

    Returns:
        idx: (B, M, nsample) indices of points in each ball
    """
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape

    # Compute pairwise distances between query centers and all points
    # new_xyz: (B, M, 1, 3), xyz: (B, 1, N, 3)
    diff = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)  # (B, M, N, 3)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (B, M, N)

    # Find points within radius
    radius_sq = radius * radius
    mask = dist_sq < radius_sq  # (B, M, N)

    # For each query point, select up to nsample nearest neighbors within radius
    # Sort by distance and take top nsample
    dist_sq_masked = torch.where(mask, dist_sq, torch.full_like(dist_sq, float('inf')))

    # Get sorted indices
    sorted_dist, sorted_idx = torch.sort(dist_sq_masked, dim=2)  # (B, M, N)

    # Take first nsample points
    idx = sorted_idx[:, :, :nsample]  # (B, M, nsample)

    # If a query point has fewer than nsample neighbors, repeat the first valid index
    # This matches the CUDA implementation behavior (lines 40-42)
    first_valid_idx = idx[:, :, 0:1]  # (B, M, 1)
    idx = torch.where(sorted_dist[:, :, :nsample] == float('inf'),
                      first_valid_idx.expand(-1, -1, nsample),
                      idx)

    return idx


def furthest_point_sample(xyz, npoint):
    """
    Furthest Point Sampling (FPS).
    Re-implements sampling_gpu.cu

    Args:
        xyz: (B, N, 3) input points
        npoint: int, number of points to sample

    Returns:
        idx: (B, npoint) sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape

    # Initialize output
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)

    # Distance array: keep track of min distance to selected points
    distance = torch.ones(B, N, device=device) * 1e10

    # Randomly select first point (or use point 0)
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    # Batch indices for gathering
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        # Record current farthest point
        idx[:, i] = farthest

        # Get coordinates of current farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # (B, 1, 3)

        # Compute distances from centroid to all points
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)

        # Update minimum distances
        distance = torch.min(distance, dist)

        # Select next farthest point (point with maximum minimum distance)
        farthest = torch.max(distance, dim=1)[1]  # (B,)

    return idx


def group_points(features, idx):
    """
    Group points by indices (gather operation).
    Re-implements group_points_gpu.cu (gather_points_kernel)

    Args:
        features: (B, C, N) input features
        idx: (B, M) or (B, M, nsample) indices to gather

    Returns:
        output: (B, C, M) or (B, C, M, nsample) gathered features
    """
    B, C, N = features.shape

    if idx.dim() == 2:
        # (B, M) indices
        M = idx.shape[1]
        # Expand idx for gathering: (B, C, M)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
        output = torch.gather(features, dim=2, index=idx_expanded)
        return output
    elif idx.dim() == 3:
        # (B, M, nsample) indices
        M, nsample = idx.shape[1], idx.shape[2]
        # Reshape for gathering
        idx_flat = idx.view(B, -1)  # (B, M*nsample)
        idx_expanded = idx_flat.unsqueeze(1).expand(-1, C, -1)  # (B, C, M*nsample)
        output_flat = torch.gather(features, dim=2, index=idx_expanded)  # (B, C, M*nsample)
        output = output_flat.view(B, C, M, nsample)  # (B, C, M, nsample)
        return output
    else:
        raise ValueError(f"idx must be 2D or 3D, got shape {idx.shape}")


def three_nn(unknown, known):
    """
    Find 3 nearest neighbors for interpolation.
    Re-implements interpolate_gpu.cu (three_nn_kernel)

    Args:
        unknown: (B, N, 3) points to interpolate to
        known: (B, M, 3) points to interpolate from

    Returns:
        dist: (B, N, 3) squared distances to 3 nearest neighbors
        idx: (B, N, 3) indices of 3 nearest neighbors
    """
    B, N, _ = unknown.shape
    M = known.shape[1]

    # Compute pairwise distances
    # unknown: (B, N, 1, 3), known: (B, 1, M, 3)
    diff = unknown.unsqueeze(2) - known.unsqueeze(1)  # (B, N, M, 3)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (B, N, M)

    # Find 3 nearest neighbors
    dist, idx = torch.topk(dist_sq, k=3, dim=2, largest=False, sorted=True)  # (B, N, 3)

    return dist, idx


def three_interpolate(features, idx, weight):
    """
    Interpolate features using 3 nearest neighbors.
    Re-implements interpolate_gpu.cu (three_interpolate_kernel)

    Args:
        features: (B, C, M) features to interpolate from
        idx: (B, N, 3) indices of 3 nearest neighbors
        weight: (B, N, 3) interpolation weights

    Returns:
        output: (B, C, N) interpolated features
    """
    B, C, M = features.shape
    N = idx.shape[1]

    # Gather features for 3 nearest neighbors
    # idx: (B, N, 3) -> (B, N*3)
    idx_flat = idx.view(B, -1)  # (B, N*3)
    idx_expanded = idx_flat.unsqueeze(1).expand(-1, C, -1)  # (B, C, N*3)

    # Gather: (B, C, N*3)
    gathered = torch.gather(features, dim=2, index=idx_expanded)
    gathered = gathered.view(B, C, N, 3)  # (B, C, N, 3)

    # Apply weights and sum
    weight_expanded = weight.permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, 3)
    output = torch.sum(gathered * weight_expanded, dim=-1)  # (B, C, N)

    return output


def cylinder_query(xyz, new_xyz, rot_c2w, radius, hmin, hmax, nsample):
    """
    Cylinder query: find points within a cylindrical region.
    Re-implements cylinder_query_gpu.cu

    This is specific to grasp detection - transforms points to cylinder local frame
    and checks if they're within radius (in y-z plane) and height range (x-axis).

    Args:
        xyz: (B, N, 3) input points
        new_xyz: (B, M, 3) cylinder centers
        rot_c2w: (B, M, 3, 3) rotation matrices from cylinder to world frame
        radius: float, cylinder radius
        hmin: float, minimum height
        hmax: float, maximum height
        nsample: int, maximum number of points in each cylinder

    Returns:
        idx: (B, M, nsample) indices of points in each cylinder
    """
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]

    # Compute relative positions
    # xyz: (B, 1, N, 3), new_xyz: (B, M, 1, 3)
    rel_pos = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)  # (B, M, N, 3)

    # Transform to cylinder local frame
    # rot_c2w: (B, M, 3, 3) -> inverse is rot_c2w.transpose(-2, -1)
    rot_w2c = rot_c2w.transpose(-2, -1)  # (B, M, 3, 3)

    # Apply rotation: (B, M, N, 3) @ (B, M, 3, 3).T
    # rel_pos_local = (rel_pos @ rot_w2c.transpose(-2, -1))
    # Better: use torch.einsum for clarity
    rel_pos_local = torch.einsum('bmni,bmij->bmnj', rel_pos, rot_w2c)  # (B, M, N, 3)

    # Extract coordinates in cylinder frame
    x_local = rel_pos_local[..., 0]  # height axis
    y_local = rel_pos_local[..., 1]
    z_local = rel_pos_local[..., 2]

    # Check cylinder constraints
    # 1. Within radius (in y-z plane)
    dist_yz_sq = y_local ** 2 + z_local ** 2
    radius_sq = radius * radius
    in_radius = dist_yz_sq < radius_sq

    # 2. Within height range
    in_height = (x_local > hmin) & (x_local < hmax)

    # Combined mask
    in_cylinder = in_radius & in_height  # (B, M, N)

    # Sort by distance and select up to nsample points
    dist_yz_masked = torch.where(in_cylinder, dist_yz_sq, torch.full_like(dist_yz_sq, float('inf')))

    sorted_dist, sorted_idx = torch.sort(dist_yz_masked, dim=2)  # (B, M, N)

    # Take first nsample points
    idx = sorted_idx[:, :, :nsample]  # (B, M, nsample)

    # If insufficient points, repeat first valid index (matches CUDA behavior)
    first_valid_idx = idx[:, :, 0:1]
    idx = torch.where(sorted_dist[:, :, :nsample] == float('inf'),
                      first_valid_idx.expand(-1, -1, nsample),
                      idx)

    return idx


def cylinder_query_and_group(xyz, new_xyz, rot_c2w, features, radius, hmin, hmax, nsample):
    """
    Cylinder query followed by grouping.
    Combines cylinder_query + group_points for convenience.

    Args:
        xyz: (B, N, 3) input points
        new_xyz: (B, M, 3) cylinder centers
        rot_c2w: (B, M, 3, 3) rotation matrices
        features: (B, C, N) input features
        radius: float
        hmin: float
        hmax: float
        nsample: int

    Returns:
        grouped_features: (B, C, M, nsample) grouped features
        grouped_xyz: (B, M, nsample, 3) grouped point coordinates (relative to center)
    """
    # Query cylinder
    idx = cylinder_query(xyz, new_xyz, rot_c2w, radius, hmin, hmax, nsample)  # (B, M, nsample)

    # Group features
    grouped_features = group_points(features, idx)  # (B, C, M, nsample)

    # Group xyz coordinates
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]

    # Gather xyz: (B, N, 3) -> (B, M, nsample, 3)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, M, nsample, 3)
    xyz_expanded = xyz.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, N, 3)
    grouped_xyz = torch.gather(xyz_expanded, dim=2, index=idx_expanded)  # (B, M, nsample, 3)

    # Make coordinates relative to center
    grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # (B, M, nsample, 3)

    return grouped_features, grouped_xyz


class BallQueryModule(nn.Module):
    """Ball query module for ONNX export."""
    def __init__(self, radius, nsample):
        super().__init__()
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz, new_xyz):
        return ball_query(xyz, new_xyz, self.radius, self.nsample)


class FurthestPointSampleModule(nn.Module):
    """FPS module for ONNX export."""
    def __init__(self, npoint):
        super().__init__()
        self.npoint = npoint

    def forward(self, xyz):
        return furthest_point_sample(xyz, self.npoint)


class GroupPointsModule(nn.Module):
    """Group points module for ONNX export."""
    def forward(self, features, idx):
        return group_points(features, idx)


class ThreeInterpolateModule(nn.Module):
    """Three interpolate module for ONNX export."""
    def forward(self, features, idx, weight):
        return three_interpolate(features, idx, weight)


class CylinderQueryModule(nn.Module):
    """Cylinder query module for ONNX export."""
    def __init__(self, radius, hmin, hmax, nsample):
        super().__init__()
        self.radius = radius
        self.hmin = hmin
        self.hmax = hmax
        self.nsample = nsample

    def forward(self, xyz, new_xyz, rot_c2w):
        return cylinder_query(xyz, new_xyz, rot_c2w, self.radius,
                            self.hmin, self.hmax, self.nsample)
