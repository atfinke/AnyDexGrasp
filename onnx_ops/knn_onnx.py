"""
ONNX-compatible KNN operations
Re-implements knn/src/cuda/knn.cu in PyTorch for ONNX export
"""

import torch
import torch.nn as nn


def pairwise_distance(ref_points, query_points):
    """
    Compute pairwise squared Euclidean distances between reference and query points.

    Args:
        ref_points: (B, N, D) reference points
        query_points: (B, M, D) query points

    Returns:
        distances: (B, M, N) squared distances
    """
    # Expand dimensions for broadcasting
    # ref_points: (B, 1, N, D)
    # query_points: (B, M, 1, D)
    ref_expanded = ref_points.unsqueeze(1)  # (B, 1, N, D)
    query_expanded = query_points.unsqueeze(2)  # (B, M, 1, D)

    # Compute squared differences and sum over dimension
    diff = query_expanded - ref_expanded  # (B, M, N, D)
    distances = torch.sum(diff ** 2, dim=-1)  # (B, M, N)

    return distances


def knn_distance(ref_points, query_points, k):
    """
    Compute K-nearest neighbors using PyTorch operations (ONNX-compatible).

    Equivalent to knn.cu's knn_device function.

    Args:
        ref_points: (B, N, D) reference points
        query_points: (B, M, D) query points
        k: number of nearest neighbors

    Returns:
        distances: (B, M, K) distances to k nearest neighbors (squared, not sqrt)
        indices: (B, M, K) indices of k nearest neighbors (0-indexed)
    """
    # Compute all pairwise distances
    distances = pairwise_distance(ref_points, query_points)  # (B, M, N)

    # Find k nearest neighbors
    # PyTorch's topk returns (values, indices) in descending order by default
    # We want ascending order (smallest distances first), so use largest=False
    knn_dist, knn_idx = torch.topk(distances, k, dim=2, largest=False, sorted=True)

    # Note: Original CUDA code doesn't take sqrt, keeping squared distances
    # If sqrt is needed, uncomment: knn_dist = torch.sqrt(knn_dist)

    return knn_dist, knn_idx


def knn_points(ref_points, query_points, k, return_sorted=True):
    """
    K-nearest neighbors with additional options.

    Args:
        ref_points: (B, N, D) reference points
        query_points: (B, M, D) query points
        k: number of nearest neighbors
        return_sorted: whether to return sorted results

    Returns:
        Dictionary with:
            - 'distances': (B, M, K) distances to k nearest neighbors
            - 'indices': (B, M, K) indices of k nearest neighbors
    """
    distances, indices = knn_distance(ref_points, query_points, k)

    if not return_sorted:
        # Note: For ONNX compatibility, we keep sorted results
        # Unsorted version would require additional operations
        pass

    return {
        'distances': distances,
        'indices': indices,
    }


class KNNModule(nn.Module):
    """
    KNN module wrapper for ONNX export.
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, ref_points, query_points):
        """
        Args:
            ref_points: (B, N, D) reference points
            query_points: (B, M, D) query points

        Returns:
            distances: (B, M, K) distances
            indices: (B, M, K) indices
        """
        return knn_distance(ref_points, query_points, self.k)


def squared_distance_matrix(points1, points2=None):
    """
    Compute squared distance matrix between two point sets.
    If points2 is None, compute pairwise distances within points1.

    Args:
        points1: (B, N, D) or (N, D)
        points2: (B, M, D) or (M, D) or None

    Returns:
        distances: (B, N, M) or (N, M) squared distances
    """
    if points2 is None:
        points2 = points1

    # Handle both batched and unbatched inputs
    if points1.dim() == 2:
        # Unbatched: (N, D)
        # Use torch.cdist for efficiency
        return torch.cdist(points1.unsqueeze(0), points2.unsqueeze(0)).squeeze(0) ** 2
    else:
        # Batched: (B, N, D)
        return torch.cdist(points1, points2) ** 2


# Alias for compatibility
compute_distance = pairwise_distance
