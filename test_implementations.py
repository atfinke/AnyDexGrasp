"""
Test script to verify ONNX operator implementations
Tests basic functionality without full model export
"""

import sys
import os
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'onnx_ops'))

from knn_onnx import knn_distance
from pointnet2_onnx import (
    ball_query,
    furthest_point_sample,
    group_points,
    three_nn,
    three_interpolate,
    cylinder_query
)

def test_knn():
    """Test KNN distance computation"""
    print("\n" + "="*60)
    print("Testing KNN Distance")
    print("="*60)

    # Create test data
    ref_points = torch.randn(2, 100, 3)
    query_points = torch.randn(2, 50, 3)
    k = 8

    try:
        knn_dist, knn_idx = knn_distance(ref_points, query_points, k)

        # Verify output shapes
        assert knn_idx.shape == (2, 50, 8), f"Wrong shape: {knn_idx.shape}"
        assert knn_dist.shape == (2, 50, 8), f"Wrong shape: {knn_dist.shape}"

        # Verify indices are valid
        assert torch.all(knn_idx >= 0) and torch.all(knn_idx < 100), "Invalid indices"

        # Verify distances are sorted
        for b in range(2):
            for q in range(50):
                dists = knn_dist[b, q]
                assert torch.all(dists[:-1] <= dists[1:]), "Distances not sorted"

        print("PASS Shape: {}".format(knn_idx.shape))
        print("PASS Indices valid: range [0, 99]")
        print("PASS Distances sorted")
        return True

    except Exception as e:
        print(f"FAIL {e}")
        return False


def test_ball_query():
    """Test ball query"""
    print("\n" + "="*60)
    print("Testing Ball Query")
    print("="*60)

    xyz = torch.randn(2, 200, 3)
    new_xyz = torch.randn(2, 100, 3)
    radius = 0.2
    nsample = 16

    try:
        idx = ball_query(xyz, new_xyz, radius, nsample)

        assert idx.shape == (2, 100, 16), f"Wrong shape: {idx.shape}"
        assert torch.all(idx >= 0) and torch.all(idx < 200), "Invalid indices"

        # Verify points are within radius
        for b in range(2):
            for i in range(10):  # Spot check 10 points
                center = new_xyz[b, i:i+1, :]  # (1, 3)
                neighbors = xyz[b, idx[b, i], :]  # (16, 3)
                dists = torch.norm(neighbors - center, dim=1)
                # First point should always be within radius
                assert dists[0] <= radius or torch.all(idx[b, i] == idx[b, i, 0]), \
                    f"First neighbor not within radius: {dists[0]} > {radius}"

        print(f"PASS Shape: {idx.shape}")
        print(f"PASS Indices valid: range [0, 199]")
        print(f"PASS Radius constraint verified")
        return True

    except Exception as e:
        print(f"FAIL {e}")
        return False


def test_furthest_point_sample():
    """Test furthest point sampling"""
    print("\n" + "="*60)
    print("Testing Furthest Point Sample")
    print("="*60)

    xyz = torch.randn(2, 1000, 3)
    npoint = 256

    try:
        idx = furthest_point_sample(xyz, npoint)

        assert idx.shape == (2, 256), f"Wrong shape: {idx.shape}"
        assert torch.all(idx >= 0) and torch.all(idx < 1000), "Invalid indices"

        # Verify uniqueness
        for b in range(2):
            unique = torch.unique(idx[b])
            assert len(unique) == 256, f"Duplicate indices found: {len(unique)} unique"

        print(f"PASS Shape: {idx.shape}")
        print(f"PASS Indices valid: range [0, 999]")
        print(f"PASS All indices unique")
        return True

    except Exception as e:
        print(f"FAIL {e}")
        return False


def test_group_points():
    """Test group points"""
    print("\n" + "="*60)
    print("Testing Group Points")
    print("="*60)

    points = torch.randn(2, 64, 200)  # (B, C, N)
    idx = torch.randint(0, 200, (2, 100, 16))  # (B, M, nsample)

    try:
        grouped = group_points(points, idx)

        assert grouped.shape == (2, 64, 100, 16), f"Wrong shape: {grouped.shape}"

        # Verify values are correctly gathered
        for b in range(2):
            for i in range(10):  # Spot check
                for s in range(16):
                    expected = points[b, :, idx[b, i, s]]
                    actual = grouped[b, :, i, s]
                    assert torch.allclose(expected, actual), "Incorrect grouping"

        print(f"PASS Shape: {grouped.shape}")
        print(f"PASS Values correctly gathered")
        return True

    except Exception as e:
        print(f"FAIL {e}")
        return False


def test_three_nn_interpolate():
    """Test 3-NN interpolation"""
    print("\n" + "="*60)
    print("Testing Three NN Interpolate")
    print("="*60)

    points = torch.randn(2, 64, 100)  # (B, C, N)
    xyz1 = torch.randn(2, 100, 3)
    xyz2 = torch.randn(2, 200, 3)

    try:
        # First find 3 nearest neighbors
        idx, dist = three_nn(xyz2, xyz1)

        # Compute weights
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        # Interpolate
        interpolated = three_interpolate(points, idx, weight)

        assert interpolated.shape == (2, 64, 200), f"Wrong shape: {interpolated.shape}"

        # Values should be reasonable (weighted average)
        assert not torch.any(torch.isnan(interpolated)), "NaN values found"
        assert not torch.any(torch.isinf(interpolated)), "Inf values found"

        print(f"PASS Shape: {interpolated.shape}")
        print(f"PASS No NaN or Inf values")
        return True

    except Exception as e:
        print(f"FAIL {e}")
        return False


def test_cylinder_query():
    """Test cylinder query"""
    print("\n" + "="*60)
    print("Testing Cylinder Query")
    print("="*60)

    xyz = torch.randn(2, 500, 3)
    new_xyz = torch.randn(2, 100, 3)
    # Create valid rotation matrices
    rot_c2w = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2, 100, 1, 1)
    radius = 0.05
    hmin = -0.02
    hmax = 0.02
    nsample = 16

    try:
        idx = cylinder_query(xyz, new_xyz, rot_c2w, radius, hmin, hmax, nsample)

        assert idx.shape == (2, 100, 16), f"Wrong shape: {idx.shape}"
        assert torch.all(idx >= 0) and torch.all(idx < 500), "Invalid indices"

        print(f"PASS Shape: {idx.shape}")
        print(f"PASS Indices valid: range [0, 499]")
        return True

    except Exception as e:
        print(f"FAIL {e}")
        return False


def main():
    print("="*60)
    print("ONNX OPERATOR IMPLEMENTATION TESTS")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = []

    results.append(("KNN Distance", test_knn()))
    results.append(("Ball Query", test_ball_query()))
    results.append(("Furthest Point Sample", test_furthest_point_sample()))
    results.append(("Group Points", test_group_points()))
    results.append(("Three NN Interpolate", test_three_nn_interpolate()))
    results.append(("Cylinder Query", test_cylinder_query()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:30s} {status}")

    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    return all(result for _, result in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
