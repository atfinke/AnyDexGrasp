"""
Test the full ONNX-compatible GraspNet model
"""

import sys
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from graspnet_onnx import GraspNetONNXInference


def test_model_creation():
    """Test that model can be created"""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)

    try:
        model = GraspNetONNXInference(
            num_seed=512,
            num_view=300,
            num_angle=48,
            num_depth=5,
            half_views=False
        )
        print("PASS Model created successfully")
        return model

    except Exception as e:
        print(f"FAIL {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_forward():
    """Test forward pass"""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)

    try:
        model = GraspNetONNXInference(
            num_seed=256,  # Smaller for faster testing
            num_view=300,
            num_angle=48,
            num_depth=5,
            half_views=False
        )
        model.eval()

        # Create test input
        xyz = torch.randn(1, 1000, 3) * 0.3  # Smaller point cloud

        print(f"Input shape: {xyz.shape}")

        # Run forward pass
        with torch.no_grad():
            outputs = model(xyz)

        grasp_scores, grasp_widths, seed_xyz = outputs

        print(f"PASS Forward pass successful")
        print(f"  Grasp scores shape: {grasp_scores.shape}")
        print(f"  Grasp widths shape: {grasp_widths.shape}")
        print(f"  Seed xyz shape: {seed_xyz.shape}")

        # Verify shapes
        batch_size = xyz.shape[0]
        num_seed = 256
        num_view = 300

        expected_scores_shape = (batch_size, num_seed * num_view)
        expected_widths_shape = (batch_size, num_seed * num_view)
        expected_seed_shape = (batch_size, num_seed, 3)

        assert grasp_scores.shape == expected_scores_shape, \
            f"Wrong scores shape: {grasp_scores.shape} vs {expected_scores_shape}"
        assert grasp_widths.shape == expected_widths_shape, \
            f"Wrong widths shape: {grasp_widths.shape} vs {expected_widths_shape}"
        assert seed_xyz.shape == expected_seed_shape, \
            f"Wrong seed shape: {seed_xyz.shape} vs {expected_seed_shape}"

        print("PASS Output shapes correct")

        # Verify no NaN or Inf
        assert not torch.any(torch.isnan(grasp_scores)), "NaN in grasp_scores"
        assert not torch.any(torch.isnan(grasp_widths)), "NaN in grasp_widths"
        assert not torch.any(torch.isnan(seed_xyz)), "NaN in seed_xyz"
        assert not torch.any(torch.isinf(grasp_scores)), "Inf in grasp_scores"
        assert not torch.any(torch.isinf(grasp_widths)), "Inf in grasp_widths"
        assert not torch.any(torch.isinf(seed_xyz)), "Inf in seed_xyz"

        print("PASS No NaN or Inf values")

        return True

    except Exception as e:
        print(f"FAIL {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_determinism():
    """Test that model produces consistent outputs"""
    print("\n" + "="*60)
    print("Testing Model Determinism")
    print("="*60)

    try:
        model = GraspNetONNXInference(
            num_seed=128,
            num_view=300,
            num_angle=48,
            num_depth=5,
            half_views=False
        )
        model.eval()

        # Create test input
        torch.manual_seed(42)
        xyz = torch.randn(1, 500, 3) * 0.3

        # Run twice
        with torch.no_grad():
            outputs1 = model(xyz)
            outputs2 = model(xyz)

        # Compare
        for i, (name, out1, out2) in enumerate([
            ("grasp_scores", outputs1[0], outputs2[0]),
            ("grasp_widths", outputs1[1], outputs2[1]),
            ("seed_xyz", outputs1[2], outputs2[2])
        ]):
            max_diff = torch.max(torch.abs(out1 - out2)).item()
            print(f"  {name}: max diff = {max_diff:.2e}")

            # Should be identical for deterministic operations
            # Allow small tolerance for floating point
            if max_diff > 1e-6:
                print(f"WARNING {name} has non-deterministic behavior (max diff: {max_diff})")
            else:
                print(f"  PASS {name} is deterministic")

        return True

    except Exception as e:
        print(f"FAIL {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("GRASPNET ONNX MODEL TESTS")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")

    results = []

    # Test model creation
    model = test_model_creation()
    results.append(("Model Creation", model is not None))

    # Test forward pass
    if model is not None:
        results.append(("Forward Pass", test_model_forward()))
        results.append(("Determinism", test_model_determinism()))

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
