"""
Accuracy Validation Script for ONNX Model

Compares PyTorch model outputs with ONNX model outputs to ensure accuracy is maintained.
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from models.graspnet_onnx import GraspNetONNXInference


def create_test_point_cloud(num_points=10000, seed=None):
    """Create a synthetic point cloud for testing."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate random points in a reasonable range
    xyz = torch.randn(1, num_points, 3) * 0.3
    return xyz


def compare_tensors(tensor1, tensor2, name, tolerance=1e-3):
    """
    Compare two tensors and return statistics.

    Args:
        tensor1: First tensor (PyTorch or numpy)
        tensor2: Second tensor (PyTorch or numpy)
        name: Name of the tensor for reporting
        tolerance: Maximum allowed difference

    Returns:
        dict with comparison statistics
    """
    # Convert to numpy if needed
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.detach().cpu().numpy()
    if torch.is_tensor(tensor2):
        tensor2 = tensor2.detach().cpu().numpy()

    # Compute differences
    abs_diff = np.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (np.abs(tensor1) + 1e-8)

    stats = {
        'name': name,
        'max_abs_error': float(np.max(abs_diff)),
        'mean_abs_error': float(np.mean(abs_diff)),
        'max_rel_error': float(np.max(rel_diff)),
        'mean_rel_error': float(np.mean(rel_diff)),
        'passed': float(np.max(abs_diff)) < tolerance
    }

    return stats


def validate_pytorch_model(num_tests=10, num_points=10000):
    """
    Validate PyTorch ONNX model against itself for consistency.

    Args:
        num_tests: Number of random test cases
        num_points: Number of points per cloud

    Returns:
        List of test statistics
    """
    print("="*80)
    print("VALIDATING PYTORCH ONNX MODEL")
    print("="*80)

    # Create model
    model = GraspNetONNXInference(
        num_seed=512,  # Use smaller values for faster testing
        num_view=300,
        num_angle=48,
        num_depth=5,
        half_views=False
    )
    model.eval()

    results = []

    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")
        print("-" * 40)

        # Create test input
        xyz = create_test_point_cloud(num_points=num_points, seed=i)

        # Run model twice
        with torch.no_grad():
            output1 = model(xyz)
            output2 = model(xyz)

        # Compare outputs (should be identical for deterministic ops)
        stats = []
        stats.append(compare_tensors(
            output1[0], output2[0], 'grasp_scores', tolerance=1e-6
        ))
        stats.append(compare_tensors(
            output1[1], output2[1], 'grasp_widths', tolerance=1e-6
        ))
        stats.append(compare_tensors(
            output1[2], output2[2], 'seed_xyz', tolerance=1e-6
        ))

        # Print results
        for stat in stats:
            status = "PASS" if stat['passed'] else "FAIL"
            print(f"  {stat['name']:20s} | Max: {stat['max_abs_error']:.2e} | Mean: {stat['mean_abs_error']:.2e} | {status}")

        results.extend(stats)

    return results


def validate_onnx_vs_pytorch(onnx_path, num_tests=10, num_points=10000, tolerance=1e-3):
    """
    Validate ONNX model against PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        num_tests: Number of test cases
        num_points: Points per cloud
        tolerance: Maximum allowed error

    Returns:
        List of test statistics
    """
    print("\n" + "="*80)
    print("VALIDATING ONNX VS PYTORCH")
    print("="*80)

    # Check if ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"FAIL ONNX model not found: {onnx_path}")
        print("  Please export model first with: python export_to_onnx.py")
        return []

    # Load ONNX model
    try:
        import onnxruntime as ort
    except ImportError:
        print("FAIL ONNXRuntime not installed. Install with: pip install onnxruntime")
        return []

    ort_session = ort.InferenceSession(onnx_path)
    print(f"PASS Loaded ONNX model: {onnx_path}")

    # Create PyTorch model
    model = GraspNetONNXInference(
        num_seed=512,
        num_view=300,
        num_angle=48,
        num_depth=5,
        half_views=False
    )
    model.eval()
    print("PASS Created PyTorch model")

    results = []

    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")
        print("-" * 40)

        # Create test input
        xyz = create_test_point_cloud(num_points=num_points, seed=i)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(xyz)

        # ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: xyz.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)

        # Compare outputs
        stats = []
        stats.append(compare_tensors(
            pytorch_output[0], onnx_output[0], 'grasp_scores', tolerance=tolerance
        ))
        stats.append(compare_tensors(
            pytorch_output[1], onnx_output[1], 'grasp_widths', tolerance=tolerance
        ))
        stats.append(compare_tensors(
            pytorch_output[2], onnx_output[2], 'seed_xyz', tolerance=tolerance
        ))

        # Print results
        for stat in stats:
            status = "PASS" if stat['passed'] else "FAIL"
            print(f"  {stat['name']:20s} | Max: {stat['max_abs_error']:.2e} | Mean: {stat['mean_abs_error']:.2e} | {status}")

        results.extend(stats)

    return results


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if not results:
        print("No results to summarize.")
        return

    # Group by tensor name
    by_name = {}
    for r in results:
        name = r['name']
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(r)

    # Print statistics
    all_passed = True
    for name, stats in by_name.items():
        max_abs = max(s['max_abs_error'] for s in stats)
        mean_abs = np.mean([s['mean_abs_error'] for s in stats])
        passed_count = sum(1 for s in stats if s['passed'])
        total_count = len(stats)

        status = "PASS" if passed_count == total_count else "FAIL"
        print(f"\n{name}:")
        print(f"  Max absolute error:  {max_abs:.2e}")
        print(f"  Mean absolute error: {mean_abs:.2e}")
        print(f"  Tests passed:        {passed_count}/{total_count} {status}")

        if passed_count < total_count:
            all_passed = False

    # Final verdict
    print("\n" + "="*80)
    if all_passed:
        print("PASS ALL TESTS PASSED - Accuracy is maintained!")
    else:
        print("FAIL SOME TESTS FAILED - Review errors above")
    print("="*80)

    return all_passed


def save_results(results, output_path='VALIDATION_RESULTS.md'):
    """Save validation results to markdown file."""
    with open(output_path, 'w') as f:
        f.write("# Accuracy Validation Results\n\n")
        f.write("Comparison between PyTorch and ONNX implementations.\n\n")

        # Group by name
        by_name = {}
        for r in results:
            name = r['name']
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(r)

        # Write table
        f.write("## Summary\n\n")
        f.write("| Tensor | Max Abs Error | Mean Abs Error | Status |\n")
        f.write("|--------|---------------|----------------|--------|\n")

        for name, stats in by_name.items():
            max_abs = max(s['max_abs_error'] for s in stats)
            mean_abs = np.mean([s['mean_abs_error'] for s in stats])
            passed_count = sum(1 for s in stats if s['passed'])
            total_count = len(stats)
            status = "PASS" if passed_count == total_count else "FAIL"

            f.write(f"| {name} | {max_abs:.2e} | {mean_abs:.2e} | {status} |\n")

        f.write("\n## Detailed Results\n\n")
        for name, stats in by_name.items():
            f.write(f"### {name}\n\n")
            for i, s in enumerate(stats):
                f.write(f"Test {i+1}:\n")
                f.write(f"- Max absolute error: {s['max_abs_error']:.2e}\n")
                f.write(f"- Mean absolute error: {s['mean_abs_error']:.2e}\n")
                f.write(f"- Max relative error: {s['max_rel_error']:.2e}\n")
                f.write(f"- Status: {'PASS' if s['passed'] else 'FAIL'}\n\n")

    print(f"\nPASS Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate ONNX model accuracy')
    parser.add_argument('--onnx_model', type=str, default=None,
                       help='Path to ONNX model (optional, tests PyTorch if not provided)')
    parser.add_argument('--num_tests', type=int, default=10,
                       help='Number of test cases')
    parser.add_argument('--num_points', type=int, default=10000,
                       help='Number of points per test cloud')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help='Maximum allowed error')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to VALIDATION_RESULTS.md')

    args = parser.parse_args()

    print("="*80)
    print("ONNX MODEL ACCURACY VALIDATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Num tests: {args.num_tests}")
    print(f"  Points per cloud: {args.num_points}")
    print(f"  Tolerance: {args.tolerance}")

    # Run validation
    if args.onnx_model:
        # Validate ONNX vs PyTorch
        results = validate_onnx_vs_pytorch(
            args.onnx_model,
            num_tests=args.num_tests,
            num_points=args.num_points,
            tolerance=args.tolerance
        )
    else:
        # Just validate PyTorch consistency
        print("\nINFO No ONNX model provided, testing PyTorch consistency only")
        print("  To test ONNX: python validate_accuracy.py --onnx_model path/to/model.onnx")
        results = validate_pytorch_model(
            num_tests=args.num_tests,
            num_points=args.num_points
        )

    # Print summary
    all_passed = print_summary(results)

    # Save results
    if args.save_results and results:
        save_results(results)

    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
