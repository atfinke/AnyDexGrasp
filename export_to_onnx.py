"""
Export GraspNet model to ONNX format
Converts MinkowskiEngine-based model to ONNX-compatible format
"""

import argparse
import os
import sys
import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from models.graspnet_onnx import GraspNetONNX, GraspNetONNXInference


def create_dummy_input(batch_size=1, num_points=20000):
    """
    Create dummy input for testing ONNX export.

    Args:
        batch_size: batch size
        num_points: number of points in point cloud

    Returns:
        xyz: (B, N, 3) dummy point cloud
    """
    # Create random point cloud
    xyz = torch.randn(batch_size, num_points, 3)
    # Normalize to reasonable range
    xyz = xyz * 0.3  # Scale to ~[-0.9, 0.9] range
    return xyz


def export_model_to_onnx(model, save_path, input_shape=(1, 20000, 3),
                         opset_version=11, verbose=True):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        save_path: path to save ONNX model
        input_shape: input shape (B, N, 3)
        opset_version: ONNX opset version
        verbose: print export info
    """
    model.eval()

    # Create dummy input
    batch_size, num_points, _ = input_shape
    dummy_input = create_dummy_input(batch_size, num_points)

    # Export to ONNX
    print(f"Exporting model to ONNX format...")
    print(f"Input shape: {input_shape}")
    print(f"Output path: {save_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['point_cloud'],
            output_names=['grasp_scores', 'grasp_widths', 'seed_xyz'],
            dynamic_axes={
                'point_cloud': {0: 'batch_size', 1: 'num_points'},
                'grasp_scores': {0: 'batch_size'},
                'grasp_widths': {0: 'batch_size'},
                'seed_xyz': {0: 'batch_size'}
            },
            verbose=verbose
        )

    print(f"✓ Model exported successfully to {save_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")

    return onnx_model


def validate_onnx_model(pytorch_model, onnx_path, num_tests=5, tolerance=1e-3):
    """
    Validate ONNX model against PyTorch model.

    Args:
        pytorch_model: original PyTorch model
        onnx_path: path to ONNX model
        num_tests: number of random inputs to test
        tolerance: maximum allowed difference
    """
    print("\n" + "="*80)
    print("VALIDATING ONNX MODEL ACCURACY")
    print("="*80)

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    pytorch_model.eval()

    errors = []
    for i in range(num_tests):
        # Create random input
        xyz = create_dummy_input(batch_size=1, num_points=10000)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(xyz)

        # ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: xyz.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Compare outputs
        pytorch_scores = pytorch_output[0].numpy()
        pytorch_widths = pytorch_output[1].numpy()
        pytorch_xyz = pytorch_output[2].numpy()

        onnx_scores = ort_outputs[0]
        onnx_widths = ort_outputs[1]
        onnx_xyz = ort_outputs[2]

        # Compute errors
        score_error = np.abs(pytorch_scores - onnx_scores).max()
        width_error = np.abs(pytorch_widths - onnx_widths).max()
        xyz_error = np.abs(pytorch_xyz - onnx_xyz).max()

        errors.append({
            'score_error': score_error,
            'width_error': width_error,
            'xyz_error': xyz_error
        })

        print(f"\nTest {i+1}/{num_tests}:")
        print(f"  Max grasp score error: {score_error:.6f}")
        print(f"  Max grasp width error: {width_error:.6f}")
        print(f"  Max seed xyz error: {xyz_error:.6f}")

        if max(score_error, width_error, xyz_error) > tolerance:
            print(f"  ⚠ Warning: Error exceeds tolerance ({tolerance})")
        else:
            print(f"  ✓ Passed (error < {tolerance})")

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    avg_score_error = np.mean([e['score_error'] for e in errors])
    avg_width_error = np.mean([e['width_error'] for e in errors])
    avg_xyz_error = np.mean([e['xyz_error'] for e in errors])
    max_score_error = np.max([e['score_error'] for e in errors])
    max_width_error = np.max([e['width_error'] for e in errors])
    max_xyz_error = np.max([e['xyz_error'] for e in errors])

    print(f"Average errors:")
    print(f"  Grasp scores: {avg_score_error:.6f}")
    print(f"  Grasp widths: {avg_width_error:.6f}")
    print(f"  Seed xyz: {avg_xyz_error:.6f}")
    print(f"\nMaximum errors:")
    print(f"  Grasp scores: {max_score_error:.6f}")
    print(f"  Grasp widths: {max_width_error:.6f}")
    print(f"  Seed xyz: {max_xyz_error:.6f}")

    if max(max_score_error, max_width_error, max_xyz_error) < tolerance:
        print(f"\n✓ All tests passed! ONNX model is accurate.")
        return True
    else:
        print(f"\n⚠ Some tests failed. Please review the errors above.")
        return False


def load_checkpoint_weights(pytorch_model, checkpoint_path):
    """
    Load weights from original MinkowskiEngine checkpoint.
    Note: This is a placeholder - actual weight conversion may be needed.

    Args:
        pytorch_model: ONNX-compatible PyTorch model
        checkpoint_path: path to original checkpoint
    """
    print(f"\nLoading checkpoint from {checkpoint_path}...")

    if not os.path.exists(checkpoint_path):
        print(f"⚠ Warning: Checkpoint file not found: {checkpoint_path}")
        print("  Proceeding with randomly initialized weights.")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Attempt to load weights
        # Note: This may require custom mapping due to architecture differences
        try:
            pytorch_model.load_state_dict(state_dict, strict=False)
            print("✓ Weights loaded successfully (some layers may be unmatched)")
        except Exception as e:
            print(f"⚠ Warning: Could not load weights directly: {e}")
            print("  Attempting partial weight loading...")

            # Try loading matching layers only
            model_dict = pytorch_model.state_dict()
            matched_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    matched_dict[k] = v

            model_dict.update(matched_dict)
            pytorch_model.load_state_dict(model_dict)
            print(f"✓ Loaded {len(matched_dict)}/{len(state_dict)} matching layers")

        return True

    except Exception as e:
        print(f"⚠ Error loading checkpoint: {e}")
        print("  Proceeding with randomly initialized weights.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export GraspNet to ONNX')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='graspnet_onnx.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--num_seed', type=int, default=1024,
                       help='Number of seed points')
    parser.add_argument('--num_view', type=int, default=300,
                       help='Number of view templates')
    parser.add_argument('--num_angle', type=int, default=48,
                       help='Number of grasp angles')
    parser.add_argument('--num_depth', type=int, default=5,
                       help='Number of grasp depths')
    parser.add_argument('--half_views', action='store_true',
                       help='Use half views (lightweight model)')
    parser.add_argument('--input_points', type=int, default=20000,
                       help='Number of input points for export')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--validate', action='store_true',
                       help='Validate ONNX model against PyTorch')
    parser.add_argument('--num_tests', type=int, default=5,
                       help='Number of validation tests')

    args = parser.parse_args()

    print("="*80)
    print("GRASPNET ONNX EXPORT")
    print("="*80)
    print(f"Configuration:")
    print(f"  Model type: {'Half-views (lightweight)' if args.half_views else 'Full'}")
    print(f"  Num seed points: {args.num_seed}")
    print(f"  Num view templates: {args.num_view}")
    print(f"  Num angles: {args.num_angle}")
    print(f"  Num depths: {args.num_depth}")
    print(f"  Input shape: ({args.batch_size}, {args.input_points}, 3)")
    print(f"  ONNX opset version: {args.opset_version}")

    # Create ONNX-compatible model
    print("\nCreating ONNX-compatible model...")
    model = GraspNetONNXInference(
        num_seed=args.num_seed,
        num_view=args.num_view,
        num_angle=args.num_angle,
        num_depth=args.num_depth,
        half_views=args.half_views
    )
    print("✓ Model created successfully")

    # Load checkpoint weights if provided
    if args.checkpoint:
        load_checkpoint_weights(model, args.checkpoint)

    # Export to ONNX
    input_shape = (args.batch_size, args.input_points, 3)
    onnx_model = export_model_to_onnx(
        model, args.output, input_shape=input_shape,
        opset_version=args.opset_version, verbose=False
    )

    # Print model info
    print(f"\nONNX Model Info:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
    print(f"  Graph nodes: {len(onnx_model.graph.node)}")
    print(f"  File size: {os.path.getsize(args.output) / (1024**2):.2f} MB")

    # Validate if requested
    if args.validate:
        try:
            validate_onnx_model(model, args.output, num_tests=args.num_tests)
        except Exception as e:
            print(f"\n⚠ Validation failed: {e}")
            print("  This may be due to missing ONNX Runtime. Install with:")
            print("    pip install onnxruntime")

    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"ONNX model saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Test the ONNX model with your inference pipeline")
    print("  2. Convert to QNN format using Qualcomm Neural Network SDK")
    print("  3. Optimize and quantize for Hexagon NPU deployment")


if __name__ == '__main__':
    main()
