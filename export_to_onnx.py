"""
Export GraspNet model to ONNX format

Converts the CUDA/MinkowskiEngine-based model to ONNX-compatible format
using PyTorch standard operations.
"""

import argparse
import os
import sys
import torch
import onnx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from models.graspnet_onnx import GraspNetONNXInference


def create_dummy_input(batch_size=1, num_points=20000):
    """
    Create dummy input for ONNX export.

    Args:
        batch_size: Batch size
        num_points: Number of points in point cloud

    Returns:
        xyz: (B, N, 3) point cloud tensor
    """
    xyz = torch.randn(batch_size, num_points, 3) * 0.3
    return xyz


def export_to_onnx(model, save_path, input_shape=(1, 20000, 3), opset_version=11):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        save_path: Output path for ONNX model
        input_shape: Input tensor shape (B, N, 3)
        opset_version: ONNX opset version (default: 11)

    Returns:
        ONNX model object
    """
    model.eval()

    # Create dummy input
    batch_size, num_points, _ = input_shape
    dummy_input = create_dummy_input(batch_size, num_points)

    print(f"Exporting model to ONNX...")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    print(f"  Output: {save_path}")

    # Export to ONNX
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
            }
        )

    print(f"✓ Export successful")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("✓ Verification passed")

    # Print model info
    file_size_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f"\nModel Info:")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Graph nodes: {len(onnx_model.graph.node)}")

    return onnx_model


def main():
    parser = argparse.ArgumentParser(description='Export GraspNet to ONNX')
    parser.add_argument('--output', type=str, default='graspnet.onnx',
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
                       help='Number of input points')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')

    args = parser.parse_args()

    print("="*70)
    print("GRASPNET ONNX EXPORT")
    print("="*70)
    print(f"Model Configuration:")
    print(f"  Type: {'Lightweight (half-views)' if args.half_views else 'Full'}")
    print(f"  Seed points: {args.num_seed}")
    print(f"  View templates: {args.num_view}")
    print(f"  Angles: {args.num_angle}")
    print(f"  Depths: {args.num_depth}")
    print(f"  Input: ({args.batch_size}, {args.input_points}, 3)")

    # Create model
    print("\nCreating ONNX-compatible model...")
    model = GraspNetONNXInference(
        num_seed=args.num_seed,
        num_view=args.num_view,
        num_angle=args.num_angle,
        num_depth=args.num_depth,
        half_views=args.half_views
    )
    print("✓ Model created")

    # Export
    input_shape = (args.batch_size, args.input_points, 3)
    export_to_onnx(
        model,
        args.output,
        input_shape=input_shape,
        opset_version=args.opset_version
    )

    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Validate accuracy:")
    print(f"     python validate_accuracy.py --onnx_model {args.output}")
    print(f"  2. Convert to QNN:")
    print(f"     python qnn/convert_to_qnn.py --onnx_model {args.output}")


if __name__ == '__main__':
    main()
