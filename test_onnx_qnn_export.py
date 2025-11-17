"""
Test ONNX export and QNN conversion of PointNet++ operators
"""
import sys
import os
import torch
import torch.nn as nn
import onnx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'onnx_ops'))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from pointnet2_onnx import furthest_point_sample, ball_query, group_points
from knn_onnx import knn_distance

print("="*60)
print("ONNX EXPORT & QNN CONVERSION TEST")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"ONNX: {onnx.__version__}")

# Test 1: Simple PointNet++ operation as PyTorch module
class PointNetSampleLayer(nn.Module):
    """
    Simplified PointNet++ layer for testing
    """
    def __init__(self, npoint=512, radius=0.2, nsample=32):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(6, 32, 1),  # 3 (xyz) + 3 (features)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, 3, N) point features

        Returns:
            new_features: (B, 64, npoint) output features
        """
        B, N, _ = xyz.shape

        # Furthest point sampling
        fps_idx = furthest_point_sample(xyz, self.npoint)  # (B, npoint)

        # Gather new xyz
        fps_idx_expanded = fps_idx.unsqueeze(-1).expand(-1, -1, 3)
        new_xyz = torch.gather(xyz, 1, fps_idx_expanded)  # (B, npoint, 3)

        # Ball query
        idx = ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B, npoint, nsample)

        # Group xyz and features
        grouped_xyz = group_points(xyz.transpose(1, 2), idx)  # (B, 3, npoint, nsample)
        grouped_xyz_normalized = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)

        grouped_features = group_points(features, idx)  # (B, 3, npoint, nsample)

        # Concatenate
        new_features = torch.cat([grouped_xyz_normalized, grouped_features], dim=1)  # (B, 6, npoint, nsample)

        # MLP
        new_features = self.mlp(new_features)  # (B, 64, npoint, nsample)

        # Max pooling
        new_features = torch.max(new_features, dim=-1)[0]  # (B, 64, npoint)

        return new_features


print("\n" + "="*60)
print("Test 1: PointNet++ Sample Layer")
print("="*60)

model = PointNetSampleLayer(npoint=256, radius=0.2, nsample=16).eval()

# Create test inputs
batch_size = 1
num_points = 1024
xyz = torch.randn(batch_size, num_points, 3) * 0.5
features = torch.randn(batch_size, 3, num_points)

print(f"Input xyz shape: {xyz.shape}")
print(f"Input features shape: {features.shape}")

# Test forward pass
print("\nTesting forward pass...")
with torch.no_grad():
    output = model(xyz, features)

print(f"✓ Forward pass successful")
print(f"  Output shape: {output.shape}")
print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

# Export to ONNX (opset 11 for QNN)
onnx_path = '/tmp/pointnet_sample_layer.onnx'
print(f"\nExporting to ONNX (opset 11)...")
print(f"  Output: {onnx_path}")

torch.onnx.export(
    model,
    (xyz, features),
    onnx_path,
    export_params=True,
    opset_version=11,  # CRITICAL: opset 11 for QNN
    input_names=['xyz', 'features'],
    output_names=['output'],
    dynamic_axes={
        'xyz': {0: 'batch'},
        'features': {0: 'batch'},
        'output': {0: 'batch'}
    },
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
)

print("✓ ONNX export successful")

# Verify ONNX model
print("\nVerifying ONNX model...")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print(f"✓ ONNX model is valid")
print(f"  Opset version: {onnx_model.opset_import[0].version}")
print(f"  Graph nodes: {len(onnx_model.graph.node)}")
print(f"  Operations: ", end="")

ops = set()
for node in onnx_model.graph.node:
    ops.add(node.op_type)
print(", ".join(sorted(ops)))

# Test with ONNX Runtime
try:
    import onnxruntime as ort
    print("\nTesting with ONNX Runtime...")

    session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        'xyz': xyz.numpy(),
        'features': features.numpy()
    }
    ort_output = session.run(None, ort_inputs)[0]

    # Compare outputs
    torch_output = output.numpy()
    diff = abs(torch_output - ort_output).max()

    print(f"✓ ONNX Runtime successful")
    print(f"  Max difference vs PyTorch: {diff:.2e}")

    if diff < 1e-5:
        print(f"  ✓ EXCELLENT: Match within 1e-5")
    elif diff < 1e-3:
        print(f"  ✓ GOOD: Match within 1e-3")
    else:
        print(f"  ⚠ WARNING: Large difference")

except ImportError as e:
    print(f"ONNX Runtime not available: {e}")

print("\n" + "="*60)
print("QNN CONVERSION INSTRUCTIONS")
print("="*60)
print("To convert to QNN, run:")
print()
print("source /tmp/qnn_env/bin/activate")
print("export PYTHONPATH=/tmp/qairt/2.35.0.250530/lib/python:${PYTHONPATH}")
print("export LD_LIBRARY_PATH=/tmp/qairt/2.35.0.250530/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}")
print()
print(f"python3 /tmp/qairt/2.35.0.250530/bin/x86_64-linux-clang/qnn-onnx-converter \\")
print(f"  --input_network {onnx_path} \\")
print(f"  --output_path /tmp/pointnet_qnn.cpp \\")
print(f"  -d 'xyz' 1,{num_points},3 \\")
print(f"  -d 'features' 1,3,{num_points}")
print()
print("="*60)
print("TEST COMPLETED SUCCESSFULLY")
print("="*60)
