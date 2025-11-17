"""
Comprehensive accuracy validation: PyTorch vs ONNX Runtime
"""
import sys
import os
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from graspnet_onnx import GraspNetONNXInference

print("="*70)
print("ACCURACY VALIDATION: PyTorch vs ONNX Runtime")
print("="*70)

# Check dependencies
try:
    import onnx
    import onnxruntime as ort
    print(f"PyTorch: {torch.__version__}")
    print(f"ONNX: {onnx.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

# Configuration
config = {
    'num_seed': 64,       # Smaller for faster export
    'num_view': 300,
    'num_angle': 48,
    'num_depth': 5,
    'half_views': False
}

print(f"\nModel Configuration:")
for key, val in config.items():
    print(f"  {key}: {val}")

# Create model
print("\n" + "="*70)
print("Step 1: Creating PyTorch Model")
print("="*70)

model = GraspNetONNXInference(**config).eval()
print("✓ Model created")

# Test inputs
num_tests = 5
batch_size = 1
num_points = 256  # Small for fast testing

print(f"\nTest Configuration:")
print(f"  Number of tests: {num_tests}")
print(f"  Batch size: {batch_size}")
print(f"  Points per cloud: {num_points}")

# Export to ONNX
onnx_path = '/tmp/graspnet_validate.onnx'

print("\n" + "="*70)
print("Step 2: Exporting to ONNX")
print("="*70)
print(f"Output: {onnx_path}")

# Create sample input for export
sample_xyz = torch.randn(batch_size, num_points, 3) * 0.3

try:
    torch.onnx.export(
        model,
        sample_xyz,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['point_cloud'],
        output_names=['grasp_scores', 'grasp_widths', 'seed_xyz'],
        dynamic_axes={
            'point_cloud': {0: 'batch'},
            'grasp_scores': {0: 'batch'},
            'grasp_widths': {0: 'batch'},
            'seed_xyz': {0: 'batch'}
        },
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )
    print("✓ ONNX export successful")

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model validated")
    print(f"  Opset: {onnx_model.opset_import[0].version}")
    print(f"  Nodes: {len(onnx_model.graph.node)}")

    # Get file size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

except Exception as e:
    print(f"✗ ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load ONNX Runtime session
print("\n" + "="*70)
print("Step 3: Loading ONNX Runtime")
print("="*70)

try:
    session = ort.InferenceSession(onnx_path)
    print("✓ ONNX Runtime session created")
    print(f"  Providers: {session.get_providers()}")
except Exception as e:
    print(f"✗ Failed to create session: {e}")
    sys.exit(1)

# Run accuracy tests
print("\n" + "="*70)
print("Step 4: Running Accuracy Tests")
print("="*70)

results = {
    'grasp_scores': [],
    'grasp_widths': [],
    'seed_xyz': []
}

torch.manual_seed(42)
np.random.seed(42)

for i in range(num_tests):
    print(f"\nTest {i+1}/{num_tests}")
    print("-" * 40)

    # Generate random point cloud
    xyz = torch.randn(batch_size, num_points, 3) * 0.3

    # PyTorch inference
    with torch.no_grad():
        pt_scores, pt_widths, pt_seed_xyz = model(xyz)

    # ONNX Runtime inference
    ort_inputs = {'point_cloud': xyz.numpy()}
    ort_scores, ort_widths, ort_seed_xyz = session.run(None, ort_inputs)

    # Convert to numpy for comparison
    pt_scores_np = pt_scores.numpy()
    pt_widths_np = pt_widths.numpy()
    pt_seed_xyz_np = pt_seed_xyz.numpy()

    # Compute differences
    diff_scores = np.abs(pt_scores_np - ort_scores)
    diff_widths = np.abs(pt_widths_np - ort_widths)
    diff_seed_xyz = np.abs(pt_seed_xyz_np - ort_seed_xyz)

    # Statistics
    stats_scores = {
        'max': diff_scores.max(),
        'mean': diff_scores.mean(),
        'std': diff_scores.std()
    }
    stats_widths = {
        'max': diff_widths.max(),
        'mean': diff_widths.mean(),
        'std': diff_widths.std()
    }
    stats_seed_xyz = {
        'max': diff_seed_xyz.max(),
        'mean': diff_seed_xyz.mean(),
        'std': diff_seed_xyz.std()
    }

    results['grasp_scores'].append(stats_scores)
    results['grasp_widths'].append(stats_widths)
    results['seed_xyz'].append(stats_seed_xyz)

    print(f"  Grasp Scores:")
    print(f"    Max diff:  {stats_scores['max']:.2e}")
    print(f"    Mean diff: {stats_scores['mean']:.2e}")
    print(f"  Grasp Widths:")
    print(f"    Max diff:  {stats_widths['max']:.2e}")
    print(f"    Mean diff: {stats_widths['mean']:.2e}")
    print(f"  Seed XYZ:")
    print(f"    Max diff:  {stats_seed_xyz['max']:.2e}")
    print(f"    Mean diff: {stats_seed_xyz['mean']:.2e}")

# Summary statistics
print("\n" + "="*70)
print("ACCURACY SUMMARY")
print("="*70)

def summarize_results(name, data):
    max_diffs = [d['max'] for d in data]
    mean_diffs = [d['mean'] for d in data]

    overall_max = max(max_diffs)
    overall_mean = np.mean(mean_diffs)

    print(f"\n{name}:")
    print(f"  Overall max difference:  {overall_max:.2e}")
    print(f"  Average mean difference: {overall_mean:.2e}")

    if overall_max < 1e-6:
        status = "✓ EXCELLENT (< 1e-6)"
    elif overall_max < 1e-4:
        status = "✓ GOOD (< 1e-4)"
    elif overall_max < 1e-2:
        status = "⚠ ACCEPTABLE (< 1e-2)"
    else:
        status = "✗ POOR (>= 1e-2)"

    print(f"  Status: {status}")
    return overall_max

max_score_diff = summarize_results("Grasp Scores", results['grasp_scores'])
max_width_diff = summarize_results("Grasp Widths", results['grasp_widths'])
max_xyz_diff = summarize_results("Seed XYZ", results['seed_xyz'])

# Final verdict
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

overall_max = max(max_score_diff, max_width_diff, max_xyz_diff)

print(f"\nOverall maximum difference: {overall_max:.2e}")

if overall_max < 1e-6:
    verdict = "✓ PASS - EXCELLENT ACCURACY"
    print(f"\n{verdict}")
    print("ONNX model matches PyTorch within floating-point precision.")
    exit_code = 0
elif overall_max < 1e-4:
    verdict = "✓ PASS - GOOD ACCURACY"
    print(f"\n{verdict}")
    print("ONNX model matches PyTorch with acceptable accuracy.")
    exit_code = 0
elif overall_max < 1e-2:
    verdict = "⚠ MARGINAL - ACCEPTABLE FOR SOME APPLICATIONS"
    print(f"\n{verdict}")
    print("ONNX model has noticeable differences. Review use case requirements.")
    exit_code = 0
else:
    verdict = "✗ FAIL - POOR ACCURACY"
    print(f"\n{verdict}")
    print("ONNX model does not match PyTorch. Investigation required.")
    exit_code = 1

print("\n" + "="*70)
print("QNN CONVERSION READY")
print("="*70)

if exit_code == 0:
    print("\nThe ONNX model is ready for QNN conversion:")
    print()
    print("source /tmp/qnn_env/bin/activate")
    print("export PYTHONPATH=/tmp/qairt/2.35.0.250530/lib/python:${PYTHONPATH}")
    print("export LD_LIBRARY_PATH=/tmp/qairt/2.35.0.250530/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}")
    print()
    print(f"python3 /tmp/qairt/2.35.0.250530/bin/x86_64-linux-clang/qnn-onnx-converter \\")
    print(f"  --input_network {onnx_path} \\")
    print(f"  --output_path /tmp/graspnet_qnn.cpp \\")
    print(f"  -d 'point_cloud' {batch_size},{num_points},3")
    print()

sys.exit(exit_code)
