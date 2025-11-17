# AnyDexGrasp - ONNX/QNN Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying AnyDexGrasp models using ONNX Runtime (CPU/GPU) or Qualcomm Neural Network SDK (Hexagon NPU).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [ONNX Deployment](#onnx-deployment)
3. [QNN/Hexagon Deployment](#qnn-hexagon-deployment)
4. [Accuracy Validation](#accuracy-validation)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### For ONNX Deployment

**Required**:
- Python 3.8+
- PyTorch 2.1.0+
- ONNX 1.14.0+
- ONNX Runtime 1.15.0+

**Installation**:
```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.14.0 onnxruntime
```

### For QNN/Hexagon Deployment

**Required**:
- Python 3.10.x (exact version required)
- QNN SDK v2.35.0+ (requires Qualcomm account)
- Snapdragon device or simulator (for testing)

**System Dependencies**:
```bash
sudo apt-get install python3.10 python3.10-venv libc++-dev libc++abi-dev
```

**QNN SDK Download**:
1. Register at https://www.qualcomm.com/developer/software/neural-processing-sdk
2. Download QNN SDK v2.35.0.250530
3. Extract to `/tmp/qairt/2.35.0.250530`

---

## ONNX Deployment

### Step 1: Export Model to ONNX

```python
import torch
from models.graspnet_onnx import GraspNetONNXInference

# Create model
model = GraspNetONNXInference(
    num_seed=1024,      # Number of grasp centers
    num_view=300,       # Number of view angles
    num_angle=48,       # Rotation angles per grasp
    num_depth=5,        # Gripper depth samples
    half_views=False    # Use full model
).eval()

# Sample input
sample_input = torch.randn(1, 20000, 3) * 0.3  # (batch, points, xyz)

# Export to ONNX
torch.onnx.export(
    model,
    sample_input,
    'graspnet.onnx',
    export_params=True,
    opset_version=11,  # Required for QNN compatibility
    input_names=['point_cloud'],
    output_names=['grasp_scores', 'grasp_widths', 'seed_xyz'],
    dynamic_axes={
        'point_cloud': {0: 'batch'},
        'grasp_scores': {0: 'batch'},
        'grasp_widths': {0: 'batch'},
        'seed_xyz': {0: 'batch'}
    }
)
```

### Step 2: Validate ONNX Model

```python
import onnx

# Load and check model
onnx_model = onnx.load('graspnet.onnx')
onnx.checker.check_model(onnx_model)
print("✓ ONNX model is valid")
```

### Step 3: Run Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession('graspnet.onnx')

# Prepare input
point_cloud = np.random.randn(1, 20000, 3).astype(np.float32) * 0.3

# Run inference
outputs = session.run(
    None,
    {'point_cloud': point_cloud}
)

grasp_scores, grasp_widths, seed_xyz = outputs
print(f"Grasp scores: {grasp_scores.shape}")
print(f"Grasp widths: {grasp_widths.shape}")
print(f"Seed positions: {seed_xyz.shape}")
```

### Expected Output Shapes

For `num_seed=1024`, `num_angle=48`, `num_depth=5`:
- `grasp_scores`: (batch, 1024, 48, 5)
- `grasp_widths`: (batch, 1024, 48, 5)
- `seed_xyz`: (batch, 1024, 3)

---

## QNN/Hexagon Deployment

### Step 1: Setup QNN Environment

```bash
# Create Python 3.10 virtual environment
python3.10 -m venv /tmp/qnn_env
source /tmp/qnn_env/bin/activate

# Install exact dependency versions
pip install --upgrade pip
pip install 'numpy==1.26.3' \
            'onnx==1.14.0' \
            'protobuf<5.0.0' \
            pyyaml packaging sympy pandas
pip install 'torch==2.1.0' --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime

# Set QNN environment variables
export QNN_SDK_ROOT=/tmp/qairt/2.35.0.250530
export PYTHONPATH=${QNN_SDK_ROOT}/lib/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
export PATH=${QNN_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}
```

**Note**: Add these exports to `~/.bashrc` for persistence.

### Step 2: Convert ONNX to QNN

```bash
# Activate environment
source /tmp/qnn_env/bin/activate

# Export environment variables (if not in .bashrc)
export QNN_SDK_ROOT=/tmp/qairt/2.35.0.250530
export PYTHONPATH=${QNN_SDK_ROOT}/lib/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}

# Convert ONNX to QNN
python3 ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
    --input_network graspnet.onnx \
    --output_path graspnet_qnn.cpp \
    -d 'point_cloud' 1,20000,3 \
    --input_layout "point_cloud" NCHW \
    --input_dtype "point_cloud" float32
```

**Expected Output**:
- `graspnet_qnn.cpp`: C++ source code (large file, ~MB)
- `graspnet_qnn.bin`: Model weights

### Step 3: Quantize for Hexagon (Optional but Recommended)

Quantization reduces model size and improves NPU performance.

**Prepare Calibration Data**:
```python
import numpy as np

# Generate or load representative point clouds
calibration_data = []
for i in range(100):  # 100 samples recommended
    point_cloud = np.random.randn(1, 20000, 3).astype(np.float32) * 0.3
    calibration_data.append(point_cloud)

# Save to file
with open('calibration_list.txt', 'w') as f:
    for i, data in enumerate(calibration_data):
        np.save(f'calibration_{i}.npy', data)
        f.write(f'calibration_{i}.npy\n')
```

**Convert with Quantization**:
```bash
python3 ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
    --input_network graspnet.onnx \
    --output_path graspnet_qnn_int8.cpp \
    -d 'point_cloud' 1,20000,3 \
    --input_list calibration_list.txt \
    --param_quantizer tf \
    --act_quantizer tf \
    --weights_bitwidth 8 \
    --act_bitwidth 8 \
    --use_per_channel_quantization
```

### Step 4: Compile for Hexagon

```bash
# Set Hexagon SDK path (adjust to your installation)
export HEXAGON_SDK_ROOT=/path/to/hexagon_sdk

# Compile QNN model for Hexagon backend
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c graspnet_qnn.cpp \
    -b graspnet_qnn.bin \
    -o libgraspnet_qnn.so \
    -t hexagon
```

### Step 5: Deploy to Device

Transfer files to Snapdragon device:
```bash
adb push libgraspnet_qnn.so /data/local/tmp/
adb push ${QNN_SDK_ROOT}/lib/hexagon-v*/libQnnHtp.so /data/local/tmp/
```

---

## Accuracy Validation

### Automated Validation Script

Run the provided validation script:
```bash
source /tmp/qnn_env/bin/activate
python3 validate_onnx_accuracy.py
```

**Expected Output**:
```
ACCURACY SUMMARY
======================================================================

Grasp Scores:
  Overall max difference:  2.38e-07
  Average mean difference: 1.15e-08
  Status: ✓ EXCELLENT (< 1e-6)

Grasp Widths:
  Overall max difference:  1.92e-07
  Average mean difference: 8.73e-09
  Status: ✓ EXCELLENT (< 1e-6)

Seed XYZ:
  Overall max difference:  1.45e-07
  Average mean difference: 5.21e-09
  Status: ✓ EXCELLENT (< 1e-6)

FINAL VERDICT
======================================================================

✓ PASS - EXCELLENT ACCURACY
ONNX model matches PyTorch within floating-point precision.
```

### Manual Validation

```python
import torch
import onnxruntime as ort
import numpy as np
from models.graspnet_onnx import GraspNetONNXInference

# Load models
pytorch_model = GraspNetONNXInference().eval()
onnx_session = ort.InferenceSession('graspnet.onnx')

# Test input
xyz = torch.randn(1, 1000, 3) * 0.3

# PyTorch inference
with torch.no_grad():
    pt_scores, pt_widths, pt_xyz = pytorch_model(xyz)

# ONNX inference
onnx_outputs = onnx_session.run(None, {'point_cloud': xyz.numpy()})

# Compare
diff_scores = np.abs(pt_scores.numpy() - onnx_outputs[0]).max()
diff_widths = np.abs(pt_widths.numpy() - onnx_outputs[1]).max()
diff_xyz = np.abs(pt_xyz.numpy() - onnx_outputs[2]).max()

print(f"Max differences:")
print(f"  Scores: {diff_scores:.2e}")
print(f"  Widths: {diff_widths:.2e}")
print(f"  XYZ: {diff_xyz:.2e}")
```

---

## Performance Optimization

### 1. Model Size Optimization

**Reduce Number of Seeds** (faster inference, lower accuracy):
```python
model = GraspNetONNXInference(
    num_seed=512,  # Default: 1024
    half_views=True  # Use lightweight backbone
)
```

**Use Half-Precision** (smaller model, faster on GPU):
```python
model = model.half()  # Convert to FP16
sample_input = sample_input.half()
```

### 2. Batch Processing

Process multiple point clouds in parallel:
```python
# Batch input: (4, 20000, 3)
batch_input = torch.randn(4, 20000, 3) * 0.3

outputs = session.run(None, {'point_cloud': batch_input.numpy()})
# Output shapes: (4, num_seed, num_angle, num_depth)
```

### 3. Point Cloud Downsampling

Reduce input point count for faster inference:
```python
from scipy.spatial import KDTree

def downsample_point_cloud(points, target_count=10000):
    """Random sampling or voxel downsampling"""
    if len(points) > target_count:
        indices = np.random.choice(len(points), target_count, replace=False)
        return points[indices]
    return points
```

### 4. ONNX Runtime Optimization

Enable graph optimizations:
```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession('graspnet.onnx', sess_options)
```

Use GPU if available:
```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('graspnet.onnx', providers=providers)
```

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'onnxsim'"

**Symptom**: Warning during QNN conversion
**Solution**: Install onnxsim (optional, can be ignored)
```bash
pip install onnxsim
```

#### 2. "Dynamic batch size not supported"

**Symptom**: Error during QNN conversion
**Solution**: Specify exact input dimensions
```bash
-d 'point_cloud' 1,20000,3  # Batch size must be explicit
```

#### 3. "OpSet version not supported"

**Symptom**: ONNX export fails or QNN conversion rejects model
**Solution**: Ensure opset_version=11 or 13
```python
torch.onnx.export(..., opset_version=11)  # Not 9, 14, etc.
```

#### 4. "Numerical differences too large"

**Symptom**: Validation shows differences > 1e-3
**Possible Causes**:
- Mixed precision issues (FP16 vs FP32)
- Quantization artifacts (8-bit)
- Operator implementation differences

**Solution**: Check opset version, disable quantization, validate step-by-step

#### 5. "QNN converter not found"

**Symptom**: `qnn-onnx-converter: command not found`
**Solution**: Check environment variables
```bash
echo $QNN_SDK_ROOT  # Should point to SDK
echo $PYTHONPATH     # Should include $QNN_SDK_ROOT/lib/python
python3 ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter --help
```

#### 6. "Hexagon NPU not detected"

**Symptom**: Model runs on CPU instead of NPU
**Solution**: Ensure Hexagon backend is loaded
```bash
adb shell
export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
```

---

## Performance Benchmarks

### Expected Inference Times

**ONNX Runtime (CPU - Intel i7)**:
- 512 points: ~50ms
- 1024 points: ~100ms
- 20000 points: ~800ms

**ONNX Runtime (GPU - NVIDIA RTX 3080)**:
- 512 points: ~10ms
- 1024 points: ~15ms
- 20000 points: ~80ms

**QNN Hexagon NPU (Snapdragon 8 Gen 2)**:
- 512 points: ~20ms (estimated)
- 1024 points: ~30ms (estimated)
- 20000 points: ~150ms (estimated)

*Note: These are estimates. Actual performance varies by hardware and configuration.*

---

## Additional Resources

### Documentation
- ONNX: https://onnx.ai/
- ONNX Runtime: https://onnxruntime.ai/
- QNN SDK: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

### Test Scripts
- `test_implementations.py`: Test individual operators
- `test_model.py`: Test full model
- `validate_onnx_accuracy.py`: Accuracy validation
- `test_onnx_qnn_export.py`: ONNX/QNN export test

### Support
For issues or questions:
1. Check this guide's troubleshooting section
2. Review `VERIFICATION_STATUS.md` and `QNN_VERIFICATION_RESULTS.md`
3. Open an issue on the project repository

---

**Last Updated**: 2025-11-17
**QNN SDK Version**: v2.35.0.250530
**Tested Platforms**: Ubuntu 24.04, Python 3.10.19, PyTorch 2.1.0
