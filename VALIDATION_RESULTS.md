# Accuracy Validation Results

This document describes the validation methodology for comparing PyTorch and ONNX implementations.

## Test Configuration

- **Test cases**: Variable (user-configurable)
- **Points per cloud**: 10,000 (default)
- **Tolerance**: 1e-3 (max absolute error)
- **Model config**: 512 seeds, 300 views, 48 angles, 5 depths

## Expected Results

Based on the implementation using standard PyTorch operations:

| Tensor | Expected Max Error | Expected Mean Error | Target |
|--------|-------------------|---------------------|--------|
| grasp_scores | < 1e-5 | < 1e-7 | Pass if < 1e-3 |
| grasp_widths | < 1e-5 | < 1e-7 | Pass if < 1e-3 |
| seed_xyz | < 1e-6 | < 1e-8 | Pass if < 1e-3 |

## Validation Method

The validation script compares:

1. **PyTorch Model Self-Consistency**: Run model twice with same input, verify outputs match
2. **ONNX vs PyTorch**: Compare ONNX model outputs against PyTorch outputs

### Operators Validated

All re-implemented operators are tested:

- **KNN Distance**: `torch.cdist` + `torch.topk` (deterministic)
- **Ball Query**: Radius search with masking (deterministic)
- **Furthest Point Sampling**: Iterative greedy sampling (deterministic)
- **Group Points**: `torch.gather` operations (deterministic)
- **Three NN Interpolate**: 3-nearest neighbor interpolation (deterministic)
- **Cylinder Query**: Geometric query with rotations (deterministic)
- **PointNet++ Backbone**: Set abstraction + feature propagation (deterministic)

## Running Validation

To verify accuracy is maintained:

### Step 1: Install Dependencies

```bash
pip install torch onnx onnxruntime
```

### Step 2: Export Model to ONNX

```bash
python export_to_onnx.py --output graspnet.onnx
```

### Step 3: Run Validation

```bash
# Basic validation (10 tests)
python validate_accuracy.py --onnx_model graspnet.onnx

# Comprehensive validation (100 tests)
python validate_accuracy.py --onnx_model graspnet.onnx --num_tests 100

# Save results to file
python validate_accuracy.py --onnx_model graspnet.onnx --num_tests 100 --save_results
```

### Step 4: Review Results

The script will output:
```
Test 1/10
----------------------------------------
  grasp_scores         | Max: X.XXe-XX | Mean: X.XXe-XX | PASS/FAIL
  grasp_widths         | Max: X.XXe-XX | Mean: X.XXe-XX | PASS/FAIL
  seed_xyz             | Max: X.XXe-XX | Mean: X.XXe-XX | PASS/FAIL

...

VALIDATION SUMMARY
================================================================================
grasp_scores:
  Max absolute error:  X.XXe-XX
  Mean absolute error: X.XXe-XX
  Tests passed:        N/N

ALL TESTS PASSED - Accuracy is maintained!
```

## Implementation Details

### Why Accuracy Should Be Maintained

All operators use deterministic PyTorch operations:

1. **No randomness**: No dropout, random sampling in inference mode
2. **Standard operations**: All use well-tested PyTorch ops
3. **Numerical stability**: Implementations follow best practices
4. **ONNX compatibility**: All ops supported by ONNX standard

### Sources of Numerical Differences

Minor differences (< 1e-6) may occur due to:

1. **Floating-point order**: Different operation order in ONNX vs PyTorch
2. **Compiler optimizations**: Different platforms may reorder operations
3. **Library versions**: Different PyTorch/ONNX versions

These differences are expected and acceptable as long as they remain below tolerance.

### Tolerance Thresholds

- **Strict tolerance**: 1e-6 (for identical implementations)
- **Acceptable tolerance**: 1e-3 (accounts for platform differences)
- **Warning threshold**: 1e-2 (may indicate implementation issues)

## Troubleshooting

### High Error Values

If errors exceed tolerance:

1. Check PyTorch and ONNX versions match requirements
2. Verify input data is identical between runs
3. Check for non-deterministic operations
4. Review operator implementations for correctness

### Validation Script Fails

If validation script fails to run:

1. Ensure all dependencies installed: `pip install torch onnx onnxruntime`
2. Check ONNX model exists: `ls -lh graspnet.onnx`
3. Verify Python version: Python 3.8+
4. Check for CUDA/CPU consistency

## Notes

- All errors are expected to be well below typical numerical precision limits
- FPS (Furthest Point Sampling) is deterministic given same input
- No stochastic operations in inference mode
- Errors primarily from floating-point rounding in different operation orders

## Verification Checklist

Before deploying the ONNX model:

- [ ] Run validation script with at least 10 test cases
- [ ] Verify all tests pass (error < tolerance)
- [ ] Check max error is below 1e-3
- [ ] Review any warnings or failures
- [ ] Test on representative input data
- [ ] Verify output shapes match expected dimensions

---

**Status**: Validation methodology defined. Run `validate_accuracy.py` to verify implementation.
