/**
 * Custom QNN Hexagon Operator: Cylinder Query
 *
 * This implements cylinder-based region query for QNN/Hexagon NPU deployment.
 * Used in grasp detection to find points within a cylindrical region.
 *
 * Reference: QNN SDK Documentation - Custom Operators
 */

#include "QnnTypes.h"
#include "QnnOpDef.h"
#include "QnnInterface.h"
#include <cmath>
#include <vector>
#include <algorithm>

#define PACKAGE_NAME "graspnet_ops"

/**
 * Cylinder Query custom operator
 *
 * Inputs:
 *   - xyz: (batch, N, 3) input points
 *   - new_xyz: (batch, M, 3) cylinder centers
 *   - rot_c2w: (batch, M, 3, 3) rotation matrices (cylinder to world frame)
 *   - radius: scalar float - cylinder radius
 *   - hmin: scalar float - minimum height
 *   - hmax: scalar float - maximum height
 *   - nsample: scalar int - max samples per cylinder
 *
 * Outputs:
 *   - indices: (batch, M, nsample) int tensor - indices of points in each cylinder
 *
 * Algorithm:
 *   For each cylinder center:
 *   1. Transform points to cylinder local frame using rotation matrix
 *   2. Check if point is within radius (y-z plane) and height (x-axis)
 *   3. Select up to nsample closest points
 */

Qnn_ErrorHandle_t cylinderQueryExecute(
    Qnn_OpConfig_t opConfig,
    uint32_t numInputs,
    const Qnn_Tensor_t* inputs,
    uint32_t numOutputs,
    Qnn_Tensor_t* outputs)
{
    // Validate inputs
    if (numInputs != 6 || numOutputs != 1) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Get input tensors
    const Qnn_Tensor_t& xyzTensor = inputs[0];
    const Qnn_Tensor_t& newXyzTensor = inputs[1];
    const Qnn_Tensor_t& rotTensor = inputs[2];
    const Qnn_Tensor_t& radiusTensor = inputs[3];
    const Qnn_Tensor_t& hminTensor = inputs[4];
    const Qnn_Tensor_t& hmaxTensor = inputs[5];
    Qnn_Tensor_t& outputTensor = outputs[0];

    // Get parameters from config
    int32_t nsample = 16; // Default, should be in opConfig.params
    for (uint32_t i = 0; i < opConfig.v1.numOfParams; i++) {
        if (strcmp(opConfig.v1.params[i].name, "nsample") == 0) {
            nsample = opConfig.v1.params[i].intValue;
        }
    }

    // Get dimensions
    uint32_t batch = xyzTensor.v1.dimensions[0];
    uint32_t N = xyzTensor.v1.dimensions[1];
    uint32_t M = newXyzTensor.v1.dimensions[1];

    // Get data pointers
    const float* xyz = static_cast<const float*>(xyzTensor.v1.clientBuf.data);
    const float* new_xyz = static_cast<const float*>(newXyzTensor.v1.clientBuf.data);
    const float* rot = static_cast<const float*>(rotTensor.v1.clientBuf.data);
    float radius = *static_cast<const float*>(radiusTensor.v1.clientBuf.data);
    float hmin = *static_cast<const float*>(hminTensor.v1.clientBuf.data);
    float hmax = *static_cast<const float*>(hmaxTensor.v1.clientBuf.data);
    int32_t* indices = static_cast<int32_t*>(outputTensor.v1.clientBuf.data);

    float radius2 = radius * radius;

    // Process each batch
    for (uint32_t b = 0; b < batch; b++) {
        const float* batchXyz = xyz + b * N * 3;
        const float* batchNewXyz = new_xyz + b * M * 3;
        const float* batchRot = rot + b * M * 9;
        int32_t* batchIndices = indices + b * M * nsample;

        // Process each cylinder center
        for (uint32_t j = 0; j < M; j++) {
            const float* center = batchNewXyz + j * 3;
            const float* rotation = batchRot + j * 9;
            int32_t* cylinderIndices = batchIndices + j * nsample;

            // Collect points within cylinder
            std::vector<std::pair<float, int32_t>> candidates; // (distance, index)

            for (uint32_t k = 0; k < N; k++) {
                const float* point = batchXyz + k * 3;

                // Compute relative position
                float rel_x = point[0] - center[0];
                float rel_y = point[1] - center[1];
                float rel_z = point[2] - center[2];

                // Transform to cylinder local frame (world to cylinder)
                // R_w2c = R_c2w^T
                float x_local = rotation[0] * rel_x + rotation[1] * rel_y + rotation[2] * rel_z;
                float y_local = rotation[3] * rel_x + rotation[4] * rel_y + rotation[5] * rel_z;
                float z_local = rotation[6] * rel_x + rotation[7] * rel_y + rotation[8] * rel_z;

                // Check cylinder constraints
                float dist_yz2 = y_local * y_local + z_local * z_local;
                bool in_radius = dist_yz2 < radius2;
                bool in_height = (x_local > hmin) && (x_local < hmax);

                if (in_radius && in_height) {
                    candidates.push_back({dist_yz2, k});
                }
            }

            // Sort by distance and take closest nsample points
            std::sort(candidates.begin(), candidates.end());

            // Fill indices
            int32_t first_valid = candidates.empty() ? 0 : candidates[0].second;
            for (int32_t s = 0; s < nsample; s++) {
                if (s < static_cast<int32_t>(candidates.size())) {
                    cylinderIndices[s] = candidates[s].second;
                } else {
                    cylinderIndices[s] = first_valid; // Repeat first valid index
                }
            }
        }
    }

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t cylinderQueryValidate(
    Qnn_OpConfig_t opConfig,
    uint32_t numInputs,
    const Qnn_Tensor_t* inputs,
    uint32_t numOutputs,
    const Qnn_Tensor_t* outputs)
{
    if (numInputs != 6 || numOutputs != 1) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Validate shapes
    const Qnn_Tensor_t& xyzTensor = inputs[0];
    const Qnn_Tensor_t& newXyzTensor = inputs[1];
    const Qnn_Tensor_t& rotTensor = inputs[2];

    if (xyzTensor.v1.rank != 3 || xyzTensor.v1.dimensions[2] != 3) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    if (newXyzTensor.v1.rank != 3 || newXyzTensor.v1.dimensions[2] != 3) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    if (rotTensor.v1.rank != 4 || rotTensor.v1.dimensions[2] != 3 ||
        rotTensor.v1.dimensions[3] != 3) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    return QNN_SUCCESS;
}

static Qnn_OpConfig_t cylinderQueryOpConfig = {
    .version = QNN_OP_CONFIG_VERSION_1,
    .v1 = {
        .packageName = PACKAGE_NAME,
        .typeName = "CylinderQuery",
        .numOfInputs = 6,
        .numOfOutputs = 1,
        .executeFunc = cylinderQueryExecute,
        .validateFunc = cylinderQueryValidate,
    }
};

/**
 * HEXAGON NPU OPTIMIZATION STRATEGIES:
 *
 * 1. Memory Access Patterns:
 *    - Use contiguous memory access for xyz and new_xyz
 *    - Prefetch rotation matrices to VTCM
 *    - Stream point data through L2 cache efficiently
 *
 * 2. Vectorization Opportunities:
 *    - Batch process multiple points using HVX
 *    - Parallel distance computations using SIMD
 *    - Matrix-vector multiplication using HMX
 *
 * 3. Quantization:
 *    - Use 16-bit fixed-point for coordinates
 *    - Quantize rotation matrices to int8
 *    - Store squared distances as int16 to avoid float operations
 *
 * 4. Algorithmic Optimizations:
 *    - Early exit for points far from cylinder
 *    - Spatial hashing to reduce search space
 *    - Parallel processing of independent cylinders
 *
 * 5. Performance Tuning:
 *    - Profile using QNN profiler to identify bottlenecks
 *    - Optimize loop unrolling and prefetching
 *    - Use HTA for parallel batch processing
 */

/**
 * QUANTIZED VERSION FOR HEXAGON:
 *
 * Convert to 16-bit fixed-point arithmetic:
 * - Coordinates: Q15.16 format (range: ±32768)
 * - Rotation: Q7.8 format (range: ±128)
 * - Distances: Q15.16 format
 *
 * Benefits:
 * - 2x memory bandwidth improvement
 * - 4x faster on Hexagon HVX (128-byte vectors)
 * - Lower power consumption
 */
