/**
 * Custom QNN Hexagon Operator: Furthest Point Sampling
 *
 * This implements FPS for QNN/Hexagon NPU deployment.
 * Based on the QNN custom operator API.
 *
 * Reference: QNN SDK Documentation - Custom Operators
 */

#include "QnnTypes.h"
#include "QnnOpDef.h"
#include "QnnInterface.h"
#include <cmath>
#include <limits>
#include <vector>

// Operator package name and version
#define PACKAGE_NAME "graspnet_ops"
#define PACKAGE_VERSION "1.0.0"

/**
 * Furthest Point Sampling (FPS) custom operator
 *
 * Inputs:
 *   - points: (batch, num_points, 3) float tensor
 *   - npoint: scalar int - number of points to sample
 *
 * Outputs:
 *   - indices: (batch, npoint) int tensor - indices of sampled points
 *
 * Description:
 *   Greedily samples npoint points from the input point cloud,
 *   selecting points that are maximally distant from already selected points.
 */

// Operator implementation
Qnn_ErrorHandle_t furthestPointSampleExecute(
    Qnn_OpConfig_t opConfig,
    uint32_t numInputs,
    const Qnn_Tensor_t* inputs,
    uint32_t numOutputs,
    Qnn_Tensor_t* outputs)
{
    // Validate inputs
    if (numInputs != 2 || numOutputs != 1) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Get input tensors
    const Qnn_Tensor_t& pointsTensor = inputs[0];
    const Qnn_Tensor_t& npointTensor = inputs[1];
    Qnn_Tensor_t& outputTensor = outputs[0];

    // Get input dimensions
    uint32_t batch = pointsTensor.v1.dimensions[0];
    uint32_t numPoints = pointsTensor.v1.dimensions[1];
    uint32_t dim = pointsTensor.v1.dimensions[2]; // Should be 3

    // Get npoint parameter
    int32_t npoint = *static_cast<const int32_t*>(npointTensor.v1.clientBuf.data);

    // Get data pointers
    const float* points = static_cast<const float*>(pointsTensor.v1.clientBuf.data);
    int32_t* indices = static_cast<int32_t*>(outputTensor.v1.clientBuf.data);

    // FPS algorithm for each batch
    for (uint32_t b = 0; b < batch; b++) {
        const float* batchPoints = points + b * numPoints * dim;
        int32_t* batchIndices = indices + b * npoint;

        // Distance array
        std::vector<float> distances(numPoints, std::numeric_limits<float>::max());

        // Select first point (farthest from origin or random)
        int32_t farthest = 0;
        batchIndices[0] = farthest;

        // Iteratively select farthest points
        for (int32_t i = 1; i < npoint; i++) {
            // Get current farthest point coordinates
            const float* centroid = batchPoints + farthest * dim;

            // Update distances to all points
            float maxDist = -1.0f;
            int32_t maxIdx = 0;

            for (uint32_t j = 0; j < numPoints; j++) {
                const float* point = batchPoints + j * dim;

                // Compute squared distance to centroid
                float dist = 0.0f;
                for (uint32_t d = 0; d < dim; d++) {
                    float diff = point[d] - centroid[d];
                    dist += diff * diff;
                }

                // Update minimum distance
                if (dist < distances[j]) {
                    distances[j] = dist;
                }

                // Find maximum minimum distance
                if (distances[j] > maxDist) {
                    maxDist = distances[j];
                    maxIdx = j;
                }
            }

            // Select farthest point
            farthest = maxIdx;
            batchIndices[i] = farthest;
        }
    }

    return QNN_SUCCESS;
}

// Operator validation
Qnn_ErrorHandle_t furthestPointSampleValidate(
    Qnn_OpConfig_t opConfig,
    uint32_t numInputs,
    const Qnn_Tensor_t* inputs,
    uint32_t numOutputs,
    const Qnn_Tensor_t* outputs)
{
    // Check input count
    if (numInputs != 2) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Check output count
    if (numOutputs != 1) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Validate input shapes
    const Qnn_Tensor_t& pointsTensor = inputs[0];
    if (pointsTensor.v1.rank != 3) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    if (pointsTensor.v1.dimensions[2] != 3) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Validate data types
    if (pointsTensor.v1.dataType != QNN_DATATYPE_FLOAT_32) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    return QNN_SUCCESS;
}

// Operator registration
static Qnn_OpConfig_t furthestPointSampleOpConfig = {
    .version = QNN_OP_CONFIG_VERSION_1,
    .v1 = {
        .packageName = PACKAGE_NAME,
        .typeName = "FurthestPointSample",
        .numOfInputs = 2,
        .numOfOutputs = 1,
        .executeFunc = furthestPointSampleExecute,
        .validateFunc = furthestPointSampleValidate,
    }
};

/**
 * OPTIMIZATION NOTES FOR HEXAGON NPU:
 *
 * 1. Vectorization:
 *    - Use Hexagon Vector eXtensions (HVX) for distance computations
 *    - Process multiple points in parallel using SIMD
 *
 * 2. Memory Optimization:
 *    - Use VTCM (Vector Tightly Coupled Memory) for distance array
 *    - Minimize memory transfers between L2 and VTCM
 *
 * 3. Quantization:
 *    - Consider fixed-point arithmetic for distance computations
 *    - Use 16-bit integers instead of float32 for performance
 *
 * 4. Parallelization:
 *    - Process batches in parallel using HTA (Hexagon Tensor Accelerator)
 *    - Use HMX (Hexagon Matrix Extensions) for distance matrix computation
 *
 * Example optimized version using HVX:
 */

#ifdef USE_HVX_OPTIMIZATION
#include "hvx_interface.h"

// HVX-optimized distance computation
inline float computeDistanceHVX(const float* p1, const float* p2, int dim) {
    // Use HVX vector operations for faster computation
    HVX_Vector v1 = *(HVX_Vector*)p1;
    HVX_Vector v2 = *(HVX_Vector*)p2;
    HVX_Vector diff = Q6_Vqf32_vsub_Vqf32Vqf32(v1, v2);
    HVX_Vector sq = Q6_Vqf32_vmpy_Vqf32Vqf32(diff, diff);

    // Horizontal sum
    float result = 0.0f;
    for (int i = 0; i < dim; i++) {
        result += ((float*)&sq)[i];
    }
    return result;
}
#endif

/**
 * BUILD INSTRUCTIONS:
 *
 * 1. Set up QNN SDK environment:
 *    export QNN_SDK_ROOT=/path/to/qnn/sdk
 *    export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
 *
 * 2. Compile the custom operator:
 *    qnn-op-package-generator \
 *      --input furthest_point_sample_qnn.cpp \
 *      --output_dir build \
 *      --package_name graspnet_ops \
 *      --target hexagon-v68
 *
 * 3. Link with model:
 *    qnn-context-binary-generator \
 *      --model model.cpp \
 *      --backend HTP \
 *      --op_packages build/libgraspnet_ops.so \
 *      --output_dir output
 */
