"""
Convert ONNX model to QNN (Qualcomm Neural Network) format
Requires Qualcomm Neural Network SDK
"""

import argparse
import os
import subprocess
import json
from pathlib import Path


class QNNConverter:
    """
    Wrapper for QNN model conversion and optimization.
    """
    def __init__(self, qnn_sdk_path=None):
        """
        Initialize QNN converter.

        Args:
            qnn_sdk_path: Path to QNN SDK installation
        """
        if qnn_sdk_path is None:
            qnn_sdk_path = os.environ.get('QNN_SDK_ROOT', '/opt/qcom/aistack/qnn')

        self.qnn_sdk_path = qnn_sdk_path
        self.converter_path = os.path.join(qnn_sdk_path, 'bin', 'x86_64-linux-clang', 'qnn-onnx-converter')
        self.context_binary_generator = os.path.join(qnn_sdk_path, 'bin', 'x86_64-linux-clang', 'qnn-context-binary-generator')
        self.net_run = os.path.join(qnn_sdk_path, 'bin', 'x86_64-linux-clang', 'qnn-net-run')

        if not os.path.exists(self.converter_path):
            raise FileNotFoundError(
                f"QNN converter not found at {self.converter_path}\n"
                f"Please install Qualcomm Neural Network SDK and set QNN_SDK_ROOT"
            )

    def convert_onnx_to_qnn(self, onnx_path, output_dir, input_dims, quantize=False):
        """
        Convert ONNX model to QNN format.

        Args:
            onnx_path: Path to ONNX model
            output_dir: Output directory for QNN model
            input_dims: Input dimensions string (e.g., "point_cloud 1,20000,3")
            quantize: Whether to quantize the model

        Returns:
            Path to generated QNN model
        """
        print("="*80)
        print("CONVERTING ONNX TO QNN")
        print("="*80)

        os.makedirs(output_dir, exist_ok=True)

        # Prepare converter command
        model_name = Path(onnx_path).stem
        output_path = os.path.join(output_dir, f"{model_name}.cpp")

        cmd = [
            self.converter_path,
            '--input_network', onnx_path,
            '--output_path', output_path,
        ]

        # Add input dimensions
        if input_dims:
            cmd.extend(['--input_dim', input_dims])

        # Add quantization if requested
        if quantize:
            print("WARNING Note: Quantization requires calibration data")
            cmd.extend([
                '--quantization_overrides', 'activation:16,weights:8',
                '--use_per_channel_quantization'
            ])

        # Run converter
        print(f"\nRunning QNN converter...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print("SUCCESS Conversion successful")
            print(result.stdout)

            return output_path

        except subprocess.CalledProcessError as e:
            print(f"WARNING Conversion failed:")
            print(e.stdout)
            print(e.stderr)
            raise

    def generate_context_binary(self, cpp_path, output_dir, backend='GPU', config=None):
        """
        Generate QNN context binary for deployment.

        Args:
            cpp_path: Path to QNN C++ model
            output_dir: Output directory
            backend: Target backend ('CPU', 'GPU', 'DSP', 'HTP')
            config: Optional JSON config file

        Returns:
            Path to generated context binary
        """
        print("\n" + "="*80)
        print("GENERATING QNN CONTEXT BINARY")
        print("="*80)

        model_name = Path(cpp_path).stem
        output_path = os.path.join(output_dir, f"{model_name}_{backend.lower()}.bin")

        cmd = [
            self.context_binary_generator,
            '--model', cpp_path,
            '--output_dir', output_dir,
            '--backend', backend,
        ]

        if config:
            cmd.extend(['--config_file', config])

        print(f"\nGenerating context binary for {backend} backend...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print("SUCCESS Context binary generated successfully")
            print(result.stdout)

            return output_path

        except subprocess.CalledProcessError as e:
            print(f"WARNING Generation failed:")
            print(e.stdout)
            print(e.stderr)
            raise

    def quantize_model(self, model_path, calibration_data_path, output_path):
        """
        Quantize QNN model for Hexagon NPU deployment.

        Args:
            model_path: Path to QNN model
            calibration_data_path: Path to calibration data
            output_path: Path for quantized model

        Returns:
            Path to quantized model
        """
        print("\n" + "="*80)
        print("QUANTIZING MODEL FOR HEXAGON NPU")
        print("="*80)

        print("""
Quantization Steps:
1. Prepare calibration dataset (representative point clouds)
2. Run model with calibration data to collect statistics
3. Generate quantized model with optimal parameters
        """)

        # This would use QNN quantization tools
        # Actual implementation depends on QNN SDK version and requirements
        print("WARNING Quantization requires calibration data and QNN quantization tools")
        print("  Please refer to QNN SDK documentation for quantization workflow")

        return output_path


def create_qnn_config(output_path, backend='HTP', precision='fp16'):
    """
    Create QNN configuration file for optimization.

    Args:
        output_path: Path to save config
        backend: Target backend
        precision: Precision mode ('fp32', 'fp16', 'int8')
    """
    config = {
        'backend': backend,
        'precision': precision,
        'optimization_level': 3,
        'enable_fusion': True,
        'enable_layout_optimization': True,
    }

    if backend == 'HTP':
        # Hexagon-specific optimizations
        config.update({
            'enable_hta': True,  # Hexagon Tensor Accelerator
            'enable_hmx': True,  # Hexagon Matrix Extensions
            'vtcm_size': 'large',  # Vector Tightly Coupled Memory
            'performance_profile': 'high_performance',
        })

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"SUCCESS QNN config created: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to QNN format')
    parser.add_argument('--onnx_model', type=str, required=True,
                       help='Path to ONNX model')
    parser.add_argument('--output_dir', type=str, default='qnn_models',
                       help='Output directory for QNN models')
    parser.add_argument('--qnn_sdk_path', type=str, default=None,
                       help='Path to QNN SDK installation')
    parser.add_argument('--input_dims', type=str, default='point_cloud 1,20000,3',
                       help='Input dimensions (name shape)')
    parser.add_argument('--backend', type=str, default='HTP',
                       choices=['CPU', 'GPU', 'DSP', 'HTP'],
                       help='Target backend (HTP = Hexagon NPU)')
    parser.add_argument('--quantize', action='store_true',
                       help='Enable quantization')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision mode')
    parser.add_argument('--calibration_data', type=str, default=None,
                       help='Path to calibration data for quantization')

    args = parser.parse_args()

    print("="*80)
    print("QNN MODEL CONVERSION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Input ONNX: {args.onnx_model}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Backend: {args.backend}")
    print(f"  Precision: {args.precision}")
    print(f"  Quantization: {'Enabled' if args.quantize else 'Disabled'}")

    # Initialize converter
    try:
        converter = QNNConverter(qnn_sdk_path=args.qnn_sdk_path)
    except FileNotFoundError as e:
        print(f"\nWARNING Error: {e}")
        print("\nPlease install Qualcomm Neural Network SDK:")
        print("  1. Download from: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk")
        print("  2. Extract and set environment variable:")
        print("     export QNN_SDK_ROOT=/path/to/qnn/sdk")
        return

    # Create QNN config
    config_path = os.path.join(args.output_dir, 'qnn_config.json')
    create_qnn_config(config_path, backend=args.backend, precision=args.precision)

    # Convert ONNX to QNN
    try:
        qnn_model_path = converter.convert_onnx_to_qnn(
            args.onnx_model,
            args.output_dir,
            args.input_dims,
            quantize=args.quantize
        )

        # Generate context binary
        binary_path = converter.generate_context_binary(
            qnn_model_path,
            args.output_dir,
            backend=args.backend,
            config=config_path
        )

        # Quantize if requested and calibration data provided
        if args.quantize and args.calibration_data:
            quantized_path = converter.quantize_model(
                qnn_model_path,
                args.calibration_data,
                os.path.join(args.output_dir, 'quantized_model.cpp')
            )

        print("\n" + "="*80)
        print("CONVERSION COMPLETE")
        print("="*80)
        print(f"QNN model: {qnn_model_path}")
        if 'binary_path' in locals():
            print(f"Context binary: {binary_path}")
        print("\nNext steps:")
        print("  1. Test on Hexagon NPU simulator or device")
        print("  2. Profile performance using QNN profiling tools")
        print("  3. Optimize using QNN graph optimization passes")
        print("  4. Implement custom ops if needed (see qnn/custom_ops/)")

    except Exception as e:
        print(f"\nWARNING Conversion failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check QNN SDK installation")
        print("  - Verify ONNX model compatibility")
        print("  - Review unsupported operators")
        print("  - Consider implementing custom QNN ops for unsupported operations")


if __name__ == '__main__':
    main()
