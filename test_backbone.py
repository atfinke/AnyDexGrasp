"""Test PointNet2 backbone in isolation"""
import sys
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from pointnet2_backbone_onnx import PointNet2Backbone

def test_backbone():
    print("Testing PointNet2 Backbone")
    print("="*60)

    model = PointNet2Backbone(
        in_channels=3,
        out_channels=64,
        feature_dim=512,
        half_views=False
    )
    model.eval()

    xyz = torch.randn(1, 200, 3) * 0.3  # Small for debugging
    print(f"Input: {xyz.shape}")

    with torch.no_grad():
        try:
            output, features = model(xyz, return_features=True)
            print(f"SUCCESS")
            print(f"  Output: {output.shape}")
            print(f"  Features: {features.shape}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_backbone()
