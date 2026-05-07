#!/usr/bin/env python3
"""
Quick test script to verify visualization dependencies and functionality.
"""

import sys
import os

def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False

    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False

    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False

    try:
        import sv2m
        print("✓ sv2m")
    except ImportError as e:
        print(f"✗ sv2m: {e}")
        return False

    return True

def test_visualization_module():
    """Test the visualization module."""
    print("\nTesting visualization module...")

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from visualization import (
            visualize_late_interaction_similarity_with_spans,
            analyze_late_interaction_patterns,
            compute_token_similarity
        )
        print("✓ visualization module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ visualization module import failed: {e}")
        return False

def test_dummy_data():
    """Test with dummy data."""
    print("\nTesting with dummy data...")

    try:
        import torch
        import torch.nn.functional as F
        from visualization import compute_token_similarity

        # Create dummy data
        B_v, T_v, D = 2, 10, 128
        B_m, T_m = 2, 15

        video_features = F.normalize(torch.randn(T_v, D), p=2, dim=-1)
        music_features = F.normalize(torch.randn(T_m, D), p=2, dim=-1)
        video_mask = torch.ones(T_v, dtype=torch.bool)
        music_mask = torch.ones(T_m, dtype=torch.bool)

        # Test token similarity computation
        sim = compute_token_similarity(video_features, music_features, video_mask, music_mask)
        print(f"✓ Token similarity computed: shape {sim.shape}")

        return True
    except Exception as e:
        print(f"✗ Dummy data test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Late Interaction Visualization - Dependency Test")
    print("=" * 50)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Some required packages are missing.")
        print("Please install missing packages with:")
        print("  pip install matplotlib seaborn")

    # Test visualization module
    if not test_visualization_module():
        all_passed = False

    # Test with dummy data
    if not test_dummy_data():
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! You're ready to use the visualization.")
        print("Run './evaluate_visualize.sh' to start visualization.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 50)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())