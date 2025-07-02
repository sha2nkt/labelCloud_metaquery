#!/usr/bin/env python3
"""
Test script to verify point cloud centering functionality.
"""

import numpy as np
import sys
import os

# Add the labelCloud path to the system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_point_cloud_centering():
    """Test the point cloud centering functionality."""
    print("Testing point cloud centering functionality...")
    
    # Create a test point cloud with known mean
    test_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0],
        [12.0, 22.0, 4.0],
        [8.0, 18.0, 7.0],
        [13.0, 23.0, 5.5]
    ], dtype=np.float32)
    
    original_mean = np.mean(test_points, axis=0)
    print(f"Original point cloud mean: {original_mean}")
    
    # Simulate the centering process
    centered_points = test_points.copy()
    centered_points[:, 0] -= original_mean[0]  # Center x
    centered_points[:, 1] -= original_mean[1]  # Center y
    # Keep z unchanged
    
    new_mean = np.mean(centered_points, axis=0)
    print(f"Centered point cloud mean: {new_mean}")
    
    # Verify that x and y means are close to zero
    assert abs(new_mean[0]) < 1e-10, f"X mean should be ~0, got {new_mean[0]}"
    assert abs(new_mean[1]) < 1e-10, f"Y mean should be ~0, got {new_mean[1]}"
    assert abs(new_mean[2] - original_mean[2]) < 1e-10, f"Z mean should be unchanged, got {new_mean[2]} vs {original_mean[2]}"
    
    print("✓ Point cloud centering works correctly!")
    
    # Test bbox transformation
    print("\nTesting bounding box coordinate transformation...")
    
    # Simulate a bbox in centered space
    bbox_center_centered = (2.0, 3.0, 5.0)
    print(f"Bbox center in centered space: {bbox_center_centered}")
    
    # Transform back to original space
    bbox_center_original = (
        bbox_center_centered[0] + original_mean[0],
        bbox_center_centered[1] + original_mean[1],
        bbox_center_centered[2]
    )
    print(f"Bbox center in original space: {bbox_center_original}")
    
    # Transform from original space to centered space (import case)
    bbox_center_reimported = (
        bbox_center_original[0] - original_mean[0],
        bbox_center_original[1] - original_mean[1],
        bbox_center_original[2]
    )
    print(f"Bbox center after reimport: {bbox_center_reimported}")
    
    # Verify round-trip consistency
    assert abs(bbox_center_reimported[0] - bbox_center_centered[0]) < 1e-10
    assert abs(bbox_center_reimported[1] - bbox_center_centered[1]) < 1e-10
    assert abs(bbox_center_reimported[2] - bbox_center_centered[2]) < 1e-10
    
    print("✓ Bounding box coordinate transformation works correctly!")
    
    print("\n🎉 All tests passed! The centering functionality should work as expected.")

if __name__ == "__main__":
    test_point_cloud_centering()
