#!/usr/bin/env python3

# Test basic imports to check for syntax errors
try:
    from labelCloud.view.gui import GUI
    print("✓ GUI import successful")
except Exception as e:
    print(f"✗ GUI import failed: {e}")

try:
    from labelCloud.control.bbox_controller import BoundingBoxController
    print("✓ BoundingBoxController import successful")
except Exception as e:
    print(f"✗ BoundingBoxController import failed: {e}")

try:
    from labelCloud.control.pcd_manager import PointCloudManger
    print("✓ PointCloudManger import successful")
except Exception as e:
    print(f"✗ PointCloudManger import failed: {e}")

print("Basic import test completed!")
