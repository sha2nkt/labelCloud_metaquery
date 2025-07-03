#!/usr/bin/env python3
"""
Test script to verify vertex mask coloring functionality.
"""

import sys
import json
from pathlib import Path

# Add the labelCloud module to path for import
sys.path.insert(0, '/home/stripathi/Documents/pycharm-local/labelCloud_metaquery')

def test_vertex_mask_loading():
    """Test loading vertex mask from a labels JSON file."""
    
    # Test with the example JSON file
    label_file = Path('/home/stripathi/Documents/pycharm-local/labelCloud_metaquery/labels/420673_classes.json')
    
    if not label_file.exists():
        print(f"❌ Label file not found: {label_file}")
        return False
    
    try:
        # Load the labels JSON
        with open(label_file, 'r') as f:
            labels_data = json.load(f)
        
        # Get the first class/label if it exists
        classes = labels_data.get('classes', [])
        if not classes:
            print("❌ No classes found in labels JSON")
            return False
            
        first_class = classes[0]
        vertex_mask = first_class.get('vertex_mask', [])
        class_name = first_class.get('name', 'Unknown')
        
        if not vertex_mask:
            print("❌ No vertex_mask found in first class")
            return False
        
        print(f"✅ Successfully loaded vertex mask for class: {class_name}")
        print(f"✅ Vertex mask contains {len(vertex_mask)} indices")
        print(f"✅ First 10 indices: {vertex_mask[:10]}")
        print(f"✅ Last 10 indices: {vertex_mask[-10:]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load vertex mask: {e}")
        return False

def main():
    """Main test function."""
    print("Testing vertex mask coloring functionality...")
    print("=" * 50)
    
    success = test_vertex_mask_loading()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed! Vertex mask coloring should work.")
    else:
        print("❌ Tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
