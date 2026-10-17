#!/usr/bin/env python3

import json
import os
import glob
from pathlib import Path

def merge_json_files(input_dir, output_file):
    """
    Merge all JSON files in a directory into a single JSON file.
    Each JSON file should contain an array, and we'll combine all arrays into one.
    """
    input_path = Path(input_dir)
    merged_data = []
    
    # Get all JSON files in the directory
    json_files = sorted(glob.glob(str(input_path / "*.json")))
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Assuming each file contains an array
            if isinstance(data, list):
                merged_data.extend(data)
                print(f"  - Added {len(data)} items")
            else:
                print(f"  - Warning: {json_file} does not contain an array, skipping")
                
        except Exception as e:
            print(f"  - Error processing {json_file}: {e}")
    
    # Write merged data to output file
    print(f"\nWriting merged data to {output_file}")
    print(f"Total items: {len(merged_data)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully merged {len(json_files)} files into {output_file}")

def main():
    # Merge test files
    test_dir = "/Users/weixiaochen/Desktop/Agent+SFT/converted_test_final"
    test_output = "/Users/weixiaochen/Desktop/Agent+SFT/merged_test_final.json"
    
    print("=== Merging Test Files ===")
    merge_json_files(test_dir, test_output)
    
    # Merge train files
    train_dir = "/Users/weixiaochen/Desktop/Agent+SFT/converted_train_final"
    train_output = "/Users/weixiaochen/Desktop/Agent+SFT/merged_train_final.json"
    
    print("\n=== Merging Train Files ===")
    merge_json_files(train_dir, train_output)
    
    print("\n=== Summary ===")
    print(f"Test dataset merged into: {test_output}")
    print(f"Train dataset merged into: {train_output}")

if __name__ == "__main__":
    main()