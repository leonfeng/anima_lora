#!/usr/bin/env python3
import os
import sys
from PIL import Image

def list_image_resolutions(dataset_dir):
    # Expand home directory symbol (~) if present
    target_dir = os.path.expanduser(dataset_dir)
    
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    print(f"{'Image File':<50} | {'Width':<6} x {'Height':<6} | {'Status (>=1536)'}")
    print("-" * 80)

    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    total_images = 0
    eligible_count = 0

    # Read the directory files
    for filename in sorted(os.listdir(target_dir)):
        if filename.lower().endswith(image_extensions):
            total_images += 1
            file_path = os.path.join(target_dir, filename)
            
            try:
                # opening the image loads only the metadata/header, not the whole image
                with Image.open(file_path) as img:
                    width, height = img.size
                    max_dim = max(width, height)
                    
                    if max_dim >= 1536:
                        status = "✓ Ready for 1536"
                        eligible_count += 1
                    else:
                        status = "✗ Too Small (Use 1024)"
                        
                    print(f"{filename:<50} | {width:<6} x {height:<6} | {status}")
            except Exception as e:
                print(f"{filename:<50} | Error reading image file: {e}")

    print("-" * 80)
    print(f"Summary: {eligible_count}/{total_images} images are eligible for the 1536 target resolution layer.")

if __name__ == "__main__":
    # Fallback to your specific project workspace path if no argument is passed
    default_path = "~/ai-workspace/projects/current/dataset"
    
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    list_image_resolutions(path)
