"""
Setup YOLO Dataset Structure
This script organizes your Fresh and Rotten apple images for YOLO training.
"""

import os
import shutil
from pathlib import Path

# Your source directories
FRESH_DIR = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "Fresh1")
ROTTEN_DIR = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "Rotten1")

# YOLO dataset directories
BASE_DIR = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")

def create_directory_structure():
    """Create YOLO dataset directory structure."""
    directories = [
        os.path.join(IMAGES_DIR, "train"),
        os.path.join(IMAGES_DIR, "val"),
        os.path.join(IMAGES_DIR, "test"),
        os.path.join(LABELS_DIR, "train"),
        os.path.join(LABELS_DIR, "val"),
        os.path.join(LABELS_DIR, "test"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")


def create_dummy_label(image_path, output_label_path, class_id):
    """
    Create a dummy YOLO label file (bounding box covering entire image).
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to 0-1
    
    For whole image: 0.5 0.5 1.0 1.0 (center at middle, full width/height)
    """
    # Since we don't have labeled bounding boxes, create a label that covers the whole image
    # This assumes the entire image is the apple
    with open(output_label_path, 'w') as f:
        # Format: class_id x_center y_center width height (all normalized 0-1)
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def organize_dataset(source_dir, category, class_id, train_count=32):
    """
    Organize images into train/val splits.
    
    Parameters:
    -----------
    source_dir : str
        Source directory containing images
    category : str
        'Fresh' or 'Rotten'
    class_id : int
        0 for fresh, 1 for rotten
    train_count : int
        Number of images for training (rest go to validation)
    """
    if not os.path.exists(source_dir):
        print(f"✗ Directory not found: {source_dir}")
        return
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]
    images.sort()  # Sort to ensure consistent order
    
    print(f"\nProcessing {category} apples...")
    print(f"  Total images: {len(images)}")
    
    if len(images) == 0:
        print(f"  ✗ No images found")
        return
    
    # Split into train and validation
    train_images = images[:train_count]
    val_images = images[train_count:]
    
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    
    # Copy training images and create labels
    for img_name in train_images:
        src_img = os.path.join(source_dir, img_name)
        dst_img = os.path.join(IMAGES_DIR, "train", img_name)
        
        # Copy image
        shutil.copy2(src_img, dst_img)
        
        # Create label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, "train", label_name)
        create_dummy_label(src_img, label_path, class_id)
        
        print(f"  ✓ Train: {img_name}")
    
    # Copy validation images and create labels
    for img_name in val_images:
        src_img = os.path.join(source_dir, img_name)
        dst_img = os.path.join(IMAGES_DIR, "val", img_name)
        
        # Copy image
        shutil.copy2(src_img, dst_img)
        
        # Create label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, "val", label_name)
        create_dummy_label(src_img, label_path, class_id)
        
        print(f"  ✓ Val: {img_name}")


def create_data_yaml():
    """Create data.yaml configuration file."""
    yaml_content = f"""# YOLO Dataset Configuration for Apple Grading System

# Dataset paths
path: {BASE_DIR}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 2

# Class names
names:
  0: fresh
  1: rotten
"""
    
    yaml_path = os.path.join(BASE_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created: {yaml_path}")


def main():
    """Main setup function."""
    print("="*60)
    print("YOLO DATASET SETUP")
    print("="*60)
    
    # Step 1: Create directory structure
    print("\nStep 1: Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Organize Fresh apples (class_id=0)
    print("\nStep 2: Organizing dataset...")
    organize_dataset(FRESH_DIR, "Fresh", class_id=0, train_count=32)
    
    # Step 3: Organize Rotten apples (class_id=1)
    organize_dataset(ROTTEN_DIR, "Rotten", class_id=1, train_count=32)
    
    # Step 4: Create data.yaml
    print("\nStep 3: Creating data.yaml...")
    create_data_yaml()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nDataset Summary:")
    print(f"  Training images: 64 (32 Fresh + 32 Rotten)")
    print(f"  Validation images: 16 (8 Fresh + 8 Rotten)")
    print(f"  Total: 80 images")
    print("\nYou can now run: python main.py → Option 1 (Train Model)")


if __name__ == "__main__":
    main()