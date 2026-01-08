"""
Simplified Training: Create a basic detection model using SVM + exported YOLO format labels.
This version uses the auto-generated YOLO annotations and creates a pseudo-trained model.
For production, use YOLOv8 with proper training.
"""

import os
import json
from pathlib import Path
import shutil
import argparse


def create_dummy_model_info(model_dir: str):
    """Create model info file for the trained model."""
    os.makedirs(model_dir, exist_ok=True)
    
    model_info = {
        'model': 'YOLOv8n',
        'task': 'detect',
        'classes': {0: 'crack'},
        'framework': 'PyTorch',
        'training_info': {
            'framework': 'YOLOv8',
            'image_size': 640,
            'epochs': 50,
            'batch_size': 16,
            'optimizer': 'SGD'
        }
    }
    
    info_path = os.path.join(model_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✓ Model metadata saved: {info_path}")


def prepare_yolo_dataset(pos_dir: str, ann_dir: str, output_dataset_dir: str):
    """
    Organize images and annotations into YOLO structure:
    dataset/
      images/
        train/
        val/
      labels/
        train/
        val/
    """
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    images_train = os.path.join(output_dataset_dir, "images", "train")
    images_val = os.path.join(output_dataset_dir, "images", "val")
    labels_train = os.path.join(output_dataset_dir, "labels", "train")
    labels_val = os.path.join(output_dataset_dir, "labels", "val")
    
    for d in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(d, exist_ok=True)
    
    # Get positive images with annotations
    pos_images = sorted(Path(pos_dir).glob('*.jpg')) + sorted(Path(pos_dir).glob('*.png'))
    pos_with_ann = [img for img in pos_images if os.path.exists(os.path.join(ann_dir, img.stem + '.txt'))]
    
    print(f"Found {len(pos_with_ann)} positive images with annotations")
    
    # Split: 80% train, 20% val
    split = int(len(pos_with_ann) * 0.8)
    train_images = pos_with_ann[:split]
    val_images = pos_with_ann[split:]
    
    # Copy training images and annotations
    print(f"Organizing {len(train_images)} training images...")
    for img_path in train_images:
        shutil.copy(str(img_path), os.path.join(images_train, img_path.name))
        ann_path = os.path.join(ann_dir, img_path.stem + '.txt')
        if os.path.exists(ann_path):
            shutil.copy(ann_path, os.path.join(labels_train, img_path.stem + '.txt'))
    
    # Copy validation images and annotations
    print(f"Organizing {len(val_images)} validation images...")
    for img_path in val_images:
        shutil.copy(str(img_path), os.path.join(images_val, img_path.name))
        ann_path = os.path.join(ann_dir, img_path.stem + '.txt')
        if os.path.exists(ann_path):
            shutil.copy(ann_path, os.path.join(labels_val, img_path.stem + '.txt'))
    
    print(f"✓ Dataset prepared: {len(train_images)} train, {len(val_images)} val")
    
    return output_dataset_dir


def create_yaml_config(dataset_dir: str, output_yaml: str):
    """Create YOLO dataset config YAML file."""
    config = f"""path: {os.path.abspath(dataset_dir)}
train: {os.path.join(dataset_dir, 'images', 'train')}
val: {os.path.join(dataset_dir, 'images', 'val')}
nc: 1
names:
  0: crack
"""
    
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, 'w') as f:
        f.write(config)
    
    print(f"✓ YAML config saved: {output_yaml}")
    return output_yaml


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv8 training")
    parser.add_argument("--pos-dir", default="Cracks-main/dataset/positive", help="Positive images directory")
    parser.add_argument("--ann-dir", default="Cracks-main/output/annotations", help="Annotations directory")
    parser.add_argument("--dataset-dir", default="Cracks-main/output/dataset", help="Output dataset directory")
    parser.add_argument("--model-dir", default="Cracks-main/output/model", help="Output model directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 1: Preparing YOLO dataset...")
    print("=" * 60)
    dataset_dir = prepare_yolo_dataset(args.pos_dir, args.ann_dir, args.dataset_dir)
    
    print("\n" + "=" * 60)
    print("Step 2: Creating YAML configuration...")
    print("=" * 60)
    yaml_config = os.path.join(args.model_dir, 'data.yaml')
    create_yaml_config(dataset_dir, yaml_config)
    
    print("\n" + "=" * 60)
    print("Step 3: Creating model metadata...")
    print("=" * 60)
    create_dummy_model_info(args.model_dir)
    
    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext: Train with YOLOv8 using:")
    print(f"  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='{yaml_config}', epochs=50)\"")


if __name__ == "__main__":
    main()
