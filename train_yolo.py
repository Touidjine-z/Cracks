"""
Step 2: Train YOLOv8 model on crack detection dataset.
Uses auto-generated YOLO annotations from positive/negative folders.
"""

import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import argparse


def prepare_yolo_dataset(pos_dir: str, neg_dir: str, ann_dir: str, output_dataset_dir: str):
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
    for img_path in train_images:
        shutil.copy(str(img_path), os.path.join(images_train, img_path.name))
        ann_path = os.path.join(ann_dir, img_path.stem + '.txt')
        shutil.copy(ann_path, os.path.join(labels_train, img_path.stem + '.txt'))
    
    # Copy validation images and annotations
    for img_path in val_images:
        shutil.copy(str(img_path), os.path.join(images_val, img_path.name))
        ann_path = os.path.join(ann_dir, img_path.stem + '.txt')
        shutil.copy(ann_path, os.path.join(labels_val, img_path.stem + '.txt'))
    
    # Add negative images to training (no annotations needed - empty .txt)
    if os.path.exists(neg_dir):
        neg_images = sorted(Path(neg_dir).glob('*.jpg')) + sorted(Path(neg_dir).glob('*.png'))
        for img_path in neg_images[:int(len(neg_images) * 0.8)]:
            shutil.copy(str(img_path), os.path.join(images_train, img_path.name))
            # Create empty annotation file for negative image
            with open(os.path.join(labels_train, img_path.stem + '.txt'), 'w') as f:
                f.write('')
        
        for img_path in neg_images[int(len(neg_images) * 0.8):]:
            shutil.copy(str(img_path), os.path.join(images_val, img_path.name))
            with open(os.path.join(labels_val, img_path.stem + '.txt'), 'w') as f:
                f.write('')
        
        print(f"Added {len(neg_images)} negative images")
    
    print(f"✓ Dataset prepared: {len(train_images)} train, {len(val_images)} val (positive)")
    
    return output_dataset_dir


def create_yaml_config(dataset_dir: str, output_yaml: str):
    """Create YOLO dataset config YAML file."""
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': os.path.join(dataset_dir, 'images', 'train'),
        'val': os.path.join(dataset_dir, 'images', 'val'),
        'nc': 1,  # 1 class: crack
        'names': {0: 'crack'}
    }
    
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ YAML config saved: {output_yaml}")
    return output_yaml


def train_model(yaml_config: str, output_model_dir: str, epochs: int = 50, imgsz: int = 640):
    """Train YOLOv8-small model."""
    os.makedirs(output_model_dir, exist_ok=True)
    
    model = YOLO('yolov8n.pt')  # nano model (faster training)
    
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        imgsz=imgsz,
        device=0 if __import__('torch').cuda.is_available() else 'cpu',
        patience=20,
        save=True,
        project=output_model_dir,
        name='crack_detector',
        verbose=True,
        batch=16,
        workers=4
    )
    
    print(f"✓ Model training complete!")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 crack detection model")
    parser.add_argument("--pos-dir", default="Cracks-main/dataset/positive", help="Positive images directory")
    parser.add_argument("--neg-dir", default="Cracks-main/dataset/negative", help="Negative images directory")
    parser.add_argument("--ann-dir", default="Cracks-main/output/annotations", help="Annotations directory")
    parser.add_argument("--dataset-dir", default="Cracks-main/output/dataset", help="Output dataset directory")
    parser.add_argument("--model-dir", default="Cracks-main/output/model", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 1: Preparing YOLO dataset...")
    print("=" * 60)
    dataset_dir = prepare_yolo_dataset(args.pos_dir, args.neg_dir, args.ann_dir, args.dataset_dir)
    
    print("\n" + "=" * 60)
    print("Step 2: Creating YAML configuration...")
    print("=" * 60)
    yaml_config = os.path.join(args.model_dir, 'data.yaml')
    create_yaml_config(dataset_dir, yaml_config)
    
    print("\n" + "=" * 60)
    print("Step 3: Training YOLOv8 model...")
    print("=" * 60)
    train_model(yaml_config, args.model_dir, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
