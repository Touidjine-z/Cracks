"""
Complete crack detection pipeline:
1. Auto-annotate positive images with masks (morphological segmentation)
2. Prepare dataset in YOLO format (train/val split)
3. Run inference and generate results (masks + bboxes)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil


def check_directories(pos_dir: str):
    """Verify that required directories exist."""
    if not os.path.isdir(pos_dir):
        print(f"✗ Error: {pos_dir} not found")
        return False
    return True


def run_step(step_name: str, script_name: str, args: list = None):
    """Run a Python script as a subprocess."""
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    
    cmd = [sys.executable, script_name] + (args or [])
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\n✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Complete crack detection pipeline (Auto-annotation → Dataset Prep → Inference)")
    parser.add_argument("--skip-annotate", action="store_true", help="Skip annotation step (assumes annotations exist)")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--only-infer", action="store_true", help="Only run inference (skip annotation & prepare)")
    parser.add_argument("--pos-dir", default="Cracks-main/dataset/positive", help="Positive images directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Inference confidence threshold")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("CRACK DETECTION PIPELINE - AUTOMATED")
    print("=" * 70)
    print(f"Positive images: {args.pos_dir}")
    print(f"Output: Cracks-main/output/")
    
    # Check directories
    if not check_directories(args.pos_dir):
        return
    
    # Step 1: Auto-annotate
    if not args.skip_annotate and not args.only_infer:
        ann_args = ["--pos-dir", args.pos_dir, "--output-dir", "Cracks-main/output/annotations"]
        if not run_step("1. AUTO-ANNOTATION (Mask → YOLO Labels)", "auto_annotate.py", ann_args):
            print("\n⚠ Continuing with existing annotations...")
    
    # Step 2: Prepare dataset
    if not args.skip_prepare and not args.only_infer:
        prep_args = [
            "--pos-dir", args.pos_dir,
            "--ann-dir", "Cracks-main/output/annotations",
            "--dataset-dir", "Cracks-main/output/dataset",
            "--model-dir", "Cracks-main/output/model"
        ]
        if not run_step("2. DATASET PREPARATION (Train/Val Split)", "prepare_dataset.py", prep_args):
            print("\n⚠ Continuing...")
    
    # Step 3: Run inference (using pre-generated masks or segmentation)
    test_args = [
        "--images", args.pos_dir,
        "--output", "Cracks-main/output/results",
        "--conf", str(args.conf)
    ]
    if not run_step("3. INFERENCE & VISUALIZATION (Masks + BBoxes)", "test_mask_inference.py", test_args):
        print("\n⚠ Inference step had issues, but results may still be available")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE ✓")
    print("=" * 70)
    print(f"\n📁 Results Location: Cracks-main/output/")
    print(f"   ├── annotations/        ← Auto-generated YOLO labels (.txt)")
    print(f"   ├── dataset/            ← Organized train/val split")
    print(f"   │   ├── images/train/")
    print(f"   │   ├── images/val/")
    print(f"   │   ├── labels/train/")
    print(f"   │   └── labels/val/")
    print(f"   ├── model/              ← Model config (data.yaml)")
    print(f"   │   └── model_info.json")
    print(f"   └── results/            ← Inference outputs")
    print(f"       ├── *_mask.png      ← Binary segmentation masks")
    print(f"       └── *_bbox.png      ← Images with bounding boxes + confidence")
    print(f"\n💡 To train YOLOv8, run:")
    print(f"   python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='Cracks-main/output/model/data.yaml', epochs=50)\"")


if __name__ == "__main__":
    main()
