"""
Main pipeline orchestrator - IMPROVED VERSION
Combines all steps: auto-annotation -> dataset prep -> inference
With better noise elimination and precise bounding boxes
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report results."""
    print(f"\n{description}")
    print("=" * 70)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False)
        if result.returncode != 0:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Crack Detection Pipeline")
    parser.add_argument('--skip-annotate', action='store_true', help='Skip auto-annotation step')
    parser.add_argument('--skip-prepare', action='store_true', help='Skip dataset preparation step')
    parser.add_argument('--only-infer', action='store_true', help='Only run inference')
    parser.add_argument('--pos-dir', default='Cracks-main/dataset/positive', help='Positive images directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 70)
    print("CRACK DETECTION PIPELINE - AUTOMATED (IMPROVED VERSION)")
    print("=" * 70)
    print(f"Positive images: {args.pos_dir}")
    print(f"Output: Cracks-main/output/")
    print("=" * 70)
    
    # Step 1: Auto-annotation
    if not args.skip_annotate and not args.only_infer:
        cmd = f'python auto_annotate_v2.py --pos-dir "{args.pos_dir}" --output-dir "Cracks-main/output/annotations"'
        if not run_command(cmd, "STEP: 1. AUTO-ANNOTATION (Morphological Segmentation)"):
            sys.exit(1)
    
    # Step 2: Dataset Preparation
    if not args.skip_prepare and not args.only_infer:
        cmd = f'python prepare_dataset.py --pos-dir "{args.pos_dir}"'
        if not run_command(cmd, "STEP: 2. DATASET PREPARATION (Train/Val Split)"):
            sys.exit(1)
    
    # Step 3: Inference & Visualization
    cmd = f'python test_mask_inference_v2.py --image-dir "{args.pos_dir}" --output-dir "Cracks-main/output/results"'
    if not run_command(cmd, "STEP: 3. INFERENCE & VISUALIZATION (Masks + BBoxes)"):
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE ✓")
    print("=" * 70)
    print("\n📁 Results Location: Cracks-main/output/")
    print("   ├── annotations/        ← Auto-generated YOLO labels (.txt)")
    print("   ├── dataset/            ← Organized train/val split")
    print("   │   ├── images/train/")
    print("   │   ├── images/val/")
    print("   │   ├── labels/train/")
    print("   │   └── labels/val/")
    print("   ├── model/              ← Model config (data.yaml)")
    print("   │   └── model_info.json")
    print("   └── results/            ← Inference outputs")
    print("       ├── *_mask.png      ← Binary segmentation masks")
    print("       └── *_bbox.png      ← Images with bounding boxes")
    print("\n💡 Features:")
    print("   ✓ Advanced noise elimination (bilateral filter)")
    print("   ✓ Precise crack-focused bounding boxes")
    print("   ✓ Small noise component filtering")
    print("   ✓ Morphological refinement (open/close operations)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
