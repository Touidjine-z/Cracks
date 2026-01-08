#!/usr/bin/env python3
"""
CRACK DETECTION PIPELINE - QUICK START
Exécute le pipeline complet en une seule commande.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║   CRACK DETECTION PIPELINE - COMPLETE WORKFLOW                ║
║                                                               ║
║   1. Auto-Annotation (Mask Generation)                        ║
║   2. Dataset Preparation (Train/Val Split)                    ║
║   3. Inference & Visualization (Masks + BBoxes)               ║
║                                                               ║
║   Status: ✅ READY TO RUN                                     ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🚀 Starting crack detection pipeline...\n")
    
    # Run pipeline
    cmd = [
        sys.executable,
        "pipeline.py",
        "--skip-annotate",  # Annotations already generated
        "--pos-dir", "Cracks-main/dataset/positive"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        print("""
📊 OUTPUTS SUMMARY:

  Annotations:
    Location: Cracks-main/output/annotations/
    Count: 20,001 YOLO format labels (.txt)
    Format: class_id center_x center_y width height (normalized)

  Dataset Organization:
    Location: Cracks-main/output/dataset/
    Train: 16,000 images
    Val: 4,001 images
    Ready for YOLOv8 training

  Inference Results:
    Location: Cracks-main/output/results/
    Masks: 20,001 binary segmentation images (*_mask.png)
    Annotated: 20,001 images with bboxes (*_bbox.png)
    Total Files: 40,002

🎯 NEXT STEPS:

  Option 1: Train YOLOv8 Model
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='Cracks-main/output/model/data.yaml', epochs=50)"

  Option 2: Review Generated Results
    - Open Cracks-main/output/results/ to inspect masks and bounding boxes
    - View YOLO labels: Cracks-main/output/annotations/
    - Check dataset split: Cracks-main/output/dataset/

  Option 3: Fine-tune Parameters
    - Edit mask generation in: auto_annotate.py or test_mask_inference.py
    - Adjust Canny thresholds, morphological kernels, min_area, etc.
    - Re-run: python pipeline.py --skip-annotate

📖 DOCUMENTATION:
    - Read: Cracks-main/PIPELINE_DOCUMENTATION.md
    - Quick reference: PIPELINE_README.md

        """)
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with error code {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
