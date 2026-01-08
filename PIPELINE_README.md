# Crack Detection Pipeline - Complete Guide

## Overview
This pipeline automates crack detection in concrete/asphalt images using YOLOv8:

1. **Auto-Annotation**: Automatically generate YOLO labels from image masks (morphological segmentation)
2. **Training**: Train YOLOv8-nano model on annotated dataset
3. **Inference**: Test model and generate masks + bounding boxes with confidence scores

## Directory Structure

```
Cracks-main/
├── dataset/
│   ├── positive/        # Crack images (for detection training)
│   └── negative/        # Non-crack images (optional, improves robustness)
└── output/
    ├── annotations/     # Auto-generated YOLO labels (.txt)
    ├── dataset/         # Organized train/val split for YOLO
    ├── model/           # Trained YOLOv8 model and config
    └── results/         # Inference results (masks + bboxes)
```

## Quick Start

### Full Pipeline (Auto-annotation → Training → Testing)
```bash
python pipeline.py
```

### Individual Steps

#### Step 1: Auto-Generate Annotations from Masks
```bash
python auto_annotate.py --pos-dir Cracks-main/dataset/positive --output-dir Cracks-main/output/annotations
```
**Output**: YOLO format labels (.txt files with bounding boxes)

#### Step 2: Train YOLOv8 Model
```bash
python train_yolo.py \
  --pos-dir Cracks-main/dataset/positive \
  --neg-dir Cracks-main/dataset/negative \
  --ann-dir Cracks-main/output/annotations \
  --dataset-dir Cracks-main/output/dataset \
  --model-dir Cracks-main/output/model \
  --epochs 50
```
**Output**: Trained model at `Cracks-main/output/model/crack_detector/weights/best.pt`

#### Step 3: Run Inference
```bash
python test_yolo.py \
  --model Cracks-main/output/model/crack_detector/weights/best.pt \
  --images Cracks-main/dataset/positive \
  --output Cracks-main/output/results \
  --conf 0.3
```
**Output**: For each image:
- `{image}_mask.png` - Binary segmentation mask
- `{image}_bbox.png` - Original image with bounding boxes + confidence scores

## Parameters & Customization

### Auto-Annotation
- `--pos-dir`: Directory containing positive (crack) images
- `--output-dir`: Directory to save generated annotations

**Mask Generation Algorithm:**
1. Grayscale conversion + CLAHE contrast enhancement
2. Gaussian blur (5×5)
3. Morphological top-hat (11×11) + black-hat (11×11) fusion
4. Otsu auto-thresholding
5. Morphological open (3×3) + close (7×17 elongated) + dilate (3×5)
6. Canny edge fusion (thresholds 40, 120) for thin crack recovery
7. Contour filtering (min area = 50 pixels)

### Training
- `--epochs`: Number of training epochs (default: 50)
- `--dataset-dir`: Where to organize images for training
- `--model-dir`: Where to save trained model

**Model Configuration:**
- Architecture: YOLOv8-nano (lightweight, fast)
- Classes: 1 (crack)
- Train/Val split: 80/20
- Batch size: 16
- Image size: 640×640

### Inference
- `--conf`: Confidence threshold (0.0-1.0, default: 0.3)
  - Lower = more detections but more false positives
  - Higher = fewer detections but higher precision

## Output Format

### Annotations (YOLO format)
```
# Example: 00001.txt
0 0.5234 0.4127 0.3421 0.2156
# Format: <class_id> <center_x_norm> <center_y_norm> <width_norm> <height_norm>
```

### Inference Results
- **Masks**: Binary PNG showing crack regions (0 = background, 255 = crack)
- **Boxed Images**: Original image with:
  - Green bounding boxes around detected cracks
  - Confidence scores displayed
  
## Troubleshooting

### Model Not Found
Ensure you've completed the training step. Model should be at:
```
Cracks-main/output/model/crack_detector/weights/best.pt
```

### No Detections
- Lower the confidence threshold: `--conf 0.2`
- Ensure positive images are in `Cracks-main/dataset/positive/`
- Check that annotations were generated: `Cracks-main/output/annotations/`

### Out of Memory
- Reduce batch size in `train_yolo.py` (edit `batch=16` to smaller value)
- Use CPU instead of GPU (edit device parameter)

### Poor Detection Quality
1. Add more training samples (both positive and negative)
2. Tune confidence threshold during inference
3. Increase training epochs
4. Check mask generation quality in `auto_annotate.py`

## Environment Setup

Required packages (install via pip or conda):
```bash
pip install ultralytics opencv-python numpy pyyaml torch torchvision
```

Or use conda:
```bash
conda install -c conda-forge ultralytics opencv numpy pyyaml pytorch torchvision pytorch-cuda
```

## Pipeline Outputs Summary

| File | Purpose | Format |
|------|---------|--------|
| `Cracks-main/output/annotations/*.txt` | YOLO labels | Text (normalized bbox) |
| `Cracks-main/output/dataset/` | Organized train/val data | Folder structure for YOLO |
| `Cracks-main/output/model/crack_detector/` | Trained model | PyTorch format |
| `Cracks-main/output/results/*_mask.png` | Segmentation masks | Binary PNG |
| `Cracks-main/output/results/*_bbox.png` | Annotated images | Color PNG with boxes |

## Next Steps

1. Verify output files in `Cracks-main/output/`
2. Inspect generated masks for quality
3. Adjust confidence threshold if needed
4. Test on new images by running Step 3 with different `--images` directory
