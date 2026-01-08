# Visual Architecture & Workflow Guide

## System Architecture Diagram

```
INPUT IMAGE (any size)
        │
        ├─→ Grayscale Conversion
        │   └─→ H × W image
        │
        ├─→ Sliding Window Scanner (64×64, stride=16)
        │   ├─→ Patch 1 at (0, 0)
        │   ├─→ Patch 2 at (16, 0)
        │   ├─→ Patch 3 at (32, 0)
        │   └─→ ... ~100-200 patches total
        │
        ├─→ FOR EACH PATCH:
        │   │
        │   ├─→ [1] Canny Edge Detection
        │   │   └─→ Edge map (black & white)
        │   │
        │   ├─→ [2] HOG Feature Extraction
        │   │   ├─→ Compute gradients
        │   │   ├─→ Histogram orientation
        │   │   └─→ 2016-dim feature vector
        │   │
        │   ├─→ [3] SVM Classification
        │   │   ├─→ Normalize features
        │   │   ├─→ RBF kernel
        │   │   ├─→ Probability output
        │   │   └─→ Confidence score
        │   │
        │   └─→ [4] Threshold Filter
        │       └─→ Keep if conf > 0.5
        │
        ├─→ Detection List (candidate boxes)
        │   ├─→ Box 1: (x1, y1, w, h, conf=0.85)
        │   ├─→ Box 2: (x2, y2, w, h, conf=0.72)
        │   └─→ Box 3: (x3, y3, w, h, conf=0.68)
        │
        ├─→ Non-Maximum Suppression (NMS)
        │   ├─→ Remove overlapping boxes
        │   ├─→ Keep highest confidence
        │   └─→ Final list: [Box 1, Box 3]
        │
        └─→ OUTPUT: Bounding Boxes with Confidence
            ├─→ Visualization on original image
            └─→ Detection report


## Training Pipeline

```
TRAINING DATA
    │
    ├─→ Load Annotations (YOLO format)
    │   └─→ Convert normalized coords to pixels
    │
    ├─→ Extract Positive Samples (cracks)
    │   ├─→ From annotated regions
    │   ├─→ Resize to 64×64
    │   └─→ ~200-500 samples
    │
    ├─→ Extract Negative Samples (no cracks)
    │   ├─→ Random regions
    │   ├─→ Resize to 64×64
    │   └─→ ~600-1500 samples
    │
    ├─→ Feature Extraction (for all samples)
    │   ├─→ Canny edge detection
    │   ├─→ HOG descriptors
    │   └─→ Normalize with StandardScaler
    │
    ├─→ SVM Training
    │   ├─→ C=1.0, kernel='rbf'
    │   ├─→ Fit decision boundary
    │   └─→ Store probability model
    │
    └─→ SAVE MODEL (pickle)
        └─→ crack_detector_model.pkl (~2-5MB)


## Feature Extraction Detail

```
INPUT PATCH (64×64 grayscale)
    │
    ├─→ Canny Edge Detection
    │   ├─→ Gaussian blur (?)
    │   ├─→ Sobel gradients (dx, dy)
    │   ├─→ Non-maximum suppression
    │   ├─→ Double threshold (50, 150)
    │   └─→ Edge tracing
    │   Output: Binary edge map
    │
    └─→ HOG Descriptor
        │
        ├─→ Divide into cells (16×16 cells)
        │   └─→ Each cell: 16×16 pixels
        │   └─→ Total: 4×4 = 16 cells
        │
        ├─→ For each cell:
        │   ├─→ Compute pixel gradients
        │   ├─→ Histogram of orientations (9 bins)
        │   └─→ Result: 9 values per cell
        │
        ├─→ Block normalization (8×8 blocks)
        │   └─→ Normalize 4×4 blocks
        │
        └─→ FEATURE VECTOR
            └─→ 2016 dimensions


## Sliding Window Visualization

```
Image (640×480)

    0   64  128  192  256 ... 640 pixels (x)
    ┌───┬───┬───┬───┬────
    │[0]│[1]│[2]│[3]│...
    ├───┼───┼───┼───┼────
    │[4]│[5]│[6]│[7]│...
    ├───┼───┼───┼───┼────
    │[8]│[9]│...│
    ├───┼───┼───┼────
    │...│
    
    ┌─────────────────────┐
    │  Window position    │
    │  64×64 patch        │
    │  Stride: 16 pixels  │
    │  Total patches: ~175│
    └─────────────────────┘

y (pixels)


## Confidence Threshold Effect

```
0.3 Threshold (Many detections, lower precision)
    
    ┌─────────────────────────────────────┐
    │  ██ ██ ██ ██ ██ ██ ██ ██            │  65 detections
    │  ██ ██ ██ ██ ██ ██ ██ ██            │  (includes false positives)
    │  ██ ██ ██ ██ ██ ██ ██ ██            │
    └─────────────────────────────────────┘


0.5 Threshold (Balanced, default)
    
    ┌─────────────────────────────────────┐
    │     ██    ██    ██    ██            │  28 detections
    │           ██    ██    ██            │  (good balance)
    │                  ██                 │
    └─────────────────────────────────────┘


0.8 Threshold (Few detections, high precision)
    
    ┌─────────────────────────────────────┐
    │          ██         ██              │  5 detections
    │                                     │  (only confident boxes)
    │                                     │
    └─────────────────────────────────────┘


## NMS (Non-Maximum Suppression) Example

```
BEFORE NMS: Multiple overlapping boxes
    
    ┌─────────────────────────────────────┐
    │  ╔═════╗                            │
    │  ║ 0.9 ║  ╔═════╗                  │
    │  ║─────║  ║ 0.7 ║  ╔═════╗        │
    │  ║     ║  ║─────║  ║ 0.6 ║        │
    │  ╚═════╝  ║     ║  ║─────║        │
    │           ╚═════╝  ║     ║        │
    │                    ╚═════╝        │
    │                                    │
    │  3 detections (overlapping)        │
    └─────────────────────────────────────┘


AFTER NMS: Only best box kept
    
    ┌─────────────────────────────────────┐
    │  ╔═════╗                            │
    │  ║ 0.9 ║                           │
    │  ║─────║                           │
    │  ║     ║                           │
    │  ╚═════╝                           │
    │                                    │
    │  1 detection (highest confidence)   │
    └─────────────────────────────────────┘


## Performance Characteristics

```
TRAINING TIME vs SAMPLES

Time (min)
   |      
20 |         ●
   |        /
15 |       /
   |      /
10 |     ● 
   |    /
 5 |   ●
   |  /
 0 |●──────────────
   |  100 300 500 700 900 1100 Samples


DETECTION SPEED vs IMAGE SIZE

Speed (sec)
   |
15 |         ●
   |        /
10 |       ●
   |      /
 5 |   ●
   |  /
 0 |●──────────────
   | 320  640 1280 1920 Pixels (width)


## File Organization

```
Crack_project/
│
├── 📄 crack_detector_cv.py (main class)
│   └── CrackDetectorCV
│       ├── __init__
│       ├── load_annotations
│       ├── extract_hog_features
│       ├── train
│       ├── detect_cracks
│       ├── visualize_detections
│       └── _nms
│
├── 🚀 quick_start.py (quick interface)
│   ├── train_quick()
│   └── detect_image()
│
├── 📊 batch_utils.py (batch processing)
│   ├── batch_detect_and_visualize
│   ├── evaluate_speed
│   └── generate_report
│
├── 🎮 demo.py (interactive demos)
│   ├── demo_training
│   ├── demo_single_image
│   ├── demo_batch
│   ├── demo_confidence_analysis
│   ├── demo_edge_detection
│   └── demo_model_info
│
├── 📦 requirements.txt
├── 📖 README_CV.md
├── 📋 SETUP.md
├── ⚡ QUICK_REFERENCE.py
├── 📝 IMPLEMENTATION_SUMMARY.py
│
├── 🤖 crack_detector_model.pkl (generated)
│
└── 📁 Cracks-main/
    ├── annotations/ (YOLO format)
    ├── dataset/
    │   ├── positive/
    │   ├── negative/
    │   ├── train/
    │   └── val/
    └── ...


## Decision Tree Example

```
START DETECTION
   │
   ├─→ Patch at position (x, y)
   │   └─→ Extract 64×64 region
   │
   ├─→ Extract HOG features
   │   └─→ 2016-dimensional vector
   │
   ├─→ SVM Decision Function
   │   │
   │   ├─→ Is score > 0 ? [Not crack]
   │   │   └─→ SKIP this patch
   │   │
   │   └─→ Is score < 0 ? [Potential crack]
   │       └─→ Get probability
   │           │
   │           ├─→ Is prob > 0.5 ? [Confidence threshold]
   │           │   └─→ ADD to detections
   │           │
   │           └─→ Is prob < 0.5 ? [Not confident]
   │               └─→ SKIP this patch
   │
   └─→ Continue next patch


## Data Format

```
YOLO ANNOTATION FORMAT
========================

File: 11336_1.txt
Line: "0 0.500000 0.665198 1.000000 0.669604"

Meaning:
  class_id:  0 (crack)
  center_x:  0.500000 (50% from left)
  center_y:  0.665198 (66.5% from top)
  width:     1.000000 (100% of image width)
  height:    0.669604 (67% of image height)

Conversion to pixels (for 640×480 image):
  pixel_x1 = int((0.500 - 1.0/2) × 640) = 0
  pixel_x2 = int((0.500 + 1.0/2) × 640) = 640
  pixel_y1 = int((0.665 - 0.669/2) × 480) = 156
  pixel_y2 = int((0.665 + 0.669/2) × 480) = 480


DETECTION OUTPUT FORMAT
========================

Box: {
  'x': 100,              ← Top-left X coordinate
  'y': 150,              ← Top-left Y coordinate  
  'width': 64,           ← Bounding box width
  'height': 64,          ← Bounding box height
  'confidence': 0.85     ← Probability score (0-1)
}

Visualization:
  ┌─────────────────────┐
  │ (100,150)           │
  │ ╔═══════════════╗   │
  │ ║   [crack]     ║ 64│
  │ ║   conf:0.85   ║ px│
  │ ║               ║   │
  │ ╚═══════════════╝   │
  │         64 px       │
  └─────────────────────┘
```

This visual guide shows the complete flow from input image to final detections!
