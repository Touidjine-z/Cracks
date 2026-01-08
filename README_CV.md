# Classical Computer Vision Crack Detection

A **fast, CPU-only crack detection system** using classical computer vision (Canny edge detection, HOG features, SVM classifier). No deep learning required!

## Key Features

- ✅ **Fast Training**: Minutes on CPU (not hours with GPUs)
- ✅ **CPU Only**: Works on any machine without GPU
- ✅ **Classical CV**: Canny edges + HOG features + SVM
- ✅ **Bounding Box Output**: Precise crack localization
- ✅ **Non-Maximum Suppression**: Removes overlapping detections
- ✅ **Well-Documented**: Clean, readable code

## Method

1. **Edge Detection**: Canny edge detector to highlight cracks
2. **Feature Extraction**: HOG (Histogram of Oriented Gradients) on 64×64 patches
3. **Classification**: SVM classifier trained on positive/negative patches
4. **Detection**: Sliding window approach with confidence thresholding
5. **Post-Processing**: Non-maximum suppression to clean up results

## Installation

```bash
pip install opencv-python scikit-learn numpy tqdm
```

## Usage

### Train the Model (5-10 minutes)

```bash
python quick_start.py --train
```

This will:
- Load 300 samples from annotations
- Extract HOG features from Canny edges
- Train an SVM classifier
- Save model as `crack_detector_model.pkl`

### Detect Cracks in an Image

```bash
python quick_start.py --detect path/to/image.jpg
```

This will:
- Load the trained model
- Scan image with sliding window (64×64 patches)
- Detect cracks with confidence scores
- Draw bounding boxes on image
- Save result as `image_cracks_detected.jpg`

### Advanced Usage

```bash
# Train with more samples
python crack_detector_cv.py --train --max-samples 1000 \
  --annotations Cracks-main/annotations \
  --images Cracks-main/dataset/positive

# Detect with custom confidence threshold
python crack_detector_cv.py --detect image.jpg --confidence 0.6

# Batch detection on directory
python crack_detector_cv.py --detect Cracks-main/dataset/positive/
```

## Architecture Details

### CrackDetectorCV Class

```python
class CrackDetectorCV:
    window_size = (64, 64)        # Patch size
    
    # Methods:
    load_annotations()            # Load YOLO format annotations
    extract_hog_features()        # HOG from Canny edges
    train()                       # Train SVM classifier
    detect_cracks()              # Sliding window detection
    _nms()                       # Non-maximum suppression
    visualize_detections()       # Draw bounding boxes
```

### Processing Pipeline

```
Input Image
    ↓
Grayscale Conversion
    ↓
Sliding Window (64×64, stride=16)
    ↓
Canny Edge Detection
    ↓
HOG Feature Extraction
    ↓
SVM Classification
    ↓
Confidence Filtering
    ↓
Non-Maximum Suppression
    ↓
Bounding Boxes + Visualization
```

## Performance

**Training** (300 samples):
- ~5-10 minutes on CPU
- Memory: <500MB
- Model size: ~2-5MB

**Inference** (640×480 image):
- ~2-5 seconds per image
- Real-time on modern CPU

## Output Format

Each detection includes:
```python
{
    'x': 10,              # Top-left X coordinate
    'y': 20,              # Top-left Y coordinate
    'width': 64,          # Bounding box width
    'height': 64,         # Bounding box height
    'confidence': 0.85    # Confidence score (0-1)
}
```

## Advantages vs Deep Learning

| Feature | Classical CV | Deep Learning |
|---------|-------------|---------------|
| Training time | Minutes | Hours |
| GPU required | ❌ No | ✅ Yes |
| Model size | 2-5MB | 100-1000MB |
| CPU inference | Fast | Slow |
| Interpretability | High | Low |
| Memory usage | Low | High |

## Limitations

- Sensitivity to image quality and lighting
- Works best on horizontal/vertical cracks
- May miss very thin or very thick cracks
- No transfer learning capability

## Future Improvements

1. Multi-scale HOG features
2. Template matching for crack patterns
3. Morphological operations for post-processing
4. Ensemble with multiple classifiers
5. Parameter tuning based on crack types

## Files

- `crack_detector_cv.py` - Main detector class
- `quick_start.py` - Training and inference scripts
- `crack_detector_model.pkl` - Trained SVM model (generated)

## License

MIT

## References

- Canny Edge Detector: Canny, J. (1986). "A Computational Approach to Edge Detection"
- HOG Features: Dalal, N., & Triggs, B. (2005). "Histograms of Oriented Gradients for Visual Recognition"
- SVM: Vapnik, V. N. (1995). "The Nature of Statistical Learning Theory"
