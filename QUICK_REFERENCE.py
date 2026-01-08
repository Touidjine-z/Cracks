#!/usr/bin/env python3
"""
Quick Reference Commands - Copy and paste these commands!
"""

print("""
╔════════════════════════════════════════════════════════════════╗
║   CLASSICAL CV CRACK DETECTION - QUICK REFERENCE COMMANDS     ║
╚════════════════════════════════════════════════════════════════╝

1. INSTALLATION
═══════════════════════════════════════════════════════════════

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, sklearn; print('✓ Ready!')"


2. TRAINING
═══════════════════════════════════════════════════════════════

# Quick train (5-10 min, 300 samples)
python quick_start.py --train

# Full train (10-20 min, 1000 samples)
python crack_detector_cv.py --train --max-samples 1000

# Train on custom dataset
python crack_detector_cv.py --train \\
  --annotations /path/to/annotations \\
  --images /path/to/images \\
  --max-samples 500


3. SINGLE IMAGE DETECTION
═══════════════════════════════════════════════════════════════

# Simple detection
python quick_start.py --detect image.jpg

# Detection with custom threshold
python quick_start.py --detect image.jpg --confidence 0.6

# Detection on full path
python quick_start.py --detect /full/path/to/image.jpg


4. BATCH DETECTION
═══════════════════════════════════════════════════════════════

# Detect in 10 images
python batch_utils.py --batch ./images --max-images 10

# Detect with custom output
python batch_utils.py --batch ./images \\
  --output ./results \\
  --confidence 0.5 \\
  --max-images 100

# Benchmark speed
python batch_utils.py --benchmark --max-images 20


5. TESTING & DEMO
═══════════════════════════════════════════════════════════════

# Run full demo (shows all features)
python demo.py

# Run specific demo
python -c "from demo import demo_single_image_detection; demo_single_image_detection()"


6. ADVANCED PYTHON USAGE
═══════════════════════════════════════════════════════════════

# In Python script or notebook:

from crack_detector_cv import CrackDetectorCV

# Create detector
detector = CrackDetectorCV()

# Train
positive, negative = detector.load_annotations('annotations/', 'images/')
detector.train(positive, negative)

# Detect
detector.load_model()
img, detections = detector.detect_cracks('test.jpg', confidence_threshold=0.5)

# Visualize
result = detector.visualize_detections(img, detections, 'output.jpg')

# Get detection info
for det in detections:
    print(f"Box: ({det['x']}, {det['y']}), "
          f"Size: {det['width']}x{det['height']}, "
          f"Confidence: {det['confidence']:.2%}")


7. COMMON TASKS
═══════════════════════════════════════════════════════════════

# Check if model exists
python -c "import os; print('✓ Model ready' if os.path.exists('crack_detector_model.pkl') else '✗ Train first')"

# Show model info
python -c "from demo import demo_model_info; demo_model_info()"

# Test on all images in folder
for f in Cracks-main/dataset/positive/*.jpg; do
  python quick_start.py --detect "$f"
done

# Generate detection report
python batch_utils.py --batch ./images --output ./results


8. PARAMETER TUNING
═══════════════════════════════════════════════════════════════

# Low confidence (more detections, lower precision)
python quick_start.py --detect image.jpg --confidence 0.3

# Medium confidence (balanced)
python quick_start.py --detect image.jpg --confidence 0.5

# High confidence (fewer detections, higher precision)
python quick_start.py --detect image.jpg --confidence 0.8


9. TROUBLESHOOTING
═══════════════════════════════════════════════════════════════

# Check Python version
python --version  # Need 3.7+

# Verify packages
pip list | grep -E 'opencv|scikit-learn|numpy'

# Test OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# Test scikit-learn
python -c "from sklearn.svm import SVC; print('✓ scikit-learn OK')"

# Check GPU (should show CPU only)
python -c "import cv2; print(cv2.getBuildInformation()[:500])"


10. FILE MANAGEMENT
═══════════════════════════════════════════════════════════════

# Save detections
mkdir -p results
python batch_utils.py --batch ./images --output ./results

# Clean detections
rm -rf results/
rm *.jpg  # Remove demo outputs

# Backup model
cp crack_detector_model.pkl crack_detector_model.pkl.backup

# Reset to fresh start
rm crack_detector_model.pkl  # Will retrain on next --train


11. PERFORMANCE BENCHMARKS
═══════════════════════════════════════════════════════════════

# Expected training time:
# 200 samples:   3-5 minutes
# 500 samples:   5-10 minutes
# 1000 samples:  10-20 minutes

# Expected detection time:
# 640×480:  2-5 seconds
# 1280×720: 5-10 seconds
# 1920×1080: 10-20 seconds

# Run benchmark
python batch_utils.py --benchmark --max-images 10


12. QUICK START TEMPLATE
═══════════════════════════════════════════════════════════════

#!/bin/bash
# Save as run_detection.sh and chmod +x

set -e  # Exit on error

echo "=== Crack Detection Pipeline ===="

# Check model
if [ ! -f crack_detector_model.pkl ]; then
    echo "Training model..."
    python quick_start.py --train
fi

# Detect
echo "Running detection..."
python quick_start.py --detect "$1" --confidence 0.5

# Report
echo "✓ Complete! Check output files."


13. PYTHON BATCH SCRIPT
═══════════════════════════════════════════════════════════════

#!/usr/bin/env python3
# Save as batch_detect.py

from pathlib import Path
from crack_detector_cv import CrackDetectorCV

detector = CrackDetectorCV()
detector.load_model()

for img_file in Path('images/').glob('*.jpg'):
    img, detections = detector.detect_cracks(str(img_file))
    print(f"{img_file.name}: {len(detections)} cracks")
    
    detector.visualize_detections(
        img, detections,
        output_path=f'results/{img_file.stem}_detected.jpg'
    )


14. TIPS & TRICKS
═══════════════════════════════════════════════════════════════

# Monitor training progress
# (Shows percentage and time estimate)
python quick_start.py --train

# Try multiple confidence thresholds
for conf in 0.3 0.5 0.7 0.9; do
  python quick_start.py --detect image.jpg --confidence $conf
done

# Profile speed
time python quick_start.py --detect image.jpg

# Use numpy for batch processing
import numpy as np
images = np.array([cv2.imread(f) for f in image_files])


═══════════════════════════════════════════════════════════════

For full details, see:
  - README_CV.md      (System overview)
  - SETUP.md          (Installation & setup)
  - crack_detector_cv.py (Source code)

═══════════════════════════════════════════════════════════════
""")

if __name__ == '__main__':
    # This script just prints the reference
    # No actual execution needed
    pass
