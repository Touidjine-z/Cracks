"""
Demo script showing all features of the crack detection system
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from crack_detector_cv import CrackDetectorCV


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def demo_training():
    """Demo: Train the model"""
    print_header("DEMO 1: Training Crack Detector")
    
    print("\n1. Initializing detector...")
    detector = CrackDetectorCV()
    print("   ✓ CrackDetectorCV initialized")
    
    print("\n2. Loading annotations and images...")
    annotations_dir = "Cracks-main/annotations"
    images_dir = "Cracks-main/dataset/positive"
    
    positive_samples, negative_samples = detector.load_annotations(
        annotations_dir, images_dir, max_samples=200
    )
    
    print(f"\n3. Training SVM classifier...")
    detector.train(positive_samples, negative_samples)
    
    print("\n✓ Training demo complete!")
    return detector


def demo_single_image_detection():
    """Demo: Detect cracks in single image"""
    print_header("DEMO 2: Single Image Detection")
    
    detector = CrackDetectorCV()
    detector.load_model()
    
    # Try to find a test image
    test_image = "Cracks-main/dataset/positive/11336_1.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    print(f"\n1. Loading image: {test_image}")
    img_display = cv2.imread(test_image)
    print(f"   Image size: {img_display.shape}")
    
    print(f"\n2. Detecting cracks (confidence > 0.5)...")
    img, detections = detector.detect_cracks(test_image, confidence_threshold=0.5)
    
    print(f"\n3. Results:")
    print(f"   Found {len(detections)} crack regions")
    
    if detections:
        for i, det in enumerate(detections[:5]):  # Show first 5
            print(f"     {i+1}. Box: ({det['x']}, {det['y']}) "
                  f"Size: {det['width']}x{det['height']} "
                  f"Confidence: {det['confidence']:.2%}")
        if len(detections) > 5:
            print(f"     ... and {len(detections)-5} more")
    
    print(f"\n4. Visualizing detections...")
    vis = detector.visualize_detections(
        img, detections,
        output_path="demo_detection_output.jpg"
    )
    print(f"   ✓ Visualization saved to 'demo_detection_output.jpg'")
    
    print("\n✓ Single image detection demo complete!")


def demo_batch_detection():
    """Demo: Batch detection on multiple images"""
    print_header("DEMO 3: Batch Detection")
    
    detector = CrackDetectorCV()
    detector.load_model()
    
    batch_dir = "Cracks-main/dataset/positive"
    images = sorted(Path(batch_dir).glob('*.jpg'))[:5]  # First 5 images
    
    print(f"\n1. Processing {len(images)} images from {batch_dir}")
    
    os.makedirs("demo_batch_output", exist_ok=True)
    
    total_cracks = 0
    for i, img_file in enumerate(images):
        print(f"\n   [{i+1}/{len(images)}] {img_file.name}")
        
        img, detections = detector.detect_cracks(str(img_file), confidence_threshold=0.5)
        
        if img is not None:
            vis = detector.visualize_detections(
                img, detections,
                output_path=os.path.join("demo_batch_output", f"{img_file.stem}_detected.jpg")
            )
            print(f"      Found {len(detections)} cracks")
            total_cracks += len(detections)
    
    print(f"\n2. Summary:")
    print(f"   Total images processed: {len(images)}")
    print(f"   Total cracks found: {total_cracks}")
    print(f"   Output directory: demo_batch_output/")
    
    print("\n✓ Batch detection demo complete!")


def demo_confidence_analysis():
    """Demo: Analyze effect of confidence threshold"""
    print_header("DEMO 4: Confidence Threshold Analysis")
    
    detector = CrackDetectorCV()
    detector.load_model()
    
    test_image = "Cracks-main/dataset/positive/11336_1.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    print(f"\nAnalyzing detection confidence on: {test_image}")
    print(f"\nThreshold vs Detection Count:")
    print(f"{'Threshold':<12} {'Detections':<15} {'Relative':<10}")
    print("-" * 40)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    counts = []
    
    for threshold in thresholds:
        _, detections = detector.detect_cracks(test_image, confidence_threshold=threshold)
        counts.append(len(detections))
        relative = f"{100*threshold:.0f}%"
        print(f"{threshold:<12.1f} {len(detections):<15} {relative:<10}")
    
    print("\n✓ Analysis complete!")
    print("Note: Higher threshold = fewer but more confident detections")


def demo_edge_detection_visualization():
    """Demo: Show Canny edge detection preprocessing"""
    print_header("DEMO 5: Edge Detection Visualization")
    
    detector = CrackDetectorCV()
    
    test_image = "Cracks-main/dataset/positive/11336_1.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    print(f"\n1. Loading image...")
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    print(f"   Size: {img.shape}")
    
    print(f"\n2. Applying Canny edge detection...")
    edges = cv2.Canny(img, 50, 150)
    print(f"   Edges extracted")
    
    print(f"\n3. Saving visualizations...")
    cv2.imwrite("demo_original.jpg", img)
    cv2.imwrite("demo_edges.jpg", edges)
    print(f"   ✓ Original image saved to 'demo_original.jpg'")
    print(f"   ✓ Edge map saved to 'demo_edges.jpg'")
    
    # Count edge pixels
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_percentage = 100 * edge_pixels / total_pixels
    
    print(f"\n4. Statistics:")
    print(f"   Edge pixels: {edge_pixels} ({edge_percentage:.2f}%)")
    print(f"   Background: {total_pixels - edge_pixels} ({100-edge_percentage:.2f}%)")
    
    print("\n✓ Edge detection demo complete!")


def demo_model_info():
    """Demo: Show model information"""
    print_header("DEMO 6: Model Information")
    
    detector = CrackDetectorCV()
    
    model_path = detector.model_path
    
    print(f"\n1. Model Configuration:")
    print(f"   Window size: {detector.window_size[0]}×{detector.window_size[1]} pixels")
    print(f"   Feature descriptor: HOG (Histogram of Oriented Gradients)")
    print(f"   HOG cell size: 16×16 pixels")
    print(f"   Orientations: 9 bins")
    print(f"   Classifier: Support Vector Machine (SVM)")
    print(f"   Kernel: RBF (Radial Basis Function)")
    
    if os.path.exists(model_path):
        detector.load_model()
        file_size = os.path.getsize(model_path) / 1024 / 1024
        
        print(f"\n2. Trained Model:")
        print(f"   Path: {model_path}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Status: ✓ Ready for inference")
    else:
        print(f"\n2. Trained Model:")
        print(f"   Status: ✗ Not trained yet")
        print(f"   Run: python quick_start.py --train")
    
    print(f"\n3. Detection Parameters:")
    print(f"   Sliding window stride: 16 pixels")
    print(f"   NMS IOU threshold: 0.3")
    print(f"   Default confidence threshold: 0.5")
    
    print("\n✓ Model info demo complete!")


def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print("  CLASSICAL CV CRACK DETECTION SYSTEM - DEMO")
    print("="*70)
    
    print("\nThis demo showcases the crack detection system capabilities.")
    print("Make sure the model is trained before running detection demos.\n")
    
    print("Running demos...\n")
    
    # Demo 6: Model info (no model required)
    try:
        demo_model_info()
    except Exception as e:
        print(f"  ✗ Model info demo failed: {e}")
    
    # Demo 5: Edge detection (no model required)
    try:
        demo_edge_detection_visualization()
    except Exception as e:
        print(f"  ✗ Edge detection demo failed: {e}")
    
    # Check if model exists before running detection demos
    model_path = "crack_detector_model.pkl"
    if not os.path.exists(model_path):
        print_header("⚠ NOTICE")
        print(f"\nModel not found. Training is required for detection demos.")
        print(f"Run: python quick_start.py --train")
        print(f"\nAfter training, run this demo again for full features.\n")
        return
    
    # Demo 1: Training (optional - shown as reference)
    # demo_training()  # Uncomment to see training demo
    
    # Demo 2: Single image
    try:
        demo_single_image_detection()
    except Exception as e:
        print(f"  ✗ Single image demo failed: {e}")
    
    # Demo 3: Batch
    try:
        demo_batch_detection()
    except Exception as e:
        print(f"  ✗ Batch demo failed: {e}")
    
    # Demo 4: Confidence analysis
    try:
        demo_confidence_analysis()
    except Exception as e:
        print(f"  ✗ Confidence analysis demo failed: {e}")
    
    print_header("DEMO COMPLETE")
    print("\nAll demonstrations finished!")
    print("\nNext steps:")
    print("  - Review generated demo images")
    print("  - Try detection on your own images")
    print("  - Adjust confidence threshold for your use case")
    print("  - Run batch processing for large image sets")
    print("\n")


if __name__ == '__main__':
    main()
