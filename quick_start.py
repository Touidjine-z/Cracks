"""
Quick start script for crack detection training and inference
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crack_detector_cv import CrackDetectorCV

def train_quick(mode: str = 'classification', max_samples: int | None = 300, model_type: str = 'linear'):
    """
    Train the model.
    mode='classification' -> uses folder-based positives/negatives (no YOLO)
    mode='yolo' -> uses YOLO txt annotations to extract patches
    max_samples: for yolo, limit number of annotation files; for classification, limit number of positive images
    """
    detector = CrackDetectorCV(model_type=model_type)

    # Normalize max_samples: 0 -> None (use all)
    ms = None if (max_samples is None or max_samples == 0) else max_samples

    if mode == 'classification':
        pos_dir = "Cracks-main/dataset/positive"
        neg_dir = "Cracks-main/dataset/negative" if os.path.isdir("Cracks-main/dataset/negative") else None
        positive_samples, negative_samples = detector.load_classification_folders(
            pos_dir, neg_dir, max_pos=ms, max_neg=ms, patches_per_image=5
        )
    else:
        annotations_dir = "Cracks-main/annotations"
        images_dir = "Cracks-main/dataset/positive"
        positive_samples, negative_samples = detector.load_annotations(
            annotations_dir, images_dir, max_samples=ms
        )

    # Train model
    detector.train(positive_samples, negative_samples)
    print("\n✓ Model training complete! Saved as 'crack_detector_model.pkl'")


def detect_image(image_path, threshold=0.5, stride=8, nms_iou=0.3, max_det=50):
    """Detect cracks in a single image"""
    detector = CrackDetectorCV()
    detector.stride = stride
    detector.load_model()
    
    img, detections = detector.detect_cracks(
        image_path,
        confidence_threshold=threshold,
        stride=stride,
        nms_iou=nms_iou,
        max_detections=max_det,
    )
    
    if img is not None:
        vis = detector.visualize_detections(
            img, detections,
            output_path=image_path.replace('.jpg', '_cracks_detected.jpg')
        )
        
        print(f"\n✓ Detection complete!")
        print(f"  Found {len(detections)} cracks (threshold={threshold}, stride={stride}, nms_iou={nms_iou}, max_det={max_det})")
        for i, det in enumerate(detections):
            print(f"    {i+1}. Box: ({det['x']}, {det['y']}) "
                  f"Size: {det['width']}x{det['height']} "
                  f"Confidence: {det['confidence']:.2%}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Crack Detection Quick Start')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--detect', type=str, help='Detect cracks in image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--stride', type=int, default=8, help='Sliding window stride in pixels (default: 8)')
    parser.add_argument('--nms-iou', type=float, default=0.3, help='NMS IoU threshold (default: 0.3)')
    parser.add_argument('--max-det', type=int, default=50, help='Max detections to keep after NMS (default: 50)')
    parser.add_argument('--max-samples', type=int, default=300, help='Limit number of annotation files; 0 = all')
    parser.add_argument('--all', action='store_true', help='Use all annotations/images for training')
    parser.add_argument('--model-type', type=str, default='linear', choices=['linear', 'rbf'], help="Model type: 'linear' (fast) or 'rbf' (slower)")
    parser.add_argument('--mode', type=str, default='classification', choices=['classification', 'yolo'], help="Training mode: classification (folders) or yolo (txt boxes)")
    
    args = parser.parse_args()
    
    if args.train:
        # Determine training sample limit
        max_samples = None if args.all else args.max_samples
        if args.all:
            print("Training with ALL data (this may take longer)...")
        else:
            print(f"Training with up to {max_samples if max_samples else 'all'} items...")
        train_quick(mode=args.mode, max_samples=max_samples, model_type=args.model_type)
    elif args.detect:
        detect_image(
            args.detect,
            threshold=args.threshold,
            stride=args.stride,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
        )
    else:
        print("Usage:")
        print("  Train:  python quick_start.py --train")
        print("  Detect: python quick_start.py --detect <image_path> [--threshold 0.25] [--stride 8] [--nms-iou 0.3] [--max-det 50]")
