"""
Step 3: Test/Inference - Run YOLOv8 model on images and generate results.
Outputs: bounding boxes, masks, confidence scores.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from datetime import datetime


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def generate_mask_from_bbox(image_shape, bbox_norm):
    """Generate mask from normalized bounding box."""
    h, w = image_shape[:2]
    cx, cy, bw, bh = bbox_norm
    
    x_min = max(0, int((cx - bw/2) * w))
    x_max = min(w, int((cx + bw/2) * w))
    y_min = max(0, int((cy - bh/2) * h))
    y_max = min(h, int((cy + bh/2) * h))
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    
    return mask, (x_min, y_min, x_max, y_max)


def run_inference(model_path: str, image_dir: str, output_dir: str, conf_threshold: float = 0.3):
    """Run inference on images and save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print(f"⚠ Model path: {model_path}")
        return 0
    
    images = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
    
    if not images:
        print(f"⚠ No images found in {image_dir}")
        return 0
    
    print(f"Running inference on {len(images)} images...")
    
    results_list = []
    
    for img_path in images:
        img = imread_unicode(str(img_path))
        if img is None:
            continue
        
        # Inference
        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) == 0:
                continue
            
            # Draw boxes and create mask
            img_annotated = img.copy()
            mask_combined = np.zeros(img.shape[:2], dtype=np.uint8)
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().item()
                
                # Draw bounding box
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_annotated, f'{conf:.2f}', (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add to mask
                mask_combined[y1:y2, x1:x2] = 255
                
                results_list.append({
                    'image': img_path.name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
            
            # Save mask
            mask_path = os.path.join(output_dir, f"{img_path.stem}_mask.png")
            cv2.imwrite(mask_path, mask_combined)
            
            # Save annotated image
            bbox_path = os.path.join(output_dir, f"{img_path.stem}_bbox.png")
            cv2.imwrite(bbox_path, img_annotated)
            
            print(f"✓ {img_path.name} - {len(boxes)} crack(s) detected")
    
    print(f"\n✓ Inference complete: {len(results_list)} detections saved")
    return len(results_list)


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on images")
    parser.add_argument("--model", default="Cracks-main/output/model/crack_detector/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--images", default="Cracks-main/dataset/positive", help="Images directory")
    parser.add_argument("--output", default="Cracks-main/output/results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO Crack Detection - Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")
    print(f"Confidence threshold: {args.conf}")
    print("=" * 60 + "\n")
    
    run_inference(args.model, args.images, args.output, args.conf)


if __name__ == "__main__":
    main()
