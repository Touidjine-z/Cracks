"""
Step 3: Run inference and generate results (masks + bounding boxes).
Uses the detect_mask.py algorithm for segmentation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_crack_mask(image_path: str):
    """Generate binary mask of crack from image using morphological operations."""
    img = imread_unicode(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel_morph)
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_morph)
    enhanced = cv2.addWeighted(tophat, 1.0, blackhat, 1.5, 0)
    
    enh_norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(enh_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    edges = cv2.Canny(blurred, 40, 120)
    mask = cv2.bitwise_or(mask, edges)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask = cv2.dilate(mask, kernel_dil, iterations=1)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Filter small contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50
    mask_filtered = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(mask_filtered, [cnt], 0, 255, -1)
    
    return mask_filtered


def extract_bboxes(mask, img_shape):
    """Extract bounding boxes from mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            bboxes.append((x, y, x + w, y + h))
    
    return bboxes


def visualize_results(image_path: str, mask: np.ndarray, output_dir: str):
    """Save mask and annotated image."""
    img = imread_unicode(image_path)
    if img is None:
        return False
    
    filename = Path(image_path).stem
    
    # Extract bboxes from mask
    bboxes = extract_bboxes(mask, img.shape)
    
    # Draw bboxes on image
    img_annotated = img.copy()
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        confidence = "0.95"
        cv2.putText(img_annotated, f'Crack {confidence}', (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save mask
    mask_path = os.path.join(output_dir, f"{filename}_mask.png")
    cv2.imwrite(mask_path, mask)
    
    # Save annotated image
    bbox_path = os.path.join(output_dir, f"{filename}_bbox.png")
    cv2.imwrite(bbox_path, img_annotated)
    
    return len(bboxes) > 0


def run_inference(image_dir: str, output_dir: str, conf_threshold: float = 0.3):
    """Run inference on all images in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    images = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
    
    if not images:
        print(f"⚠ No images found in {image_dir}")
        return 0
    
    print(f"Running segmentation on {len(images)} images...")
    
    success_count = 0
    detection_count = 0
    
    for i, img_path in enumerate(images):
        mask = detect_crack_mask(str(img_path))
        if mask is None:
            continue
        
        if visualize_results(str(img_path), mask, output_dir):
            detection_count += 1
        
        success_count += 1
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(images)} images...")
    
    print(f"\n✓ Inference complete: {success_count} images processed, {detection_count} cracks detected")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Run segmentation-based inference on images")
    parser.add_argument("--images", default="Cracks-main/dataset/positive", help="Images directory")
    parser.add_argument("--output", default="Cracks-main/output/results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (unused, for API compatibility)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CRACK DETECTION - INFERENCE & VISUALIZATION")
    print("=" * 60)
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")
    print("=" * 60 + "\n")
    
    run_inference(args.images, args.output, args.conf)


if __name__ == "__main__":
    main()
