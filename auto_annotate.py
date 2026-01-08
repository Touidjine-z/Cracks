"""
Step 1: Auto-generate YOLO annotations from masks.
For each positive image, extract mask -> bounding box -> YOLO format.
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


def generate_mask(image_path: str):
    """Generate binary mask of crack from image."""
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
    
    return mask


def mask_to_bbox(mask, image_shape):
    """Extract bounding box from mask in YOLO format (normalized)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    h, w = image_shape[:2]
    cx = (x_min + x_max) / 2 / w
    cy = (y_min + y_max) / 2 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    
    return (cx, cy, bw, bh)


def auto_annotate(image_dir: str, output_dir: str, class_id: int = 0):
    """Generate YOLO annotations for all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    images = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
    
    success_count = 0
    for img_path in images:
        mask = generate_mask(str(img_path))
        if mask is None:
            continue
        
        img = imread_unicode(str(img_path))
        bbox = mask_to_bbox(mask, img.shape)
        
        if bbox is None:
            continue
        
        txt_name = img_path.stem + '.txt'
        txt_path = os.path.join(output_dir, txt_name)
        
        cx, cy, bw, bh = bbox
        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        
        success_count += 1
        print(f"✓ {img_path.name} -> {txt_name}")
    
    print(f"\n✓ Auto-annotation complete: {success_count} images annotated")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Auto-generate YOLO annotations from masks")
    parser.add_argument("--pos-dir", default="Cracks-main/dataset/positive", help="Positive images directory")
    parser.add_argument("--output-dir", default="Cracks-main/output/annotations", help="Output annotations directory")
    args = parser.parse_args()
    
    auto_annotate(args.pos_dir, args.output_dir)


if __name__ == "__main__":
    main()
