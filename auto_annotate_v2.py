"""
Step 1: Auto-generate YOLO annotations from masks (IMPROVED VERSION).
For each positive image, extract mask -> bounding box -> YOLO format.
- Better noise elimination
- Focused bounding boxes on actual cracks
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


def denoise_image(image):
    """
    Apply advanced noise elimination before crack detection.
    """
    # Bilateral filter - preserves edges while removing noise
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply morphological opening to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    return denoised


def generate_mask(image_path: str):
    """Generate binary mask of crack from image with improved noise reduction."""
    img = imread_unicode(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Denoise the image first
    gray = denoise_image(gray)
    
    # Step 2: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Step 3: Subtle blur to further reduce noise
    blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    
    # Step 4: Morphological operations for crack enhancement
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel_tophat)
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_blackhat)
    
    # Weighted fusion - emphasize darker features (cracks)
    enhanced = cv2.addWeighted(tophat, 0.8, blackhat, 1.5, 0)
    
    # Step 5: Threshold with Otsu
    enh_norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(enh_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 6: Add Canny edges for thin crack detection
    edges = cv2.Canny(blurred, 50, 150)  # Adjusted thresholds
    mask = cv2.bitwise_or(mask, edges)
    
    # Step 7: Morphological refinement (reduce noise)
    # Remove small noise components
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Close gaps in cracks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))  # Elongated kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Slight dilation to enhance cracks
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    mask = cv2.dilate(mask, kernel_dil, iterations=1)
    
    return mask


def filter_contours_by_area(contours, min_area=100, max_area=None):
    """Filter contours by area to remove noise."""
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            if max_area is None or area <= max_area:
                filtered.append(cnt)
    return filtered


def get_crack_contours(mask):
    """
    Extract contours representing actual cracks.
    Filters by area to eliminate noise.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours: remove very small noise
    filtered_contours = filter_contours_by_area(contours, min_area=80, max_area=None)
    
    return filtered_contours


def contours_to_bbox(contours, image_shape):
    """
    Convert contours to a single bounding box that encompasses all cracks.
    Returns YOLO format (normalized).
    """
    if not contours:
        return None
    
    # Find the overall bounding box of all contours
    all_points = []
    for cnt in contours:
        all_points.extend(cnt.reshape(-1, 2))
    
    if not all_points:
        return None
    
    all_points = np.array(all_points)
    x_min = all_points[:, 0].min()
    x_max = all_points[:, 0].max()
    y_min = all_points[:, 1].min()
    y_max = all_points[:, 1].max()
    
    # Add small padding to focus better on crack
    pad_x = int((x_max - x_min) * 0.05)
    pad_y = int((y_max - y_min) * 0.05)
    
    x_min = max(0, x_min - pad_x)
    x_max = min(image_shape[1] - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(image_shape[0] - 1, y_max + pad_y)
    
    # Convert to YOLO format (normalized)
    h, w = image_shape[:2]
    cx = (x_min + x_max) / 2 / w
    cy = (y_min + y_max) / 2 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    
    # Clamp to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    bw = max(0, min(1, bw))
    bh = max(0, min(1, bh))
    
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
        
        # Get original image for shape
        img = imread_unicode(str(img_path))
        if img is None:
            continue
        
        # Extract contours of cracks
        contours = get_crack_contours(mask)
        
        # Convert to YOLO bbox
        bbox = contours_to_bbox(contours, img.shape)
        
        if bbox is None:
            continue
        
        # Save annotation file
        output_file = Path(output_dir) / (img_path.stem + '.txt')
        with open(output_file, 'w') as f:
            cx, cy, bw, bh = bbox
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        
        success_count += 1
        if success_count % 1000 == 0:
            print(f"✓ {success_count} images annotated...")
    
    print(f"✓ Auto-annotation complete: {success_count} images annotated")
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-annotate crack images in YOLO format")
    parser.add_argument('--pos-dir', default='Cracks-main/dataset/positive', help='Directory with positive images')
    parser.add_argument('--output-dir', default='Cracks-main/output/annotations', help='Output directory for YOLO labels')
    parser.add_argument('--class-id', type=int, default=0, help='Class ID for cracks')
    
    args = parser.parse_args()
    
    auto_annotate(args.pos_dir, args.output_dir, args.class_id)
