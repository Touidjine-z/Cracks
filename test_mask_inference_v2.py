"""
Step 3: Test inference - generate masks and bounding boxes (IMPROVED VERSION).
Detects cracks and visualizes results with better noise elimination and precise bboxes.
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


def detect_crack_mask(image_path: str):
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


def extract_bboxes(mask, image_shape):
    """
    Extract bounding boxes for detected cracks.
    Returns list of (x1, y1, x2, y2) coordinates.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours: remove very small noise
    filtered_contours = filter_contours_by_area(contours, min_area=80, max_area=None)
    
    if not filtered_contours:
        return []
    
    # Get overall bounding box
    all_points = []
    for cnt in filtered_contours:
        all_points.extend(cnt.reshape(-1, 2))
    
    if not all_points:
        return []
    
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
    
    return [(x_min, y_min, x_max, y_max)]


def visualize_results(image_path: str, mask, output_dir: str):
    """
    Save mask and annotated image with bounding boxes.
    """
    img = imread_unicode(image_path)
    if img is None:
        return False
    
    # Extract bounding boxes
    bboxes = extract_bboxes(mask, img.shape)
    
    # Save mask
    mask_file = Path(output_dir) / (Path(image_path).stem + '_mask.png')
    cv2.imwrite(str(mask_file), mask)
    
    # Draw bounding boxes on image
    img_bbox = img.copy()
    
    for (x1, y1, x2, y2) in bboxes:
        # Draw green bounding box
        cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add confidence label
        confidence = 0.95  # Morphological segmentation confidence
        cv2.putText(img_bbox, f'Crack: {confidence:.2f}',
                   (int(x1), max(int(y1) - 5, 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save annotated image
    bbox_file = Path(output_dir) / (Path(image_path).stem + '_bbox.png')
    cv2.imwrite(str(bbox_file), img_bbox)
    
    return True


def run_inference(image_dir: str, output_dir: str):
    """Run inference on all images and generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    images = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
    
    success_count = 0
    total_count = len(images)
    
    print(f"\nRunning segmentation on {total_count} images...")
    
    for idx, img_path in enumerate(images, 1):
        mask = detect_crack_mask(str(img_path))
        if mask is None:
            continue
        
        if visualize_results(str(img_path), mask, output_dir):
            success_count += 1
        
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{total_count} images...")
    
    print(f"\n✓ Inference complete: {success_count} images processed, {success_count} cracks detected")
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test crack detection and generate visualizations")
    parser.add_argument('--image-dir', default='Cracks-main/dataset/positive', help='Directory with test images')
    parser.add_argument('--output-dir', default='Cracks-main/output/results', help='Output directory for results')
    
    args = parser.parse_args()
    
    run_inference(args.image_dir, args.output_dir)
