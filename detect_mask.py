"""
Simple crack mask + bounding box detector (OpenCV + NumPy only).
- Input: one image path
- Output: binary mask and boxed image in output/.
"""

import argparse
import os
import cv2
import numpy as np


def imread_unicode(path: str):
    # cv2.imread struggles with some unicode paths on Windows; use imdecode.
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def detect_crack(image_path: str, output_dir: str):
    ensure_dir(output_dir)

    img = imread_unicode(image_path)
    if img is None:
        raise RuntimeError(f"Impossible de charger l'image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Local contrast (helps thin cracks)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Enhance thin dark structures: combine top-hat and black-hat
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel_morph)
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_morph)

    enhanced = cv2.addWeighted(tophat, 1.0, blackhat, 1.5, 0)

    # Normalize and threshold (Otsu)
    enh_norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(enh_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Edge cue to recover missing thin parts
    edges = cv2.Canny(blurred, 40, 120)
    # keep edges thin (no dilation) to avoid over-thickening when fused
    mask = cv2.bitwise_or(mask, edges)

    # Clean mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    # very light dilation to connect fragmented crack pixels
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask = cv2.dilate(mask, kernel_dil, iterations=1)

    # Keep meaningful components (no top-k cap to avoid truncation of elongated cracks)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    min_area = 50  # filter small blobs to reduce overseg
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(cleaned, [c], -1, 255, thickness=cv2.FILLED)

    # Bounding box from cleaned mask
    ys, xs = np.where(cleaned > 0)
    boxed = img.copy()
    if len(xs) == 0:
        # Save empty mask and original image
        base = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(output_dir, f"{base}_mask.png")
        boxed_path = os.path.join(output_dir, f"{base}_bbox.png")
        cv2.imwrite(mask_path, cleaned)
        cv2.imwrite(boxed_path, boxed)
        return mask_path, boxed_path

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cv2.rectangle(boxed, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{base}_mask.png")
    boxed_path = os.path.join(output_dir, f"{base}_bbox.png")
    cv2.imwrite(mask_path, cleaned)
    cv2.imwrite(boxed_path, boxed)
    return mask_path, boxed_path


def main():
    parser = argparse.ArgumentParser(description="Crack mask + bounding box (OpenCV+NumPy only)")
    parser.add_argument("image", help="Chemin de l'image à traiter")
    parser.add_argument("--output", default="output", help="Dossier de sortie (par défaut: output)")
    args = parser.parse_args()

    mask_path, boxed_path = detect_crack(args.image, args.output)
    print(f"✓ Masque sauvegardé: {mask_path}")
    print(f"✓ Image avec bbox: {boxed_path}")


if __name__ == "__main__":
    main()
