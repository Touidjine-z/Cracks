"""
Classical Computer Vision-based Crack Detection System
======================================================
Uses Canny edge detection, HOG features, and a CPU-only classifier.
Supports two training modes:
    - classification: build positives/negatives from image folders
    - yolo: extract patches from YOLO txt annotations (normalized coords)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import argparse
from tqdm import tqdm
import random
import glob

class CrackDetectorCV:
    """Classical CV-based crack detector using HOG + SVM"""
    
    def __init__(self, model_path='crack_detector_model.pkl', stride=16, model_type='linear'):
        self.model_path = model_path
        self.model = None
        self.window_size = (64, 64)  # Patch size for feature extraction
        self.hog = cv2.HOGDescriptor(self.window_size, (16, 16), (8, 8), (8, 8), 9)
        self.stride = stride  # Sliding window stride (pixels)
        self.model_type = model_type  # 'linear' (fast, CPU-parallel) or 'rbf' (slower)

    # ----------------------
    # Data loading utilities
    # ----------------------

    def load_classification_folders(self, pos_dir, neg_dir=None, max_pos=None, max_neg=None, patches_per_image=5):
        """
        Build positive/negative patch sets from folders (classification mode).

        - pos_dir: folder with images containing cracks
        - neg_dir: optional folder with non-crack images; if None, negatives are drawn
                   from random background patches of positive images
        - max_pos/max_neg: cap number of images (None for all)
        - patches_per_image: how many 64x64 patches to sample per image
        """
        positive_samples = []
        negative_samples = []

        pos_files = sorted(glob.glob(os.path.join(pos_dir, '*.jpg')) + glob.glob(os.path.join(pos_dir, '*.png')))
        if max_pos:
            pos_files = pos_files[:max_pos]

        neg_files = []
        if neg_dir:
            neg_files = sorted(glob.glob(os.path.join(neg_dir, '*.jpg')) + glob.glob(os.path.join(neg_dir, '*.png')))
            if max_neg:
                neg_files = neg_files[:max_neg]

        print(f"Loading classification data: {len(pos_files)} positive images, {len(neg_files)} negative images")

        # Positives: sample patches from each crack image
        for img_path in tqdm(pos_files, desc="PosImages"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            for _ in range(patches_per_image):
                x = random.randint(0, max(0, w - self.window_size[0]))
                y = random.randint(0, max(0, h - self.window_size[1]))
                patch = img[y:y+self.window_size[1], x:x+self.window_size[0]]
                if patch.shape == self.window_size:
                    positive_samples.append(patch)

        # Negatives: from neg_dir if provided, otherwise from positives background
        if neg_files:
            for img_path in tqdm(neg_files, desc="NegImages"):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                for _ in range(patches_per_image):
                    x = random.randint(0, max(0, w - self.window_size[0]))
                    y = random.randint(0, max(0, h - self.window_size[1]))
                    patch = img[y:y+self.window_size[1], x:x+self.window_size[0]]
                    if patch.shape == self.window_size:
                        negative_samples.append(patch)
        else:
            # If no explicit negatives, use background patches from positives
            for img_path in tqdm(pos_files, desc="NegFromPos"):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                for _ in range(patches_per_image):
                    x = random.randint(0, max(0, w - self.window_size[0]))
                    y = random.randint(0, max(0, h - self.window_size[1]))
                    patch = img[y:y+self.window_size[1], x:x+self.window_size[0]]
                    if patch.shape == self.window_size:
                        negative_samples.append(patch)

        print(f"Loaded classification patches -> Pos: {len(positive_samples)}, Neg: {len(negative_samples)}")
        return positive_samples, negative_samples
        
    def load_annotations(self, annotations_dir, images_dir, max_samples=None):
        """Load YOLO format annotations and image data (detection-mode patches)."""
        positive_samples = []
        negative_samples = []
        
        annotation_files = sorted(Path(annotations_dir).glob('*.txt'))
        if max_samples:
            annotation_files = annotation_files[:max_samples]
        
        print(f"Loading {len(annotation_files)} annotations...")
        
        for ann_file in tqdm(annotation_files):
            img_name = ann_file.stem + '.jpg'
            img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            h, w = img.shape
            
            # Read bounding boxes from annotation
            try:
                with open(ann_file) as f:
                    lines = f.readlines()
            except:
                continue
            
            if not lines:
                continue
            
            # Extract positive patches (cracks)
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO format: class_id center_x center_y width height (normalized)
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # Convert from normalized to pixel coordinates
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 - x1 >= self.window_size[0] and y2 - y1 >= self.window_size[1]:
                    # Extract patch
                    patch = img[y1:y2, x1:x2]
                    if patch.size > 0:
                        patch_resized = cv2.resize(patch, self.window_size)
                        positive_samples.append(patch_resized)
            
            # Extract negative patches (non-cracks)
            # Random patches away from crack regions
            for _ in range(3):  # 3 negative patches per image
                x = random.randint(0, max(0, w - self.window_size[0]))
                y = random.randint(0, max(0, h - self.window_size[1]))
                patch = img[y:y+self.window_size[1], x:x+self.window_size[0]]
                if patch.shape == self.window_size:
                    negative_samples.append(patch)
        
        print(f"Loaded {len(positive_samples)} positive samples")
        print(f"Loaded {len(negative_samples)} negative samples")
        
        return positive_samples, negative_samples
    
    def extract_hog_features(self, image):
        """Extract HOG features from image"""
        if image.shape != self.window_size:
            image = cv2.resize(image, self.window_size)
        
        # Apply Canny edge detection to enhance crack features
        edges = cv2.Canny(image, 50, 150)
        
        # Extract HOG from edges
        features = self.hog.compute(edges)
        return features.flatten()
    
    def _build_classifier(self):
        """Return a classifier pipeline according to model_type."""
        if self.model_type == 'linear':
            # Linear, CPU-parallel, good for large datasets.
            clf = SGDClassifier(
                loss='hinge',
                alpha=1e-4,
                max_iter=2000,
                class_weight='balanced',
                n_jobs=-1,
            )
        else:
            # Fallback to RBF SVM (single-core, slower).
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])

    def train(self, positive_samples, negative_samples):
        """Train classifier"""
        print("Extracting HOG features...")
        
        X = []
        y = []
        
        # Extract features from positive samples
        for sample in tqdm(positive_samples, desc="Positive"):
            features = self.extract_hog_features(sample)
            X.append(features)
            y.append(1)
        
        # Extract features from negative samples
        for sample in tqdm(negative_samples, desc="Negative"):
            features = self.extract_hog_features(sample)
            X.append(features)
            y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training SVM on {len(X)} samples...")
        
        # Train classifier pipeline
        self.model = self._build_classifier()
        self.model.fit(X, y)
        
        print("Training complete!")
        self.save_model()
    
    def save_model(self):
        """Save trained model"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.model_path}")
    
    def _score_to_confidence(self, score):
        """Convert decision_function score to pseudo-probability via sigmoid."""
        return 1.0 / (1.0 + np.exp(-score))

    def detect_cracks(self, image_path, confidence_threshold=0.5, stride=None, nms_iou=0.3, max_detections=None):
        """Detect cracks using sliding window"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None, []
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        
        detections = []
        step_size = stride if stride is not None else self.stride
        
        print(f"Scanning {image_path}...")
        
        for y in range(0, h - self.window_size[1], step_size):
            for x in range(0, w - self.window_size[0], step_size):
                patch = img_gray[y:y+self.window_size[1], x:x+self.window_size[0]]
                
                if patch.shape != self.window_size:
                    continue
                
                # Extract features
                features = self.extract_hog_features(patch).reshape(1, -1)
                
                # Get prediction and confidence
                pred = self.model.predict(features)[0]
                if hasattr(self.model, 'predict_proba'):
                    conf = self.model.predict_proba(features)[0][1]
                else:
                    score = self.model.decision_function(features)[0]
                    conf = float(self._score_to_confidence(score))
                
                if pred == 1 and conf >= confidence_threshold:
                    detections.append({
                        'x': x,
                        'y': y,
                        'width': self.window_size[0],
                        'height': self.window_size[1],
                        'confidence': conf
                    })
        
        # Apply non-maximum suppression
        detections = self._nms(detections, iou_threshold=nms_iou)

        # Optionally cap number of detections (keep highest confidence)
        if max_detections is not None and len(detections) > max_detections:
            detections = detections[:max_detections]
        
        print(f"Found {len(detections)} cracks")
        
        return img, detections
    
    def _nms(self, detections, iou_threshold=0.3):
        """Non-Maximum Suppression to remove overlapping boxes"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping boxes
            remaining = []
            for det in detections:
                iou = self._compute_iou(current, det)
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union"""
        x1_min, y1_min = box1['x'], box1['y']
        x1_max = x1_min + box1['width']
        y1_max = y1_min + box1['height']
        
        x2_min, y2_min = box2['x'], box2['y']
        x2_max = x2_min + box2['width']
        y2_max = y2_min + box2['height']
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def visualize_detections(self, image, detections, output_path=None):
        """Draw bounding boxes on image"""
        img_vis = image.copy()
        
        for det in detections:
            x, y = det['x'], det['y']
            w, h = det['width'], det['height']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence score
            text = f"{conf:.2f}"
            cv2.putText(img_vis, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img_vis)
            print(f"Visualization saved to {output_path}")
        
        return img_vis


def main():
    parser = argparse.ArgumentParser(description='Classical CV Crack Detection')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--detect', type=str, help='Detect cracks in image')
    parser.add_argument('--max-samples', type=int, default=500, 
                       help='Max samples for training')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detection')
    parser.add_argument('--annotations', type=str, 
                       default='Cracks-main/annotations',
                       help='Path to annotations directory')
    parser.add_argument('--images', type=str,
                       default='Cracks-main/dataset/positive',
                       help='Path to images directory')
    
    args = parser.parse_args()
    
    detector = CrackDetectorCV()
    
    if args.train:
        print("Training crack detector...")
        positive_samples, negative_samples = detector.load_annotations(
            args.annotations, args.images, max_samples=args.max_samples
        )
        detector.train(positive_samples, negative_samples)
    
    elif args.detect:
        print("Loading model...")
        detector.load_model()
        
        if os.path.isfile(args.detect):
            img, detections = detector.detect_cracks(args.detect, args.confidence)
            if img is not None:
                vis = detector.visualize_detections(
                    img, detections, 
                    output_path=args.detect.replace('.jpg', '_detected.jpg')
                )
                print(f"\nDetected {len(detections)} cracks")
                for i, det in enumerate(detections):
                    print(f"  {i+1}. Confidence: {det['confidence']:.2f}")
        
        else:
            # Batch detection on directory
            print(f"Detecting cracks in {args.detect}")
            for img_file in sorted(Path(args.detect).glob('*.jpg'))[:10]:  # First 10
                img, detections = detector.detect_cracks(str(img_file), args.confidence)
                if img is not None:
                    detector.visualize_detections(
                        img, detections,
                        output_path=str(img_file.parent / f"{img_file.stem}_detected.jpg")
                    )


if __name__ == '__main__':
    main()
