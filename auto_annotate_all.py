"""
Script pour annoter automatiquement TOUTES les images sans interface graphique
Utilise la détection automatique pour générer les bounding boxes
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def auto_detect_crack(image: np.ndarray):
    """
    Détection automatique des fissures pour créer un bounding box
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image.shape[:2]
    bboxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + w_box + margin)
            y2 = min(h, y + h_box + margin)
            bboxes.append([x1, y1, x2, y2])
    
    if len(bboxes) > 0:
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)
        return [[x_min, y_min, x_max, y_max]]
    
    # Si aucune fissure détectée, bbox sur toute l'image
    return [[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]]

def convert_to_yolo(bbox, img_width, img_height):
    """Convertit un bounding box au format YOLO"""
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def annotate_all():
    base_dir = Path(__file__).parent
    positive_dir = base_dir / "dataset" / "positive"
    annotations_dir = base_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    
    images = list(positive_dir.glob("*.jpg"))
    
    print(f"\n🤖 Annotation automatique de {len(images)} images...")
    print("Cette opération peut prendre quelques minutes...\n")
    
    annotated_count = 0
    
    for img_path in tqdm(images, desc="Annotation en cours"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        bboxes = auto_detect_crack(image)
        
        # Sauvegarder au format YOLO
        txt_path = annotations_dir / f"{img_path.stem}.txt"
        with open(txt_path, 'w') as f:
            for bbox in bboxes:
                yolo_line = convert_to_yolo(bbox, w, h)
                f.write(yolo_line + '\n')
        
        annotated_count += 1
    
    print(f"\n✅ {annotated_count} images annotées avec succès!")
    print(f"📁 Annotations sauvegardées dans: {annotations_dir}")
    print(f"\n   Prochaine étape: Exécuter prepare_yolo_dataset.py")

if __name__ == "__main__":
    annotate_all()
