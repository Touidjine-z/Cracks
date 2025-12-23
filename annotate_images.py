"""
Script pour annoter les images avec fissures en créant des bounding boxes
Support pour annotation manuelle et semi-automatique
"""
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple

class CrackAnnotator:
    def __init__(self, positive_dir: str):
        self.positive_dir = Path(positive_dir)
        self.annotations_dir = Path(__file__).parent / "annotations"
        self.annotations_dir.mkdir(exist_ok=True)
        
        self.current_image = None
        self.current_image_name = None
        self.bbox_start = None
        self.bbox_end = None
        self.bboxes = []
        self.drawing = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback pour dessiner des bounding boxes à la souris"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.bbox_start = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.current_image.copy()
                cv2.rectangle(img_copy, self.bbox_start, (x, y), (0, 255, 0), 2)
                cv2.imshow("Annotation", img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox_end = (x, y)
            
            # Ajouter le bounding box
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            
            # S'assurer que x1 < x2 et y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            self.bboxes.append([x1, y1, x2, y2])
            
            # Dessiner tous les bounding boxes
            img_copy = self.current_image.copy()
            for bbox in self.bboxes:
                cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imshow("Annotation", img_copy)
    
    def auto_detect_crack(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Détection semi-automatique des fissures pour créer un bounding box initial
        Utilise le traitement d'image pour détecter les zones sombres (fissures)
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Appliquer un filtre de flou
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Détection des contours (les fissures sont plus sombres)
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Morphologie pour relier les segments de fissures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        h, w = image.shape[:2]
        
        # Pour chaque contour, créer un bounding box si suffisamment grand
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filtrer les petits contours (bruit)
                x, y, w_box, h_box = cv2.boundingRect(contour)
                # Ajouter une marge
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + w_box + margin)
                y2 = min(h, y + h_box + margin)
                bboxes.append([x1, y1, x2, y2])
        
        # Si plusieurs petits bboxes, créer un bbox englobant
        if len(bboxes) > 0:
            x_min = min(b[0] for b in bboxes)
            y_min = min(b[1] for b in bboxes)
            x_max = max(b[2] for b in bboxes)
            y_max = max(b[3] for b in bboxes)
            return [[x_min, y_min, x_max, y_max]]
        
        # Si aucune fissure détectée, bbox sur toute l'image
        return [[0, 0, w, h]]
    
    def convert_to_yolo(self, bbox: List[int], img_width: int, img_height: int) -> str:
        """
        Convertit un bounding box [x1, y1, x2, y2] au format YOLO
        Format YOLO: class x_center y_center width height (normalisé)
        """
        x1, y1, x2, y2 = bbox
        
        # Calculer le centre et les dimensions
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Classe 0 = fissure
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def annotate_manual(self):
        """Annotation manuelle avec interface graphique"""
        images = list(self.positive_dir.glob("*.jpg"))
        
        if not images:
            print("❌ Aucune image trouvée dans dataset/positive/")
            return
        
        print(f"\n🖊️  Mode d'annotation MANUELLE")
        print("Instructions:")
        print("  • Cliquez et glissez pour dessiner un bounding box")
        print("  • Appuyez sur 's' pour sauvegarder et passer à l'image suivante")
        print("  • Appuyez sur 'u' pour annuler le dernier bounding box")
        print("  • Appuyez sur 'n' pour passer à l'image suivante sans sauvegarder")
        print("  • Appuyez sur 'q' pour quitter\n")
        
        for idx, img_path in enumerate(images):
            self.current_image = cv2.imread(str(img_path))
            self.current_image_name = img_path.stem
            self.bboxes = []
            
            cv2.namedWindow("Annotation")
            cv2.setMouseCallback("Annotation", self.mouse_callback)
            
            print(f"[{idx+1}/{len(images)}] Annotation de: {img_path.name}")
            cv2.imshow("Annotation", self.current_image)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # Sauvegarder
                    if self.bboxes:
                        self.save_annotations()
                        print(f"  ✅ {len(self.bboxes)} bounding box(es) sauvegardé(s)")
                    break
                    
                elif key == ord('u'):  # Annuler le dernier bbox
                    if self.bboxes:
                        self.bboxes.pop()
                        img_copy = self.current_image.copy()
                        for bbox in self.bboxes:
                            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.imshow("Annotation", img_copy)
                        print("  ↶ Dernier bounding box annulé")
                        
                elif key == ord('n'):  # Next sans sauvegarder
                    print("  ⏭️  Passé sans annotation")
                    break
                    
                elif key == ord('q'):  # Quitter
                    print("\n👋 Annotation terminée")
                    cv2.destroyAllWindows()
                    return
        
        cv2.destroyAllWindows()
        print(f"\n✅ Annotation de {len(images)} images terminée!")
    
    def annotate_auto(self):
        """Annotation semi-automatique avec possibilité d'ajustement manuel"""
        images = list(self.positive_dir.glob("*.jpg"))
        
        if not images:
            print("❌ Aucune image trouvée dans dataset/positive/")
            return
        
        print(f"\n🤖 Mode d'annotation SEMI-AUTOMATIQUE")
        print("Instructions:")
        print("  • Un bounding box est généré automatiquement")
        print("  • Appuyez sur 's' pour accepter et sauvegarder")
        print("  • Cliquez et glissez pour ajuster ou ajouter des bounding boxes")
        print("  • Appuyez sur 'r' pour réinitialiser et redessiner")
        print("  • Appuyez sur 'n' pour passer sans sauvegarder")
        print("  • Appuyez sur 'q' pour quitter\n")
        
        for idx, img_path in enumerate(images):
            self.current_image = cv2.imread(str(img_path))
            self.current_image_name = img_path.stem
            
            # Détection automatique
            self.bboxes = self.auto_detect_crack(self.current_image)
            
            # Afficher avec le bbox auto
            img_copy = self.current_image.copy()
            for bbox in self.bboxes:
                cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            cv2.namedWindow("Annotation")
            cv2.setMouseCallback("Annotation", self.mouse_callback)
            
            print(f"[{idx+1}/{len(images)}] Annotation de: {img_path.name}")
            cv2.imshow("Annotation", img_copy)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # Sauvegarder
                    self.save_annotations()
                    print(f"  ✅ {len(self.bboxes)} bounding box(es) sauvegardé(s)")
                    break
                    
                elif key == ord('r'):  # Réinitialiser
                    self.bboxes = []
                    cv2.imshow("Annotation", self.current_image)
                    print("  ↻ Réinitialisé")
                    
                elif key == ord('n'):  # Next sans sauvegarder
                    print("  ⏭️  Passé sans annotation")
                    break
                    
                elif key == ord('q'):  # Quitter
                    print("\n👋 Annotation terminée")
                    cv2.destroyAllWindows()
                    return
        
        cv2.destroyAllWindows()
        print(f"\n✅ Annotation de {len(images)} images terminée!")
    
    def save_annotations(self):
        """Sauvegarde les annotations au format YOLO"""
        if not self.bboxes or self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        
        # Sauvegarder au format YOLO (.txt)
        txt_path = self.annotations_dir / f"{self.current_image_name}.txt"
        with open(txt_path, 'w') as f:
            for bbox in self.bboxes:
                yolo_line = self.convert_to_yolo(bbox, w, h)
                f.write(yolo_line + '\n')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Annotation d'images de fissures")
    parser.add_argument('--mode', type=str, choices=['manual', 'auto'], default='auto',
                        help='Mode d\'annotation: manual (manuel) ou auto (semi-automatique)')
    parser.add_argument('--positive-dir', type=str, default='dataset/positive',
                        help='Dossier contenant les images positives')
    
    args = parser.parse_args()
    
    annotator = CrackAnnotator(args.positive_dir)
    
    if args.mode == 'manual':
        annotator.annotate_manual()
    else:
        annotator.annotate_auto()
    
    print(f"\n📁 Annotations sauvegardées dans: {annotator.annotations_dir}")
    print(f"   Prochaine étape: Exécuter prepare_yolo_dataset.py pour préparer le dataset d'entraînement")

if __name__ == "__main__":
    main()
