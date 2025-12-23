"""
Script pour la détection de fissures en temps réel via caméra
Détecte les fissures et sauvegarde automatiquement les images avec détections
"""
from ultralytics import YOLO
import cv2
from pathlib import Path
from datetime import datetime
import argparse

class CrackDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialise le détecteur de fissures
        
        Args:
            model_path: Chemin vers le modèle YOLO entraîné
            confidence_threshold: Seuil de confiance pour les détections
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Modèle non trouvé: {self.model_path}")
        
        print(f"📥 Chargement du modèle: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print(f"✅ Modèle chargé avec succès!")
        
        # Dossier pour sauvegarder les détections
        self.output_dir = Path(__file__).parent / "detected_fissures"
        self.output_dir.mkdir(exist_ok=True)
        
        # Compteurs
        self.frame_count = 0
        self.detection_count = 0
    
    def detect_from_camera(self, camera_id: int = 0, save_all: bool = False):
        """
        Détection en temps réel depuis la caméra
        
        Args:
            camera_id: ID de la caméra (0 par défaut)
            save_all: Si True, sauvegarde tous les frames, sinon seulement ceux avec détections
        """
        print(f"\n🎥 Ouverture de la caméra {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la caméra {camera_id}")
            return
        
        # Configurer la résolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"✅ Caméra ouverte!")
        print(f"\n📊 Paramètres:")
        print(f"   Seuil de confiance: {self.confidence_threshold}")
        print(f"   Dossier de sauvegarde: {self.output_dir}")
        print(f"   Sauvegarde: {'Tous les frames' if save_all else 'Seulement les détections'}")
        
        print(f"\n🎮 Contrôles:")
        print(f"   's' - Sauvegarder le frame actuel")
        print(f"   'c' - Effacer les statistiques")
        print(f"   'q' - Quitter")
        print(f"\n🚀 Détection en cours...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Impossible de lire le frame")
                    break
                
                self.frame_count += 1
                
                # Détection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Annoter le frame
                annotated_frame = results[0].plot()
                
                # Vérifier s'il y a des détections
                has_detection = len(results[0].boxes) > 0
                
                if has_detection:
                    self.detection_count += 1
                
                # Ajouter les statistiques sur l'image
                stats_text = [
                    f"Frames: {self.frame_count}",
                    f"Detections: {self.detection_count}",
                    f"Confidence: {self.confidence_threshold:.2f}"
                ]
                
                y_offset = 30
                for i, text in enumerate(stats_text):
                    cv2.putText(annotated_frame, text, (10, y_offset + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Afficher le nombre de fissures détectées
                if has_detection:
                    num_cracks = len(results[0].boxes)
                    crack_text = f"FISSURES DETECTEES: {num_cracks}"
                    cv2.putText(annotated_frame, crack_text, (10, annotated_frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Afficher
                cv2.imshow("Detection de Fissures", annotated_frame)
                
                # Sauvegarder automatiquement si détection
                if has_detection or save_all:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"crack_{timestamp}_conf{int(self.confidence_threshold*100)}.jpg"
                    output_path = self.output_dir / filename
                    cv2.imwrite(str(output_path), annotated_frame)
                    
                    if has_detection:
                        # Afficher les infos de détection
                        for box in results[0].boxes:
                            conf = float(box.conf[0])
                            print(f"  ✅ Fissure détectée (confiance: {conf:.2%}) - Sauvegardée: {filename}")
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Arrêt de la détection...")
                    break
                    
                elif key == ord('s'):
                    # Sauvegarder manuellement
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"manual_{timestamp}.jpg"
                    output_path = self.output_dir / filename
                    cv2.imwrite(str(output_path), annotated_frame)
                    print(f"  💾 Frame sauvegardé manuellement: {filename}")
                    
                elif key == ord('c'):
                    # Effacer les stats
                    self.frame_count = 0
                    self.detection_count = 0
                    print("  🔄 Statistiques réinitialisées")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 Résumé:")
            print(f"   Frames traités: {self.frame_count}")
            print(f"   Détections: {self.detection_count}")
            if self.frame_count > 0:
                detection_rate = (self.detection_count / self.frame_count) * 100
                print(f"   Taux de détection: {detection_rate:.2f}%")
            print(f"   Images sauvegardées dans: {self.output_dir}")
    
    def detect_from_video(self, video_path: str):
        """
        Détection depuis un fichier vidéo
        
        Args:
            video_path: Chemin vers le fichier vidéo
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Vidéo non trouvée: {video_path}")
            return
        
        print(f"\n🎬 Ouverture de la vidéo: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo")
            return
        
        # Obtenir les infos de la vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✅ Vidéo ouverte!")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"\n🚀 Détection en cours...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Détection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Vérifier détections
                if len(results[0].boxes) > 0:
                    self.detection_count += 1
                    
                    # Sauvegarder
                    filename = f"video_frame_{self.frame_count}_cracks.jpg"
                    output_path = self.output_dir / filename
                    cv2.imwrite(str(output_path), annotated_frame)
                    
                    print(f"  ✅ Frame {self.frame_count}/{total_frames}: Fissure détectée - {filename}")
                
                # Afficher progression
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"  ⏳ Progression: {progress:.1f}% ({self.frame_count}/{total_frames} frames)")
                
                # Afficher
                cv2.imshow("Detection de Fissures - Video", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n✅ Traitement terminé!")
            print(f"   Frames traités: {self.frame_count}")
            print(f"   Détections: {self.detection_count}")

def main():
    parser = argparse.ArgumentParser(description="Détection de fissures en temps réel")
    parser.add_argument('--model', type=str, default='models/crack_detector.pt',
                        help='Chemin vers le modèle YOLO entraîné')
    parser.add_argument('--source', type=str, default='camera',
                        help='Source de détection: "camera" ou chemin vers une vidéo')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='ID de la caméra (défaut: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Seuil de confiance pour les détections (0.0-1.0)')
    parser.add_argument('--save-all', action='store_true',
                        help='Sauvegarder tous les frames, pas seulement les détections')
    
    args = parser.parse_args()
    
    try:
        detector = CrackDetector(args.model, confidence_threshold=args.conf)
        
        if args.source.lower() == 'camera':
            detector.detect_from_camera(camera_id=args.camera_id, save_all=args.save_all)
        else:
            detector.detect_from_video(args.source)
            
    except FileNotFoundError as e:
        print(f"\n{e}")
        print(f"\n💡 Astuce: Assurez-vous d'avoir entraîné le modèle d'abord avec:")
        print(f"   python train.py")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

if __name__ == "__main__":
    main()
