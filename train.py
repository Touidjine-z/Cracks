"""
Script pour entraîner le modèle YOLOv8 sur le dataset de fissures
"""
from ultralytics import YOLO
from pathlib import Path
import argparse

def train_model(
    model_size: str = 'n',
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    patience: int = 20
):
    """
    Entraîne un modèle YOLOv8 pour la détection de fissures
    
    Args:
        model_size: Taille du modèle (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Nombre d'époques d'entraînement
        img_size: Taille des images d'entrée
        batch_size: Taille du batch
        patience: Nombre d'époques sans amélioration avant early stopping
    """
    print(f"🚀 Démarrage de l'entraînement YOLOv8-{model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {img_size}")
    print(f"   Batch size: {batch_size}\n")
    
    # Charger le modèle pré-entraîné YOLO
    model_name = f"yolov8{model_size}.pt"
    print(f"📥 Chargement du modèle pré-entraîné: {model_name}")
    model = YOLO(model_name)
    
    # Chemin vers le fichier de configuration
    base_dir = Path(__file__).parent
    data_yaml = base_dir / "dataset" / "data.yaml"
    
    if not data_yaml.exists():
        print(f"❌ Erreur: {data_yaml} n'existe pas!")
        print("   Veuillez d'abord exécuter prepare_yolo_dataset.py")
        return
    
    print(f"📂 Dataset config: {data_yaml}\n")
    
    # Entraîner le modèle
    print("🏋️  Entraînement en cours...\n")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,
        save=True,
        project=str(base_dir / "models"),
        name="crack_detector",
        exist_ok=True,
        # Augmentation de données
        degrees=10.0,          # Rotation
        translate=0.1,         # Translation
        scale=0.5,             # Zoom
        flipud=0.5,            # Flip vertical
        fliplr=0.5,            # Flip horizontal
        mosaic=1.0,            # Mosaic augmentation
        # Optimisations
        cache=True,            # Cache les images en RAM
        optimizer='AdamW',     # Optimiseur
        lr0=0.001,            # Learning rate initial
        lrf=0.01,             # Learning rate final
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Visualisation
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("✅ Entraînement terminé!")
    print("="*60)
    
    # Afficher les résultats
    print(f"\n📊 Résultats:")
    print(f"   Meilleur modèle: models/crack_detector/weights/best.pt")
    print(f"   Dernier modèle: models/crack_detector/weights/last.pt")
    
    # Copier le meilleur modèle
    best_model_path = base_dir / "models" / "crack_detector" / "weights" / "best.pt"
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, base_dir / "models" / "crack_detector.pt")
        print(f"   Modèle copié vers: models/crack_detector.pt")
    
    # Évaluation sur le set de validation
    print(f"\n📈 Évaluation sur le set de validation...")
    metrics = model.val()
    
    print(f"\n📊 Métriques de performance:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Précision: {metrics.box.mp:.4f}")
    print(f"   Rappel: {metrics.box.mr:.4f}")
    
    print(f"\n📁 Résultats détaillés dans: models/crack_detector/")
    print(f"   - Courbes d'entraînement: results.png")
    print(f"   - Matrice de confusion: confusion_matrix.png")
    print(f"   - Exemples de prédictions: val_batch*_pred.jpg")
    
    print(f"\n✅ Prochaine étape: Exécuter detect.py pour la détection en temps réel!")

def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle YOLO pour la détection de fissures")
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Taille du modèle (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Taille des images d\'entrée')
    parser.add_argument('--batch', type=int, default=16,
                        help='Taille du batch')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience pour early stopping')
    
    args = parser.parse_args()
    
    train_model(
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
