"""
Script pour préparer le dataset YOLO à partir des annotations
Divise le dataset en train/val et copie les images et labels
"""
import shutil
from pathlib import Path
import random
from typing import List

def prepare_yolo_dataset(
    positive_dir: str = "dataset/positive",
    negative_dir: str = "dataset/negative",
    annotations_dir: str = "annotations",
    train_ratio: float = 0.8
):
    """
    Prépare le dataset YOLO en divisant les images en train/val
    """
    base_dir = Path(__file__).parent
    positive_path = base_dir / positive_dir
    negative_path = base_dir / negative_dir
    annotations_path = base_dir / annotations_dir
    
    train_images_dir = base_dir / "dataset" / "train" / "images"
    train_labels_dir = base_dir / "dataset" / "train" / "labels"
    val_images_dir = base_dir / "dataset" / "val" / "images"
    val_labels_dir = base_dir / "dataset" / "val" / "labels"
    
    # Créer les dossiers
    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Préparation du dataset YOLO...")
    
    # Collecter les images positives (avec annotations)
    positive_images = []
    for img_path in positive_path.glob("*.jpg"):
        annotation_file = annotations_path / f"{img_path.stem}.txt"
        if annotation_file.exists():
            positive_images.append(img_path)
    
    # Collecter les images négatives
    negative_images = list(negative_path.glob("*.jpg"))
    
    print(f"\n📊 Dataset:")
    print(f"  • Images positives annotées: {len(positive_images)}")
    print(f"  • Images négatives: {len(negative_images)}")
    
    # Mélanger et diviser
    random.seed(42)
    random.shuffle(positive_images)
    random.shuffle(negative_images)
    
    split_idx_pos = int(len(positive_images) * train_ratio)
    split_idx_neg = int(len(negative_images) * train_ratio)
    
    train_pos = positive_images[:split_idx_pos]
    val_pos = positive_images[split_idx_pos:]
    train_neg = negative_images[:split_idx_neg]
    val_neg = negative_images[split_idx_neg:]
    
    print(f"\n📂 Division train/val ({int(train_ratio*100)}/{int((1-train_ratio)*100)}):")
    print(f"  Train: {len(train_pos)} positives + {len(train_neg)} négatives = {len(train_pos) + len(train_neg)}")
    print(f"  Val:   {len(val_pos)} positives + {len(val_neg)} négatives = {len(val_pos) + len(val_neg)}")
    
    # Copier les images et labels pour train
    print(f"\n📋 Copie des fichiers d'entraînement...")
    copy_count = 0
    
    # Positives train
    for img_path in train_pos:
        shutil.copy2(img_path, train_images_dir / img_path.name)
        annotation_file = annotations_path / f"{img_path.stem}.txt"
        shutil.copy2(annotation_file, train_labels_dir / f"{img_path.stem}.txt")
        copy_count += 1
    
    # Négatives train (créer des fichiers .txt vides)
    for img_path in train_neg:
        shutil.copy2(img_path, train_images_dir / img_path.name)
        # Fichier label vide pour les images négatives
        (train_labels_dir / f"{img_path.stem}.txt").touch()
        copy_count += 1
    
    print(f"  ✅ {copy_count} images d'entraînement copiées")
    
    # Copier les images et labels pour validation
    print(f"\n📋 Copie des fichiers de validation...")
    copy_count = 0
    
    # Positives val
    for img_path in val_pos:
        shutil.copy2(img_path, val_images_dir / img_path.name)
        annotation_file = annotations_path / f"{img_path.stem}.txt"
        shutil.copy2(annotation_file, val_labels_dir / f"{img_path.stem}.txt")
        copy_count += 1
    
    # Négatives val
    for img_path in val_neg:
        shutil.copy2(img_path, val_images_dir / img_path.name)
        (val_labels_dir / f"{img_path.stem}.txt").touch()
        copy_count += 1
    
    print(f"  ✅ {copy_count} images de validation copiées")
    
    print(f"\n✅ Dataset YOLO préparé avec succès!")
    print(f"   Prochaine étape: Exécuter train.py pour entraîner le modèle")

if __name__ == "__main__":
    prepare_yolo_dataset()
