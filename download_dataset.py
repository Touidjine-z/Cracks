"""
Script pour télécharger et organiser le dataset de fissures depuis Kaggle
"""
import kagglehub
import os
import shutil
from pathlib import Path

def download_and_organize_dataset():
    """
    Télécharge le dataset depuis Kaggle et organise les images
    en dossiers positive/ et negative/
    """
    print("📥 Téléchargement du dataset depuis Kaggle...")
    
    # Télécharger le dataset
    path = kagglehub.dataset_download("arnavr10880/concrete-crack-images-for-classification")
    print(f"✅ Dataset téléchargé à: {path}")
    
    # Chemins de destination
    base_dir = Path(__file__).parent
    positive_dir = base_dir / "dataset" / "positive"
    negative_dir = base_dir / "dataset" / "negative"
    
    # Créer les dossiers si nécessaire
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    
    # Explorer le dataset téléchargé
    downloaded_path = Path(path)
    print(f"\n📁 Structure du dataset téléchargé:")
    
    # Rechercher les dossiers Positive et Negative
    positive_source = None
    negative_source = None
    
    for item in downloaded_path.rglob("*"):
        if item.is_dir():
            if "positive" in item.name.lower():
                positive_source = item
                print(f"  ✓ Trouvé: {item}")
            elif "negative" in item.name.lower():
                negative_source = item
                print(f"  ✓ Trouvé: {item}")
    
    # Copier les images
    if positive_source:
        print(f"\n📋 Copie des images positives (avec fissures)...")
        count = 0
        for img_file in positive_source.glob("*.jpg"):
            shutil.copy2(img_file, positive_dir / img_file.name)
            count += 1
        print(f"✅ {count} images positives copiées vers {positive_dir}")
    
    if negative_source:
        print(f"\n📋 Copie des images négatives (sans fissures)...")
        count = 0
        for img_file in negative_source.glob("*.jpg"):
            shutil.copy2(img_file, negative_dir / img_file.name)
            count += 1
        print(f"✅ {count} images négatives copiées vers {negative_dir}")
    
    # Résumé
    positive_count = len(list(positive_dir.glob("*.jpg")))
    negative_count = len(list(negative_dir.glob("*.jpg")))
    
    print(f"\n📊 Résumé:")
    print(f"  • Images avec fissures (positive): {positive_count}")
    print(f"  • Images sans fissures (negative): {negative_count}")
    print(f"  • Total: {positive_count + negative_count}")
    
    print(f"\n✅ Dataset organisé avec succès!")
    print(f"   Prochaine étape: Exécuter annotate_images.py pour créer les bounding boxes")

if __name__ == "__main__":
    download_and_organize_dataset()
