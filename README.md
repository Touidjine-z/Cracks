# 🔍 Détection de Fissures dans le Béton avec YOLO

Projet complet de détection automatique de fissures dans le béton utilisant YOLOv8 avec bounding boxes et détection en temps réel via caméra.

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Workflow complet](#-workflow-complet)
- [Résultats](#-résultats)
- [Personnalisation](#-personnalisation)

## ✨ Fonctionnalités

- ✅ Téléchargement automatique du dataset depuis Kaggle
- ✅ Annotation semi-automatique des images avec bounding boxes
- ✅ Entraînement d'un modèle YOLOv8 personnalisé
- ✅ Détection en temps réel via caméra
- ✅ Sauvegarde automatique des images avec détections
- ✅ Support de détection sur vidéos
- ✅ Interface graphique pour l'annotation

## 📁 Structure du projet

```
Cracks/
│
├── dataset/
│   ├── positive/           # Images avec fissures
│   ├── negative/           # Images sans fissures
│   ├── train/
│   │   ├── images/         # Images d'entraînement
│   │   └── labels/         # Labels YOLO d'entraînement
│   ├── val/
│   │   ├── images/         # Images de validation
│   │   └── labels/         # Labels YOLO de validation
│   └── data.yaml           # Configuration YOLO
│
├── annotations/            # Annotations brutes (format YOLO)
├── models/                 # Modèles entraînés
│   └── crack_detector.pt   # Meilleur modèle
├── detected_fissures/      # Images avec détections sauvegardées
│
├── download_dataset.py     # Script 1: Téléchargement du dataset
├── annotate_images.py      # Script 2: Annotation des images
├── prepare_yolo_dataset.py # Script 3: Préparation du dataset YOLO
├── train.py                # Script 4: Entraînement du modèle
├── detect.py               # Script 5: Détection en temps réel
├── requirements.txt        # Dépendances Python
└── README.md              # Ce fichier
```

## 🚀 Installation

### 1. Cloner le projet

```bash
cd Cracks
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer Kaggle

Pour télécharger le dataset depuis Kaggle, vous devez configurer votre API key:

1. Créez un compte sur [Kaggle](https://www.kaggle.com/)
2. Allez dans `Account > Create New API Token`
3. Téléchargez le fichier `kaggle.json`
4. Placez-le dans:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

## 📖 Utilisation

### Workflow complet

#### Étape 1: Télécharger le dataset

```bash
python download_dataset.py
```

Cela télécharge le dataset depuis Kaggle et organise les images dans `dataset/positive/` et `dataset/negative/`.

#### Étape 2: Annoter les images

##### Mode semi-automatique (recommandé)

```bash
python annotate_images.py --mode auto
```

Le script génère automatiquement des bounding boxes que vous pouvez ajuster.

**Contrôles:**
- `s` - Sauvegarder et passer à l'image suivante
- `r` - Réinitialiser l'annotation
- `n` - Passer sans sauvegarder
- `q` - Quitter

##### Mode manuel

```bash
python annotate_images.py --mode manual
```

Dessinez manuellement les bounding boxes en cliquant et glissant.

**Contrôles:**
- Clic + Glisser - Dessiner un bounding box
- `s` - Sauvegarder et passer à l'image suivante
- `u` - Annuler le dernier bounding box
- `n` - Passer sans sauvegarder
- `q` - Quitter

#### Étape 3: Préparer le dataset YOLO

```bash
python prepare_yolo_dataset.py
```

Divise les données en ensembles d'entraînement (80%) et de validation (20%).

#### Étape 4: Entraîner le modèle

##### Entraînement de base (modèle nano)

```bash
python train.py
```

##### Entraînement personnalisé

```bash
# Modèle plus grand (meilleure précision mais plus lent)
python train.py --model m --epochs 100 --batch 8

# Modèle léger (plus rapide)
python train.py --model n --epochs 50 --batch 16
```

**Options:**
- `--model`: Taille du modèle (n, s, m, l, x)
- `--epochs`: Nombre d'époques (défaut: 50)
- `--img-size`: Taille des images (défaut: 640)
- `--batch`: Taille du batch (défaut: 16)
- `--patience`: Early stopping (défaut: 20)

#### Étape 5: Détection en temps réel

##### Depuis la caméra

```bash
python detect.py --source camera
```

##### Depuis une vidéo

```bash
python detect.py --source chemin/vers/video.mp4
```

##### Options avancées

```bash
# Changer la caméra (si plusieurs caméras)
python detect.py --source camera --camera-id 1

# Ajuster le seuil de confiance
python detect.py --source camera --conf 0.7

# Sauvegarder tous les frames (pas seulement les détections)
python detect.py --source camera --save-all

# Utiliser un modèle spécifique
python detect.py --model models/crack_detector/weights/best.pt
```

**Contrôles pendant la détection:**
- `s` - Sauvegarder manuellement le frame actuel
- `c` - Réinitialiser les statistiques
- `q` - Quitter

## 📊 Résultats

Après l'entraînement, vous trouverez:

### Modèles entraînés
- `models/crack_detector.pt` - Meilleur modèle (copié automatiquement)
- `models/crack_detector/weights/best.pt` - Meilleur modèle original
- `models/crack_detector/weights/last.pt` - Dernier modèle

### Visualisations
- `models/crack_detector/results.png` - Courbes d'entraînement
- `models/crack_detector/confusion_matrix.png` - Matrice de confusion
- `models/crack_detector/val_batch*_pred.jpg` - Exemples de prédictions

### Détections
- `detected_fissures/` - Images avec fissures détectées

## 🎯 Métriques de performance

Le modèle est évalué sur:
- **mAP50**: Précision moyenne à IoU=0.50
- **mAP50-95**: Précision moyenne de IoU=0.50 à 0.95
- **Précision**: Proportion de vraies détections parmi toutes les détections
- **Rappel**: Proportion de fissures détectées parmi toutes les fissures

## 🔧 Personnalisation

### Modifier les classes

Éditez `dataset/data.yaml`:

```yaml
nc: 2  # Nombre de classes
names: ['fissure_fine', 'fissure_large']  # Noms des classes
```

### Ajuster l'augmentation de données

Modifiez les paramètres dans [train.py](train.py):

```python
degrees=10.0,      # Rotation
translate=0.1,     # Translation
scale=0.5,         # Zoom
flipud=0.5,        # Flip vertical
fliplr=0.5,        # Flip horizontal
```

### Changer le seuil de confiance

Pour la détection, utilisez `--conf`:

```bash
python detect.py --conf 0.3  # Plus sensible (plus de détections)
python detect.py --conf 0.8  # Plus strict (moins de fausses détections)
```

## 📝 Dataset source

Ce projet utilise le dataset **Concrete Crack Images for Classification** de Kaggle:
- **URL**: https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification
- **Images**: ~40,000 images (227x227 pixels)
- **Classes**: Positives (avec fissures) et Negatives (sans fissures)

## 🛠️ Dépannage

### Problème: "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### Problème: Caméra non détectée

```bash
# Essayer d'autres IDs de caméra
python detect.py --camera-id 1
python detect.py --camera-id 2
```

### Problème: Mémoire insuffisante pendant l'entraînement

```bash
# Réduire la taille du batch
python train.py --batch 4

# Utiliser un modèle plus petit
python train.py --model n --batch 8
```

### Problème: Dataset Kaggle non téléchargé

Vérifiez que votre fichier `kaggle.json` est bien placé dans `~/.kaggle/` et qu'il contient vos identifiants.

## 📚 Ressources

- [Documentation YOLOv8](https://docs.ultralytics.com/)
- [Tutorial YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Dataset Kaggle](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification)

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 👨‍💻 Auteur

Projet créé pour la détection automatique de fissures dans les structures en béton.

---

**Note**: Ce projet est à des fins éducatives et de recherche. Pour une utilisation en production, des ajustements et validations supplémentaires sont nécessaires.
