import os

# Chemin du dossier contenant les fichiers
folder_path = r"D:\MASTER 2\Crack_project\Images"

# Liste des fichiers dans le dossier
files = os.listdir(folder_path)

# Optionnel : trier les fichiers pour qu'ils soient dans l'ordre
files.sort()

# Boucle pour renommer chaque fichier
for i, filename in enumerate(files, start=1):
    # Obtenir l'extension du fichier
    ext = os.path.splitext(filename)[1]  # par exemple ".jpg", ".png"
    
    # Nouveau nom
    new_name = f"image_{i}{ext}"
    
    # Chemins complets
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)
    
    # Renommer le fichier
    os.rename(old_file, new_file)

print("Renommage terminé !")
