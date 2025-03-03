import os
import json
import numpy as np
import shutil
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split

random_state = 42
np.random.seed(random_state)

def get_class_vector(mask_path):
    """Convierte una máscara en un vector binario indicando qué clases están presentes."""
    mask = np.array(Image.open(mask_path).convert("P"))
    class_vector = tuple((np.unique(mask) > 0).astype(int))  # Vector binario de presencia de clases
    return class_vector

def stratified_split(image_dir, mask_dir, output_dir, test_size=0.15, random_state=random_state):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train/masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val/images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val/masks"), exist_ok=True)

    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    assert len(images) == len(masks), "Número de imágenes y máscaras no coinciden."

    mask_paths = [os.path.join(mask_dir, f) for f in masks]
    class_vectors = [get_class_vector(m) for m in mask_paths]  # Obtiene etiquetas estratificadas
    class_labels = ["".join(map(str, v)) for v in class_vectors]  # Convertimos en string

    # Contamos la frecuencia de cada combinación
    class_counts = Counter(class_labels)

    # Detectamos clases con solo 1 instancia
    rare_classes = {k for k, v in class_counts.items() if v == 1}

    # Si hay clases raras, las reasignamos a la clase más parecida
    for i, label in enumerate(class_labels):
        if label in rare_classes:
            # Asignamos una clase similar o la más frecuente
            class_labels[i] = max(class_counts, key=class_counts.get)

    # División estratificada asegurando que no hay clases con 1 solo ejemplo
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        images, masks, test_size=test_size, random_state=random_state, stratify=class_labels
    )

    # Guardar datos de la división en un json:
    split_data = {
        "train": train_imgs,
        "test": test_imgs
    }
    with open("data_padded_strat/split_data.json", "w") as f:
        json.dump(split_data, f)

    # Copiar archivos a las carpetas correspondientes
    for img, mask in zip(train_imgs, train_masks):
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "train/images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "train/masks", mask))

    for img, mask in zip(test_imgs, test_masks):
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "val/images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "val/masks", mask))

    print("✅ División estratificada completada. Datos guardados en:", output_dir)

# Uso:
if __name__ == "__main__":
    # stratified_split("data_padded/images", "data_padded/masks", "data_padded_strat")   # Para dividir el conjunto original en un conjunto de train 85% y test 15%
    stratified_split("data_padded_strat/train/images", "data_padded_strat/train/masks", "data_padded_strat_for_training", test_size=0.1765)  # Para dividir el conjunto del 85% en train al 70% (del cjto. total de imagenes) y val al 15% (del cjto total de imgs).