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

def stratified_split(image_dir, mask_dir, output_dir, test_size=0.15, val_size=0.15, random_state=random_state):
    """Realiza una división estratificada en train, val y test."""
    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train/masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val/images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val/masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test/images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test/masks"), exist_ok=True)

    # Listar imágenes y máscaras
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    assert len(images) == len(masks), "Número de imágenes y máscaras no coinciden."

    # Obtener vectores de clases para estratificación
    mask_paths = [os.path.join(mask_dir, f) for f in masks]
    class_vectors = [get_class_vector(m) for m in mask_paths]
    class_labels = ["".join(map(str, v)) for v in class_vectors]

    # Contar frecuencias de combinaciones de clases
    class_counts = Counter(class_labels)
    rare_classes = {k for k, v in class_counts.items() if v == 1}

    # Reasignar clases raras a la más frecuente
    for i, label in enumerate(class_labels):
        if label in rare_classes:
            class_labels[i] = max(class_counts, key=class_counts.get)

    # Primera división: train+val y test
    train_val_imgs, test_imgs, train_val_masks, test_masks, train_val_labels, test_labels = train_test_split(
        images, masks, class_labels, test_size=test_size, random_state=random_state, stratify=class_labels
    )

    # Segunda división: train y val a partir de train+val
    train_imgs, val_imgs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        train_val_imgs, train_val_masks, train_val_labels, 
        test_size=val_size / (1 - test_size), random_state=random_state, stratify=train_val_labels
    )

    # Guardar datos de la división en un JSON
    split_data = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }
    with open(os.path.join(output_dir, "split_data.json"), "w") as f:
        json.dump(split_data, f)

    # Copiar archivos a las carpetas correspondientes
    for img, mask in zip(train_imgs, train_masks):
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "train/images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "train/masks", mask))

    for img, mask in zip(val_imgs, val_masks):
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "val/images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "val/masks", mask))

    for img, mask in zip(test_imgs, test_masks):
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "test/images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "test/masks", mask))

    print("✅ División estratificada completada. Datos guardados en:", output_dir)

# Uso:
if __name__ == "__main__":
    # Ejemplo: dividir en train (70%), val (15%), test (15%)
    stratified_split("data_padded/images", "data_padded/masks", "data_padded_stratified", test_size=0.15, val_size=0.15)