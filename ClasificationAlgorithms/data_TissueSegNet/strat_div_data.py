# import os
# import numpy as np
# import shutil
# from PIL import Image
# from sklearn.model_selection import train_test_split

# def get_class_distribution(mask_path):
#     """ Obtiene la distribución de clases en las máscaras """
#     masks = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.png')]
#     class_presence = []
    
#     for mask_file in masks:
#         mask = np.array(Image.open(mask_file).convert("P"), dtype=np.uint8)
#         classes = set(np.unique(mask))  # Clases presentes en la imagen
#         class_presence.append(classes)
    
#     return masks, class_presence

# def stratified_split(image_path, mask_path, output_dir, test_size=0.2, random_state=42):
#     """ Divide los datos asegurando una distribución similar de clases en train y test """
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "train/images"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "train/masks"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "test/images"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "test/masks"), exist_ok=True)
    
#     masks, class_presence = get_class_distribution(mask_path)
#     images = [os.path.join(image_path, os.path.basename(f)) for f in masks]
    
#     # Convertimos las clases a una lista de etiquetas (para sklearn)
#     labels = [tuple(sorted(c)) for c in class_presence]
    
#     train_imgs, test_imgs, train_masks, test_masks = train_test_split(
#         images, masks, test_size=test_size, stratify=labels, random_state=random_state
#     )
    
#     for img, mask in zip(train_imgs, train_masks):
#         shutil.copy(img, os.path.join(output_dir, "train/images", os.path.basename(img)))
#         shutil.copy(mask, os.path.join(output_dir, "train/masks", os.path.basename(mask)))
    
#     for img, mask in zip(test_imgs, test_masks):
#         shutil.copy(img, os.path.join(output_dir, "test/images", os.path.basename(img)))
#         shutil.copy(mask, os.path.join(output_dir, "test/masks", os.path.basename(mask)))
    
#     print("Datos divididos en train y test con distribución estratificada.")

# import os
# import numpy as np
# import shutil
# from PIL import Image
# from sklearn.model_selection import train_test_split

# def get_class_vector(mask_path):
#     """Convierte una máscara en un vector binario indicando qué clases están presentes."""
#     mask = np.array(Image.open(mask_path).convert("P"))
#     class_vector = tuple((np.unique(mask) > 0).astype(int))  # Convierte presencia de clases en tupla binaria
#     return class_vector

# def stratified_split(image_dir, mask_dir, output_dir, test_size=0.2, random_state=42):
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "train/images"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "train/masks"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "test/images"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "test/masks"), exist_ok=True)

#     images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
#     masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

#     assert len(images) == len(masks), "Número de imágenes y máscaras no coinciden."

#     mask_paths = [os.path.join(mask_dir, f) for f in masks]
#     class_vectors = [get_class_vector(m) for m in mask_paths]  # Obtiene etiquetas estratificadas

#     # Convertimos class_vectors a strings para que sean hashables y puedan usarse en train_test_split
#     class_labels = ["".join(map(str, v)) for v in class_vectors]

#     train_imgs, test_imgs, train_masks, test_masks = train_test_split(
#         images, masks, test_size=test_size, random_state=random_state, stratify=class_labels
#     )

#     # Copiar archivos a las carpetas correspondientes
#     for img, mask in zip(train_imgs, train_masks):
#         shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "train/images", img))
#         shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "train/masks", mask))

#     for img, mask in zip(test_imgs, test_masks):
#         shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "test/images", img))
#         shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "test/masks", mask))

#     print("✅ División estratificada completada. Datos guardados en:", output_dir)

# # Uso:
# if __name__ == "__main__":
#     stratified_split("data_padded/images", "data_padded/masks", "data_padded_strat")





import os
import numpy as np
import shutil
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split

def get_class_vector(mask_path):
    """Convierte una máscara en un vector binario indicando qué clases están presentes."""
    mask = np.array(Image.open(mask_path).convert("P"))
    class_vector = tuple((np.unique(mask) > 0).astype(int))  # Vector binario de presencia de clases
    return class_vector

def stratified_split(image_dir, mask_dir, output_dir, test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train/masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test/images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test/masks"), exist_ok=True)

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

    # Copiar archivos a las carpetas correspondientes
    for img, mask in zip(train_imgs, train_masks):
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "train/images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "train/masks", mask))

    for img, mask in zip(test_imgs, test_masks):
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "test/images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, "test/masks", mask))

    print("✅ División estratificada completada. Datos guardados en:", output_dir)

# Uso:
if __name__ == "__main__":
    stratified_split("data_padded/images", "data_padded/masks", "data_padded_strat")