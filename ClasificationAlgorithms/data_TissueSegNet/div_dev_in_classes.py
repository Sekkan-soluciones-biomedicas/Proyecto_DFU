import os
import shutil
import numpy as np
from PIL import Image

# Definir las rutas principales
train_dir = 'data_padded_stratified/train'
masks_dir = os.path.join(train_dir, 'masks')
images_dir = os.path.join(train_dir, 'images')

# Definir las nuevas carpetas y las clases correspondientes
new_dirs = {
    3: 'data_padded_stratified/p_class/train_callus',  # Clase 3: calloso
    2: 'data_padded_stratified/p_class/train_gran',    # Clase 2: granulación
    1: 'data_padded_stratified/p_class/train_fibrin'   # Clase 1: fibrina
}

# Crear las nuevas carpetas y sus subcarpetas
for dir_name in new_dirs.values():
    os.makedirs(os.path.join(dir_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir_name, 'masks'), exist_ok=True)

# Función para verificar si una máscara contiene una clase específica
def contains_class(mask_path, class_value):
    mask = np.array(Image.open(mask_path))
    return np.any(mask == class_value)

# Procesar cada máscara en train/masks
for mask_file in os.listdir(masks_dir):
    mask_path = os.path.join(masks_dir, mask_file)
    
    # Verificar cada clase y copiar archivos si corresponde
    for class_value, dir_name in new_dirs.items():
        if contains_class(mask_path, class_value):
            # Copiar la máscara a la subcarpeta masks
            shutil.copy(mask_path, os.path.join(dir_name, 'masks', mask_file))
            
            # Copiar la imagen correspondiente a la subcarpeta images
            image_file = mask_file  # Las máscaras e imágenes tienen el mismo nombre
            image_path = os.path.join(images_dir, image_file)
            shutil.copy(image_path, os.path.join(dir_name, 'images', image_file))

print("¡Procesamiento completado! Las nuevas carpetas han sido creadas.")