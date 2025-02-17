import os
import numpy as np
from PIL import Image
import shutil

# Definir rutas
data_dir = './'  # Cambia esto si tu archivo .py está en una ruta diferente
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')

# Crear nuevas carpetas para las divisiones
train_images_dir = os.path.join(data_dir, 'train_images')
train_masks_dir = os.path.join(data_dir, 'train_masks')
val_images_dir = os.path.join(data_dir, 'val_images')
val_masks_dir = os.path.join(data_dir, 'val_masks')
test_images_dir = os.path.join(data_dir, 'test_images')
test_masks_dir = os.path.join(data_dir, 'test_masks')

# Crear directorios si no existen
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# Obtener todas las imágenes y máscaras
images = sorted(os.listdir(images_dir))  # Aseguramos que el orden sea consistente
masks = sorted(os.listdir(masks_dir))

# Verificamos que cada imagen tenga su máscara correspondiente
assert len(images) == len(masks), "El número de imágenes y máscaras no coincide"

# Asegurarse de que las imágenes y máscaras coinciden por nombre
for img, mask in zip(images, masks):
    assert img.split('.')[0] +'_mask' == mask.split('.')[0], f"Imagen y máscara no coinciden: {img} y {mask}"

# Mezclar datos aleatoriamente
indices = np.arange(len(images))
np.random.shuffle(indices)

# Calcular los tamaños de los conjuntos
train_size = int(0.84 * len(images))
val_size = int(0.08 * len(images))
test_size = len(images) - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Función para copiar archivos
def copy_files(indices, src_images_dir, src_masks_dir, dest_images_dir, dest_masks_dir):
    for idx in indices:
        img_file = images[idx]
        mask_file = masks[idx]

        img_path = os.path.join(src_images_dir, img_file)
        mask_path = os.path.join(src_masks_dir, mask_file)

        shutil.copy(img_path, os.path.join(dest_images_dir, img_file))
        shutil.copy(mask_path, os.path.join(dest_masks_dir, mask_file))

# Copiar archivos a las carpetas correspondientes
copy_files(train_indices, images_dir, masks_dir, train_images_dir, train_masks_dir)
copy_files(val_indices, images_dir, masks_dir, val_images_dir, val_masks_dir)
copy_files(test_indices, images_dir, masks_dir, test_images_dir, test_masks_dir)

print("División de datos completada.")
