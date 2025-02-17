""""Divide los datos en conjuntos de entrenamiento, validación y prueba a partir de las carpetas images y masks."""

import pandas as pd
from PIL import Image
import shutil
import numpy as np
import os
import io


# Definir rutas
data_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI" # Dirección de la carpeta Ims_double.
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')

# Definir los porcentajes de división de los datos (el test es del tamaño sobrante).
train_size_perc = 0.85
val_size_perc = 0.15

def div_data(data_dir=data_dir, images_dir=images_dir, masks_dir=masks_dir, train_size_perc=train_size_perc, val_size_perc=val_size_perc):
    # Crear nuevas carpetas para las divisiones
    test_size_perc = 1.0 - (train_size_perc + val_size_perc)
    train_images_dir = os.path.join(data_dir, 'train_images')
    train_masks_dir = os.path.join(data_dir, 'train_masks')
    val_images_dir = os.path.join(data_dir, 'val_images')
    val_masks_dir = os.path.join(data_dir, 'val_masks')
    test_images_dir = os.path.join(data_dir, 'test_images')
    test_masks_dir = os.path.join(data_dir, 'test_masks')

    # Borrar directorios si existen
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, test_images_dir, test_masks_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Crear directorios
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)

    # Obtener todas las imágenes y máscaras
    images = sorted(os.listdir(images_dir))  # Aseguramos que el orden sea consistente
    masks = sorted(os.listdir(masks_dir))

    # Verificamos que cada imagen tenga su máscara correspondiente (opcional)
    assert len(images) == len(masks), "El número de imágenes y máscaras no coincide"

    # Asegurarse de que las imágenes y máscaras coinciden por nombre (opcional)
    for img, mask in zip(images, masks):
        assert img.split('.')[0] == mask.split('.')[0], f"Imagen y máscara no coinciden: {img} y {mask}"

    # Mezclar datos aleatoriamente
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    # Calcular los tamaños de los conjuntos
    train_size = int(train_size_perc * len(images))
    val_size = int(val_size_perc * len(images))
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

    print("División de datos completada. Train_size: ", train_size, " . Val_size: ", val_size, " . Test_size: ", test_size )
    print("Porcentajes: ",int(train_size_perc*100),'-',int(val_size_perc*100),'-',int(round(test_size_perc*100)))

if __name__ == '__main__': # Solo se ejecuta este script (no se llama como un módulo), se ejecuta la función div_data().
    div_data()