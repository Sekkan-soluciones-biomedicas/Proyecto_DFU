""""Este script es para organizar la división del sataset de DFUTissue de acuerdo a sus archivos de texto
para obtener la misma división que ellos."""

import os
import shutil

def organizar_dataset_unet(
    dir_base="data_padded",
    archivo_train="labeled_train_names.txt",
    archivo_val="labeled_val_names.txt",
    archivo_test="test_names.txt",
    dir_salida="data_padded_dfutissue"
):
    # Crear estructura de carpetas
    subconjuntos = ['train', 'val', 'test']
    for subconjunto in subconjuntos:
        for tipo in ['images', 'masks']:
            os.makedirs(os.path.join(dir_salida, subconjunto, tipo), exist_ok=True)

    # Función auxiliar para procesar un archivo de nombres
    def copiar_imagenes_y_mascaras(archivo_nombres, subconjunto):
        with open(archivo_nombres, 'r') as f:
            nombres = [line.strip() for line in f.readlines()]
        
        for nombre in nombres:
            nombre_con_extension = f"{nombre}.png"

            # Ruta de origen
            path_img_src = os.path.join(dir_base, "images", nombre_con_extension)
            path_mask_src = os.path.join(dir_base, "masks", nombre_con_extension)

            # Ruta de destino
            path_img_dst = os.path.join(dir_salida, subconjunto, "images", nombre_con_extension)
            path_mask_dst = os.path.join(dir_salida, subconjunto, "masks", nombre_con_extension)

            # Copiar archivos si existen
            if os.path.exists(path_img_src) and os.path.exists(path_mask_src):
                shutil.copy(path_img_src, path_img_dst)
                shutil.copy(path_mask_src, path_mask_dst)
            else:
                print(f"Advertencia: archivo no encontrado para {nombre}")

    # Procesar los tres conjuntos
    copiar_imagenes_y_mascaras(archivo_train, "train")
    copiar_imagenes_y_mascaras(archivo_val, "val")
    copiar_imagenes_y_mascaras(archivo_test, "test")

    print("Organización de dataset completada.")



if __name__ == "__main__":
    # Definir rutas de los archivos de texto
    dir_base = "data_padded"
    archivo_train = "div_archivos_txt/labeled_train_names.txt"
    archivo_val = "div_archivos_txt/labeled_val_names.txt"
    archivo_test = "div_archivos_txt/test_names.txt"

    # Llamar a la función para organizar el dataset
    organizar_dataset_unet(
        dir_base=dir_base,
        archivo_train=archivo_train,
        archivo_val=archivo_val,
        archivo_test=archivo_test,
        dir_salida="data_padded_dfutissue"
    )