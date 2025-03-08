import os
import shutil
import numpy as np
from PIL import Image
import albumentations as A
import random

# Definir las transformaciones para augmentations
augment_transform = A.Compose([
    A.Rotate(limit=35, p=0.8),  # Rotación más probable
    A.HorizontalFlip(p=0.8),    # Volteo horizontal más probable
    A.VerticalFlip(p=0.3),      # Volteo vertical moderado
    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.3, border_mode=0),  # Traslación/escala/rotación
    A.OneOf([
        A.Perspective(p=0.2),
        A.GaussNoise(p=0.2),
        A.Sharpen(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
    ], p=0.6),  # Efectos ópticos
    A.OneOf([
        A.CLAHE(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
        A.RandomGamma(p=0.25),
        ], p=0.4),
    ], additional_targets={'mask': 'mask'})

# Definir una paleta de colores (RGB para 256 valores, ajustada a tus 4 clases)
palette = [
    0, 0, 0,      # 0: Negro (fondo)
    255, 0, 0,    # 1: Rojo (fibrina)
    0, 255, 0,    # 2: Verde (granulación)
    0, 0, 255     # 3: Azul (calloso)
] + [0, 0, 0] * 252  # Rellenar el resto con negro para llegar a 256 entradas

# Función para aplicar augmentations y guardar nuevas imágenes
def apply_augmentations(image_path, mask_path, num_augmentations, output_dir, class_name):
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path), dtype=np.uint8)  # Asegurar tipo uint8

    for i in range(num_augmentations):
        augmented = augment_transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        # Convertir la máscara a imagen con paleta
        mask_img = Image.fromarray(aug_mask, mode='P')
        mask_img.putpalette(palette)
        
        # Guardar las imágenes y máscaras generadas
        new_image_path = os.path.join(output_dir, 'images', f"{class_name}_aug_{i}_{os.path.basename(image_path)}")
        new_mask_path = os.path.join(output_dir, 'masks', f"{class_name}_aug_{i}_{os.path.basename(mask_path)}")
        
        Image.fromarray(aug_image).save(new_image_path)
        # Image.fromarray(aug_mask).save(new_mask_path)
        mask_img.save(new_mask_path)

# Función para obtener las clases presentes en una máscara
def get_classes_in_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    unique_classes = np.unique(mask)
    return set(unique_classes)

# Definir los directorios
train_dir = 'data_padded_stratified/train'
train_callus_dir = 'data_padded_stratified/p_class/train_callus'
train_gran_dir = 'data_padded_stratified/p_class/train_gran'
train_fibrin_dir = 'data_padded_stratified/p_class/train_fibrin'
train_b_dir = 'data_padded_stratified/train_b'

# Crear la estructura de carpetas en train_b
os.makedirs(os.path.join(train_b_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_b_dir, 'masks'), exist_ok=True)

# Copiar todas las imágenes y máscaras originales de train a train_b
for file in os.listdir(os.path.join(train_dir, 'images')):
    shutil.copy(os.path.join(train_dir, 'images', file), os.path.join(train_b_dir, 'images', file))
for file in os.listdir(os.path.join(train_dir, 'masks')):
    shutil.copy(os.path.join(train_dir, 'masks', file), os.path.join(train_b_dir, 'masks', file))

# Calcular cuántas imágenes adicionales se necesitan
num_per_clas = [len(os.listdir(os.path.join(class_dir, 'masks'))) for class_dir in [train_fibrin_dir, train_callus_dir, train_gran_dir]]
num_per_clas = sorted(num_per_clas, reverse=True)  # Cantidad de datos por clase en orden descendente
target_num = num_per_clas[0]
num_callus = num_per_clas[1]
num_fibrin = num_per_clas[2]
# target_num = 64
# num_callus = 58
# num_fibrin = 52

augment_fibrin = target_num - num_fibrin  # 13 imágenes
augment_callus = target_num - num_callus  # 8 imágenes

# Función para seleccionar imágenes con prioridad
def select_images_with_priority(mask_dir, target_class, avoid_class, num_to_select, random_state=None):
    if random_state is not None:
        random.seed(random_state)  # Fijar la semilla para reproducibilidad
    
    all_masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    selected = []
    
    # Prioridad 1: imágenes con solo la clase objetivo
    only_target = [f for f in all_masks if get_classes_in_mask(os.path.join(mask_dir, f)) == {0, target_class}]
    selected.extend(only_target[:num_to_select])
    
    # Prioridad 2: imágenes sin la clase a evitar
    if len(selected) < num_to_select:
        no_avoid_class = [f for f in all_masks if avoid_class not in get_classes_in_mask(os.path.join(mask_dir, f)) and f not in selected]
        selected.extend(no_avoid_class[:num_to_select - len(selected)])
    
    # Prioridad 3: selección aleatoria del resto si faltan
    if len(selected) < num_to_select:
        remaining = [f for f in all_masks if f not in selected]
        selected.extend(random.sample(remaining, num_to_select - len(selected)))
    
    return selected

# Seleccionar imágenes para fibrina (clase 1), evitando granulación (clase 2)
fibrin_masks = select_images_with_priority(os.path.join(train_fibrin_dir, 'masks'), target_class=1, avoid_class=2, num_to_select=augment_fibrin, random_state=42)

# Seleccionar imágenes para calloso (clase 3), evitando granulación (clase 2)
callus_masks = select_images_with_priority(os.path.join(train_callus_dir, 'masks'), target_class=3, avoid_class=2, num_to_select=augment_callus, random_state=42)

# Aplicar augmentations para fibrina (clase 1)
for mask_file in fibrin_masks:
    image_file = mask_file  # Asume que las imágenes tienen el mismo nombre que las máscaras
    image_path = os.path.join(train_fibrin_dir, 'images', image_file)
    mask_path = os.path.join(train_fibrin_dir, 'masks', mask_file)
    apply_augmentations(image_path, mask_path, 1, train_b_dir, 'fibrin')

# Aplicar augmentations para calloso (clase 3)
for mask_file in callus_masks:
    image_file = mask_file  # Asume que las imágenes tienen el mismo nombre que las máscaras
    image_path = os.path.join(train_callus_dir, 'images', image_file)
    mask_path = os.path.join(train_callus_dir, 'masks', mask_file)
    apply_augmentations(image_path, mask_path, 1, train_b_dir, 'callus')

print(f"✅ Proceso completado. Se generaron {augment_fibrin} imágenes para fibrina y {augment_callus} para calloso.")
print(f"Datos guardados en: {train_b_dir}")