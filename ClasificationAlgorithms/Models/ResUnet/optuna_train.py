import json
import shutil
import pandas as pd
import os
import zipfile
import torch
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import numpy as np
import torch.optim as optim
import optuna
from main import ResUnet
from metrics import check_metrics, dice_loss_multiclass, sum_of_losses
from utils import get_loaders

# --------------- history --------------------------

# Define las rutas
carpeta_origen = 'output_assets_model/Optuna'
carpeta_destino = 'output_assets_model/history_oam/Optuna'

# Crea la carpeta de destino si no existe
os.makedirs(carpeta_destino, exist_ok=True)

# Recorre todos los elementos en la carpeta origen
for item in os.listdir(carpeta_origen):
    # Obtén la ruta completa del elemento
    ruta_completa_item = os.path.join(carpeta_origen, item)
    
    # Verifica si es un archivo y no una carpeta
    if os.path.isfile(ruta_completa_item):
        # Copia el archivo a la carpeta de destino
        shutil.copy(ruta_completa_item, carpeta_destino)

print("Archivos de oam copiados exitosamente al history.")

# ------------------- Parámetros de entrenamiento --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE, "is available \n ----------------------")

# TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_images"
# TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_masks"
# VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_images"
# VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_masks"
# TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_strat_for_training/train/images"
# TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_strat_for_training/train/masks"
# VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_strat_for_training/val/images"
# VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_strat_for_training/val/masks"
TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/train_b/images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/train_b/masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/val/images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/val/masks"

# epochs_per_trial = 12
# n_trials = 50
epochs_per_trial = int(input("Enter epochs per trial :"))
n_trials = int(input("Enter number of trials: "))
total_time_m = epochs_per_trial*n_trials*8/60
print("El tiempo de total de búsqueda y entrenamiento será de aprox. ", round(total_time_m, 2),"minutos, (", round(total_time_m/60, 2), "horas).")

# Crear el archivo CSV y escribir los encabezados
csv_file = "output_assets_model/Optuna/trial_hyp_no_cv.csv"
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["trial_number", "learning_rate", "optimizer", "batch_size", "dropout_prob", "weight_decay", "mean_dice"])

# Crear el archivo CSV y escribir los encabezados
csv_file_b = "output_assets_model/Optuna/trial_bad_hyp_no_cv.csv"
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
with open(csv_file_b, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["trial_number", "learning_rate", "optimizer", "batch_size", "dropout_prob", "weight_decay", "mean_dice_one_fold"])

#------------------- Funciones de entrenamiento -------------------

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0
    num_batches = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward:
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def objective(trial):
    # Hyperparameter optimization
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_int('batch_size', 2, 8)
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 0.4)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3) if optimizer_name == "AdamW" else None

    model = ResUnet(in_channels=3, out_channels=4, dropout=dropout_prob).to(DEVICE)
    # loss_fn = dice_loss_multiclass
    loss_fn = sum_of_losses
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == 'Adam' else optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_transform = A.Compose([
        A.Resize(height=240, width=240),
        A.Rotate(limit=35, p=0.5),

        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
                A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Perspective(p=0.2),
                A.GaussNoise(p=0.2),
                A.Sharpen(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
            ],
            p=0.6,
        ),

        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.8,
        ),

        A.OneOf(
            [
                A.CLAHE(p=0.25),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                A.RandomGamma(p=0.25),
                A.HueSaturationValue(p=0.25),
            ],
            p=0.3,
        ),

        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(height=240, width=240),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        batch_size,
        train_transform,
        val_transforms,
        0,
        True
    )

    scaler = torch.amp.GradScaler('cuda')
    best_mean_dice = 0.0
    cnt_detect_zero_dice = 0

    for epoch in range(epochs_per_trial):  # Fixed number of epochs for optimization
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        dict_metrics_per_class = check_metrics(val_loader, model, device=DEVICE)
        epoch_mean_dice = np.mean(dict_metrics_per_class["dice_coefficient"])

        if any(dc == 0.0000 for dc in dict_metrics_per_class["dice_coefficient"]): # Parar el trial si se detecta frecuentemente un Dice de 0 para alguna clase.
            cnt_detect_zero_dice += 1
            if cnt_detect_zero_dice == 12:
                print("Saltando trial debido a varios 0's consecutivos...")
                with open(csv_file_b, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([trial.number, learning_rate, optimizer_name, batch_size, dropout_prob, weight_decay, epoch_mean_dice])
                break
        else:
            cnt_detect_zero_dice = 0  # Se reinicia el contador para que solo se cuenten los consecutivos.

        if epoch_mean_dice > best_mean_dice:
            best_mean_dice = epoch_mean_dice

        # Guardar los hiperparámetros y el mejor modelo hasta el momento en el archivo CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([trial.number, learning_rate, optimizer_name, batch_size, dropout_prob, weight_decay, best_mean_dice])

    return best_mean_dice

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Guardar los mejores hiperparámetros en un archivo JSON
    best_params = {
        "mean Dice value": trial.value,
        "Number of epochs per trial": epochs_per_trial,
        "Number of trials": n_trials,
        "params": trial.params
    }
    os.makedirs("output_assets_model/Optuna", exist_ok=True)
    with open("output_assets_model/Optuna/optuna_best_hyp_w_strat_data.json", "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()