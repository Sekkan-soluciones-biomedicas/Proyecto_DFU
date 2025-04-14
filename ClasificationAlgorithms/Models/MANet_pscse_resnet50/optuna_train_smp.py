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
import segmentation_models_pytorch as smp
from metrics import check_metrics, DiceLoss, FocalLoss, WeightedSumOfLosses
from utils import get_loaders

# --------------- history ----------------------------------------

carpeta_origen = 'output_assets_model/Optuna'
carpeta_destino = 'output_assets_model/history_oam/Optuna'
os.makedirs(carpeta_destino, exist_ok=True)
for item in os.listdir(carpeta_origen):
    ruta_completa_item = os.path.join(carpeta_origen, item)
    if os.path.isfile(ruta_completa_item):
        shutil.copy(ruta_completa_item, carpeta_destino)
print("Archivos de oam copiados exitosamente al history.")

# ------------------- Parámetros de entrenamiento --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE, "is available \n ----------------------")

TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/train_b/images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/train_b/masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/val/images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/val/masks"

epochs_per_trial = int(input("Enter epochs per trial :"))
n_trials = int(input("Enter number of trials: "))
total_time_m = epochs_per_trial * n_trials * 8 / 60
print("El tiempo de total de búsqueda y entrenamiento será de aprox. ", round(total_time_m, 2), "minutos, (", round(total_time_m / 60, 2), "horas).")

csv_file = "output_assets_model/Optuna/trial_hyp_no_cv.csv"
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["trial_number", "learning_rate", "optimizer", "batch_size", "dropout_prob", "weight_decay", "mean_dice"])

csv_file_b = "output_assets_model/Optuna/trial_bad_hyp_no_cv.csv"
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
with open(csv_file_b, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["trial_number", "learning_rate", "optimizer", "batch_size", "dropout_prob", "weight_decay", "mean_dice_one_fold"])

# ------------------- Funciones de entrenamiento -------------------

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0
    num_batches = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float32)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def objective(trial):
    # Optimización de hiperparámetros
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_int('batch_size', 2, 8)
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 0.4)  # No se usa directamente en este modelo
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3) if optimizer_name == "AdamW" else None
    encoder_name = 'resnet34'  # Codificador fijo
    
    # Preprocesamiento específico del codificador
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')
    
    # Definición del modelo MAnet
    model = smp.MAnet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=3,  # Imágenes RGB
        classes=4,      # Número de clases en tu tarea de segmentación
        activation=None,  # Devolvemos logits para la función de pérdida
        # Para usar PSCSE, descomenta la siguiente línea:
        decoder_attention_type='pscse',
    ).to(DEVICE)
    
    # Definición de la función de pérdida
    w_dice = trial.suggest_float('w_dice', 0.0, 1.0)
    w_focal = 1.0 - w_dice
    dice_loss = DiceLoss(eps=1e-6)
    focal_loss = FocalLoss(gamma=2.0, alpha=None, reduction="mean")
    loss_fn = WeightedSumOfLosses(dice_loss, focal_loss, w1=w_dice, w2=w_focal)
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == 'Adam' else optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Transformaciones de datos
    train_transform = A.Compose([
        A.Resize(height=256, width=256),
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
        A.Lambda(image=preprocessing_fn),  # Preprocesamiento del codificador
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.Lambda(image=preprocessing_fn),  # Preprocesamiento del codificador
        ToTensorV2(),
    ])

    # Cargar datos
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        batch_size,
        train_transform,
        val_transforms,
        num_workers=0,
        pin_memory=True
    )
    
    # Entrenamiento y validación
    scaler = torch.amp.GradScaler('cuda')
    best_mean_dice = 0.0
    cnt_detect_zero_dice = 0
    cnt_patience = 0
    patience_in_study = 24
    zeros_patience = 20

    for epoch in range(epochs_per_trial):
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        dict_metrics_per_class = check_metrics(val_loader, model, device=DEVICE)
        epoch_mean_dice = np.mean(dict_metrics_per_class["dice_coefficient"])

        if any(dc == 0.0000 for dc in dict_metrics_per_class["dice_coefficient"]):
            cnt_detect_zero_dice += 1
            if cnt_detect_zero_dice == zeros_patience:
                print("Saltando trial debido a varios 0's consecutivos...")
                with open(csv_file_b, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([trial.number, learning_rate, optimizer_name, batch_size, dropout_prob, weight_decay, epoch_mean_dice])
                break
        else:
            cnt_detect_zero_dice = 0

        if epoch_mean_dice > best_mean_dice:
            best_mean_dice = epoch_mean_dice
            cnt_patience = 0
        else:
            cnt_patience += 1

        if cnt_patience > patience_in_study:
            raise optuna.TrialPruned()

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([trial.number, learning_rate, optimizer_name, batch_size, dropout_prob, weight_decay, best_mean_dice])

    return best_mean_dice

def main():
    study = optuna.create_study(
        study_name="MANet_study",
        storage="sqlite:///MANet_study.db",
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params = {
        "mean Dice value": trial.value,
        "Number of epochs per trial": epochs_per_trial,
        "Number of trials": n_trials,
        "params": trial.params
    }
    os.makedirs("output_assets_model/Optuna", exist_ok=True)
    with open("output_assets_model/Optuna/optuna_best_hyp_w_balanced_clas.json", "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()