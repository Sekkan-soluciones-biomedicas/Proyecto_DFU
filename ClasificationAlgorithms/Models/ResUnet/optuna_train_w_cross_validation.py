import json
import shutil
import pandas as pd
import os
import zipfile
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import numpy as np
import torch.optim as optim
import optuna
from sklearn.model_selection import KFold
from main import ResUnet
from metrics import check_metrics, dice_loss_multiclass
from utils import get_loaders

# ------------------- Parámetros de entrenamiento --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE, "is available \n ----------------------")

TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_strat/train/images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_strat/train/masks"

epochs_per_trial = int(input("Enter epochs per trial :"))
n_trials = int(input("Enter number of trials: "))
n_splits = 5  # Número de folds en validación cruzada

#------------------- Función de entrenamiento -------------------

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
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_int('batch_size', 2, 8)
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3) if optimizer_name == "AdamW" else None

    model = ResUnet(in_channels=3, out_channels=4, dropout=dropout_prob).to(DEVICE)
    loss_fn = dice_loss_multiclass
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == 'Adam' else optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_transform = A.Compose([
        A.Resize(height=240, width=240),
        A.Rotate(limit=35, p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),

         A.OneOf(
                [
                    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
                    A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
                    A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
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
                p=0.5,
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

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_mean_dice = []

    img_filenames = sorted(os.listdir(TRAIN_IMG_DIR))
    mask_filenames = sorted(os.listdir(TRAIN_MASK_DIR))
    
    for train_idx, val_idx in kf.split(img_filenames):
        train_imgs = [img_filenames[i] for i in train_idx]
        train_masks = [mask_filenames[i] for i in train_idx]
        val_imgs = [img_filenames[i] for i in val_idx]
        val_masks = [mask_filenames[i] for i in val_idx]

        train_loader, val_loader = get_loaders(
            TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_IMG_DIR, TRAIN_MASK_DIR,  # Usamos la misma carpeta porque los nombres se filtran
            batch_size, train_transform, val_transforms, 0, True,
            train_imgs=train_imgs, train_masks=train_masks,
            val_imgs=val_imgs, val_masks=val_masks
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
                if cnt_detect_zero_dice == 10:
                    print("Saltando trial por varios 0's consecutivos...")
                    break

            if epoch_mean_dice > best_mean_dice:
                best_mean_dice = epoch_mean_dice

        all_mean_dice.append(best_mean_dice)

    return np.mean(all_mean_dice)  # Promedio de los valores de los k folds

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
    with open("output_assets_model/Optuna/optuna_best_hyp_w_cv.json", "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()
