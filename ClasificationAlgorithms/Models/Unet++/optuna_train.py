import json
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
from main import UnetPlusPlus
from metrics import check_metrics, dice_loss_multiclass
from utils import get_loaders

# ------------------- Parámetros de entrenamiento --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE, "is available \n ----------------------")

TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_masks"

# epochs_per_trial = 12
epochs_per_trial = int(input("Enter epochs per trial :"))
n_trials = int(input("Enter number of trials: "))
# n_trials = 50

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
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_int('batch_size', 2, 8)
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3) if optimizer_name == "AdamW" else None

    model = UnetPlusPlus(in_channels=3, out_channels=4, impl_dropout=True, prob_dropout=dropout_prob).to(DEVICE)
    loss_fn = dice_loss_multiclass
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == 'Adam' else optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_transform = A.Compose([
        A.Resize(height=240, width=240),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
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

    for epoch in range(epochs_per_trial):  # Fixed number of epochs for optimization
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        dict_metrics_per_class = check_metrics(val_loader, model, device=DEVICE)
        epoch_mean_dice = np.mean(dict_metrics_per_class["dice_coefficient"])

        if epoch_mean_dice > best_mean_dice:
            best_mean_dice = epoch_mean_dice

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
    os.makedirs("output_assets_model", exist_ok=True)
    with open("output_assets_model/Optuna/optuna_best_hyp.json", "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()