import os
import json
import shutil
import pandas as pd
import zipfile
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import numpy as np
import torch.optim as optim
import segmentation_models_pytorch as smp  # Nueva importación para usar smp.DeepLabV3+
from metrics import check_metrics, DiceLoss, FocalLoss, WeightedSumOfLosses
from utils import save_predictions_as_imgs, load_checkpoint, get_loaders, plot_dice_loss, concat_dicts_to_dataframe

# --------------- history --------------------------

# Define las rutas
carpeta_origen = 'output_assets_model'
carpeta_destino = os.path.join(carpeta_origen, 'history_oam')

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

NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE, "is available \n ----------------------")
BATCH_SIZE = 4
NUM_WORKERS = 0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False    # True si deseas cargar un modelo preentrenado
SAVE_IMS = True
SAVE_MODEL = True  # IMPORTANTE: debe estar en True para guardar el modelo y sus datos
PATIENCE = 50  # Para early stopping. Usa un valor grande para evitarlo
OPTIMIZER_NAME = 'Adam'
WEIGHT_DECAY = 1e-6  # Para el optimizador AdamW

TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_dfutissue/train/images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_dfutissue/train/masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_dfutissue/val/images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_dfutissue/val/masks"

if not os.path.exists('output_assets_model'):  # Crear el directorio assets si no existe
    os.makedirs('output_assets_model')

# Cargar los mejores hiperparámetros de Optuna
with open('output_assets_model/Optuna/optuna_best_hyp_w_balanced_clas.json', 'r') as f:
    best_hyperparams = json.load(f)
LEARNING_RATE = best_hyperparams['params']['learning_rate']
BATCH_SIZE = best_hyperparams['params']['batch_size']
OPTIMIZER_NAME = best_hyperparams['params']['optimizer']
WEIGHT_DECAY = best_hyperparams['params'].get('weight_decay', 1e-6) if OPTIMIZER_NAME == "AdamW" else None
w_dice = best_hyperparams['params']['w_dice']
w_focal = 1.0 - w_dice

# ------------------- Funciones de entrenamiento -------------------

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0  # Inicializar la pérdida total
    num_batches = 0  # Inicializar el contador de batches

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float32)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Actualizar el loop de tqdm
        loop.set_postfix(loss=loss.item())

        # Acumular la pérdida y contar los batches
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss  # Devolver la pérdida promedio

def main(NUM_EPOCHS=NUM_EPOCHS):
    # Definir el codificador para smp.DeepLabV3+
    # encoder_name = 'resnet50'  # Puedes cambiarlo a otro como 'efficientnet-b0' o 'resnet50'
    encoder_name = 'mit_b3'  # Codificador fijo
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')

    # Transformaciones de entrenamiento con preprocesamiento del codificador
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.5),
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.8,
        ),
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
                A.CLAHE(p=0.25),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                A.RandomGamma(p=0.25),
                A.HueSaturationValue(p=0.25),
            ],
            p=0.3,
        ),
        A.Lambda(image=preprocessing_fn),  # Preprocesamiento específico del codificador
        ToTensorV2(),
    ])

    # Transformaciones de validación con preprocesamiento del codificador
    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Lambda(image=preprocessing_fn),  # Preprocesamiento específico del codificador
        ToTensorV2(),
    ])


    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,      # Ejemplo de backbone
        encoder_weights="imagenet",   # Pesos preentrenados
        in_channels=3,                # Canales de entrada (RGB)
        classes=4,                    # Número de clases (fondo, granulación, fibrinoso, calloso)
        activation=None,  # Devolvemos logits para la función de pérdida
        decoder_attention_type='pscse',
    ).to(DEVICE)

    # Definir la función de pérdida
    dice_loss = DiceLoss(eps=1e-6)
    focal_loss = FocalLoss(gamma=2.0, alpha=None, reduction="mean")
    loss_fn = WeightedSumOfLosses(dice_loss, focal_loss, w1=w_dice, w2=w_focal)

    # Definir el optimizador
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) if OPTIMIZER_NAME == 'Adam' else optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler para reducir la tasa de aprendizaje si el Dice no mejora
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=True, factor=0.5, min_lr=1e-7)

    # Cargar los datos
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("output_assets_model/best_model_checkpoint_DeepLabV3+_smp.pth", weights_only=True), model)
        print("Model loaded successfully!")

    scaler = torch.amp.GradScaler('cuda')
    L_dicts_metrics = []  # Lista de diccionarios de métricas de cada época
    L_loss = []  # Lista de pérdidas
    L_mean_dices = []
    best_dice = 0.0
    cnt_patience = 0

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch + 1}")

        # Entrenar el modelo
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        L_loss.append(epoch_loss)

        # Evaluar en el conjunto de validación
        dict_metrics_per_class = check_metrics(val_loader, model, device=DEVICE)
        epoch_mean_dice = np.mean(dict_metrics_per_class["dice_coefficient"])
        # scheduler.step(epoch_mean_dice)  # Actualizar scheduler basado en el Dice promedio
        scheduler.step(epoch_loss)  # Actualizar scheduler basado en la pérdida promedio
        L_dicts_metrics.append(dict_metrics_per_class)
        L_mean_dices.append(epoch_mean_dice)

        if SAVE_MODEL:
            if epoch == 0:  # Inicializar best_dice en la primera época
                best_dice = epoch_mean_dice
            # Guardar el mejor modelo basado en el Dice
            if epoch_mean_dice >= best_dice:
                print(f"Model saved with loss: {epoch_loss} and mean dice: {epoch_mean_dice}")
                best_dice = epoch_mean_dice
                best_model_epoch = epoch
                cnt_patience = 0  # Resetear el contador de paciencia

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, "output_assets_model/best_model_checkpoint_DeepLabV3+_smp.pth")
                with zipfile.ZipFile("output_assets_model/best_model_checkpoint_DeepLabV3+_smp.zip", 'w') as zipf:
                    zipf.write("output_assets_model/best_model_checkpoint_DeepLabV3+_smp.pth")
            else:
                cnt_patience += 1  # Aumentar el contador si no mejora

        # Early stopping
        if cnt_patience >= PATIENCE:
            print(f"===Early stopping at epoch: {epoch:04d}===")
            break

        # Guardar ejemplos de predicciones
        if SAVE_IMS:
            print("Saving image in training...")
            save_predictions_as_imgs(val_loader, model, folder="output_assets_model/saved_images/smp/", device=DEVICE)

    end_time = time.time()

    if SAVE_MODEL:
        print("Saving metrics...")
        # Guardar métricas por época
        df_metrics = concat_dicts_to_dataframe(L_dicts_metrics)
        df_metrics.to_csv('output_assets_model/metrics_per_epoch_DeepLabV3+_smp.csv', index=False)

        # Graficar Dice y Loss
        plot_dice_loss(L_mean_dices, L_loss, show_plot=False)

        # Guardar las mejores métricas de validación
        cols = ['Best Dice Score', 'Best IoU', 'Best Accuracy', 'Best Precision', 'Best Recall', 'Best F1 Score']
        best_metrics_c0 = [max(df_metrics[df_metrics.Class == 0]['dice_coefficient']), max(df_metrics[df_metrics.Class == 0]['IoU']), max(df_metrics[df_metrics.Class == 0]['accuracy']), max(df_metrics[df_metrics.Class == 0]['precision']), max(df_metrics[df_metrics.Class == 0]['recall']), max(df_metrics[df_metrics.Class == 0]['f1_score'])]
        best_metrics_c1 = [max(df_metrics[df_metrics.Class == 1]['dice_coefficient']), max(df_metrics[df_metrics.Class == 1]['IoU']), max(df_metrics[df_metrics.Class == 1]['accuracy']), max(df_metrics[df_metrics.Class == 1]['precision']), max(df_metrics[df_metrics.Class == 1]['recall']), max(df_metrics[df_metrics.Class == 1]['f1_score'])]
        best_metrics_c2 = [max(df_metrics[df_metrics.Class == 2]['dice_coefficient']), max(df_metrics[df_metrics.Class == 2]['IoU']), max(df_metrics[df_metrics.Class == 2]['accuracy']), max(df_metrics[df_metrics.Class == 2]['precision']), max(df_metrics[df_metrics.Class == 2]['recall']), max(df_metrics[df_metrics.Class == 2]['f1_score'])]
        best_metrics_c3 = [max(df_metrics[df_metrics.Class == 3]['dice_coefficient']), max(df_metrics[df_metrics.Class == 3]['IoU']), max(df_metrics[df_metrics.Class == 3]['accuracy']), max(df_metrics[df_metrics.Class == 3]['precision']), max(df_metrics[df_metrics.Class == 3]['recall']), max(df_metrics[df_metrics.Class == 3]['f1_score'])]
        best_metrics_df = pd.DataFrame([best_metrics_c0, best_metrics_c1, best_metrics_c2, best_metrics_c3], columns=cols, index=[0, 1, 2, 3])
        best_metrics_df.index.name = 'Class'
        best_metrics_df.to_csv('output_assets_model/best_metrics_val(during_training)_DeepLabV3+_smp.csv', index=True)

        # Guardar parámetros
        parameters = {
            'Num Epochs': NUM_EPOCHS,
            'Learning Rate': LEARNING_RATE,
            'Batch Size': BATCH_SIZE,
            'Image Height': IMAGE_HEIGHT,
            'Image Width': IMAGE_WIDTH,
            'Device': str(DEVICE),
            'Num Workers': NUM_WORKERS,
            'Pin Memory': PIN_MEMORY,
            'Load Model': LOAD_MODEL,
            'Save Images': SAVE_IMS,
            'Train Image Dir': TRAIN_IMG_DIR,
            'Val Image Dir': VAL_IMG_DIR,
            'Elapsed Time[m]': round((end_time - start_time) / 60, 4),
            'Best_model_epoch': best_model_epoch,
            'Patience (early_stop)': PATIENCE,
            'Encoder': encoder_name,
            'Optimizer': OPTIMIZER_NAME,
            'Weight Decay': WEIGHT_DECAY if OPTIMIZER_NAME == 'AdamW' else None,
            'w_dice': w_dice,
            'w_focal': w_focal,
        }
        pd.DataFrame([parameters]).to_csv('output_assets_model/parameters_DeepLabV3+_smp.csv', index=False)
        with open('output_assets_model/parameters_DeepLabV3+_smp.json', 'w') as json_file:
            json.dump(parameters, json_file, indent=4)

        print('Best model epoch:', best_model_epoch + 1, "with dice score:", best_dice)
        print("Elapsed time [m]:", round((end_time - start_time) / 60, 4))
        print("Elapsed time [h]:", round((end_time - start_time) / 3600, 4))
        print("Metrics saved successfully!")
    return model

# ------------------- Entrenamiento -------------------
num_ep = input("Enter # of epochs: ")
try:
    print("Training for ", num_ep, " epochs.")
    Modl = main(NUM_EPOCHS=int(num_ep))
except ValueError:
    print("Número inválido. Se entrenará con 10 épocas por default.")
    Modl = main()