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
from main import ResUnet
from metrics import check_metrics, dice_loss_multiclass, calculate_metrics
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
print("Device: ",DEVICE, "is available \n ----------------------")
BATCH_SIZE = 4
NUM_WORKERS = 0
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False    # True if you want to load a pre-trained model
SAVE_IMS = True
SAVE_MODEL = True  # ! IMPORTANTE: debe esta en True para guardar el modelo y sus datos.
PATIENCE = 24 # for early stopping. Set big to avoid it.
p_dropout = 0.15 # Set 0 to no implement dropout
OPTIMIZER_NAME= 'Adam'
WEIGHT_DECAY= 1e-6 # For AdamW optimizer

# TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_images"
# TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_masks"
TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_semisup_padded/Resized/unlabel_data_padded"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_semisup_padded/pseudo_masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_masks"

if not os.path.exists('output_assets_model'): # Crear el directorio assets si no existe.
    os.makedirs('output_assets_model')

## Get best Optuna hyperparameters to train:
with open('output_assets_model/Optuna/optuna_best_hyp.json', 'r') as f: # Load best hyperparameters from JSON file
    best_hyperparams = json.load(f)
LEARNING_RATE = best_hyperparams['params']['learning_rate']
BATCH_SIZE = best_hyperparams['params']['batch_size']
p_dropout = best_hyperparams['params']['dropout_prob']
OPTIMIZER_NAME = best_hyperparams['params']['optimizer']
WEIGHT_DECAY = best_hyperparams['params']['weight_decay'] if OPTIMIZER_NAME == "AdamW" else None

#------------------- Funciones de entrenamiento -------------------

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0  # Inicializar la pérdida total
    num_batches = 0  # Inicializar el contador de batches

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
        loop.set_postfix(loss=loss.item)

        # Acumular la pérdida y contar los batches
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss  # Devolver la pérdida promedio

def main(NUM_EPOCHS=NUM_EPOCHS):

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Perspective(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
            A.GaussNoise(p=0.2),
            
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.OneOf(
    #             [
    #                 A.HorizontalFlip(p=0.5),
    #                 A.VerticalFlip(p=0.5),
    #             ],
    #             p=0.8,
    #         ),
    #         A.OneOf(
    #             [
    #                 A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
    #                 A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
    #                 A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
    #                 A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0),
    #             ],
    #             p=0.9,
    #         ),
    #         A.Rotate(limit=35, p=0.5),
    #         A.OneOf(
    #             [
    #                 A.Perspective(p=0.2),
    #                 A.GaussNoise(p=0.2),
    #                 A.Sharpen(p=0.2),
    #                 A.Blur(blur_limit=3, p=0.2),
    #                 A.MotionBlur(blur_limit=3, p=0.2),
    #             ],
    #             p=0.5,
    #         ),
    #         A.OneOf(
    #             [
    #                 A.CLAHE(p=0.25),
    #                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
    #                 A.RandomGamma(p=0.25),
    #                 A.HueSaturationValue(p=0.25),
    #             ],
    #             p=0.3,
    #         ),
    #         A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    #         # A.Permute(2, 0, 1),  # Descomentar si sigue el problema
    #         ToTensorV2(),
    #     ],
    #     p=0.9,
    # )

    # # Agregamos ToTensorV2() fuera del bloque con p=0.9
    # train_transform = A.Compose([
    #     augmentations,  # 🔹 Se aplican augmentations con p=0.9
    #     ToTensorV2(),   # 🔹 Siempre convierte a tensor
    # ])

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )

    model = ResUnet(in_channels=3, out_channels=4, dropout=p_dropout).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = dice_loss
    loss_fn = dice_loss_multiclass
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) if OPTIMIZER_NAME == 'Adam' else optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # Reduce LR if validation loss plateaus

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
        load_checkpoint(torch.load("C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/Models/ResUnet/output_assets_model/best_model_checkpoint_ResUnet.pth",  weights_only=True), model)
        print("Model loaded successfully!")

    scaler = torch.amp.GradScaler('cuda')
    L_dicts_metrics = []  # Lista de diccionarios de métricas de cada época
    L_loss = []  # Lista de pérdidas
    L_mean_dices = []
    best_dice = 0.0
    best_loss = 0.0
    best_dice = 0.0
    cnt_patience = 0

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch + 1}")

        # Train the model:
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        L_loss.append(epoch_loss)
        # scheduler.step(epoch_loss) # Update scheduler based on training loss

        # Check accuracy on validation set:
        dict_metrics_per_class = check_metrics(val_loader, model, device=DEVICE)
        epoch_mean_dice = np.mean(dict_metrics_per_class["dice_coefficient"])  # Coeficiente dice promedio de todas las clases en la época actual
        scheduler.step(epoch_mean_dice) # Update scheduler based on mean_dice
        L_dicts_metrics.append(dict_metrics_per_class)
        # print(f"mean dice: {epoch_mean_dice}")
        L_mean_dices.append(epoch_mean_dice)

        if SAVE_MODEL:
            if epoch == 1:
                best_loss = epoch_loss
                best_dice = epoch_mean_dice
            # Save best model, based in dice:
            # if epoch_loss <= best_loss:
            if epoch_mean_dice >= best_dice:
                print(f"Model saved with loss: {epoch_loss} and mean dice: {epoch_mean_dice}")
                best_loss = epoch_loss
                best_dice = epoch_mean_dice
                best_dice = epoch_mean_dice
                best_model_epoch = epoch
                cnt_patience = 0  # Resetear el contador de paciencia si el modelo sí mejora.

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                # torch.save(checkpoint, f"model_checkpoint_epoch_{epoch+1}.pth")

                # Guardar el modelo en .pth y en .zip:
                torch.save(checkpoint, "output_assets_model/best_model_checkpoint_ResUnet.pth")
                # torch.save(checkpoint, "my_checkpoint.pth.tar")
                with zipfile.ZipFile("output_assets_model/best_model_checkpoint_ResUnet.zip", 'w') as zipf:
                    zipf.write("output_assets_model/best_model_checkpoint_ResUnet.pth")
            else:
                cnt_patience += 1 # Aumentar el contador si el modelo mejora.
        # Early stopping
        if cnt_patience >= PATIENCE:
            print(f"===Early stopping at epoch: {epoch:04d}===")
            break
            
        # Save some example predictions to a folder
        if SAVE_IMS:
            print("saving image in training...")
            save_predictions_as_imgs(val_loader, model, folder="output_assets_model/saved_images/", device=DEVICE)
    end_time = time.time()


    if SAVE_MODEL:
        print("Saving metrics...")
        # Save metrics for each epoch:
        df_metrics = concat_dicts_to_dataframe(L_dicts_metrics)
        df_metrics.to_csv('output_assets_model/metrics_per_epoch_ResUnet.csv', index=False)
        
        # Plot Dice (for class 0 only, it can be any class but just one at time) and Loss:
        # L_dice_0 = df_metrics[df_metrics.Class == 0]['dice_coefficient'].tolist()  # Este es para graficar el dice de una sola clase, en este caso está la clase 0 (el background)
        plot_dice_loss(L_mean_dices, L_loss, show_plot=False)

        # Save best val metrics during training in a csv file:
        cols = ['Best Dice Score', 'Best IoU', 'Best Accuracy', 'Best Precision', 'Best Recall', 'Best F1 Score']
        best_metrics_c0 = [max(df_metrics[df_metrics.Class == 0]['dice_coefficient']), max(df_metrics[df_metrics.Class == 0]['IoU']), max(df_metrics[df_metrics.Class == 0]['accuracy']), max(df_metrics[df_metrics.Class == 0]['precision']), max(df_metrics[df_metrics.Class == 0]['recall']), max(df_metrics[df_metrics.Class == 0]['f1_score'])]
        best_metrics_c1 = [max(df_metrics[df_metrics.Class == 1]['dice_coefficient']), max(df_metrics[df_metrics.Class == 1]['IoU']), max(df_metrics[df_metrics.Class == 1]['accuracy']), max(df_metrics[df_metrics.Class == 1]['precision']), max(df_metrics[df_metrics.Class == 1]['recall']), max(df_metrics[df_metrics.Class == 1]['f1_score'])]
        best_metrics_c2 = [max(df_metrics[df_metrics.Class == 2]['dice_coefficient']), max(df_metrics[df_metrics.Class == 2]['IoU']), max(df_metrics[df_metrics.Class == 2]['accuracy']), max(df_metrics[df_metrics.Class == 2]['precision']), max(df_metrics[df_metrics.Class == 2]['recall']), max(df_metrics[df_metrics.Class == 2]['f1_score'])]
        best_metrics_c3 = [max(df_metrics[df_metrics.Class == 3]['dice_coefficient']), max(df_metrics[df_metrics.Class == 3]['IoU']), max(df_metrics[df_metrics.Class == 3]['accuracy']), max(df_metrics[df_metrics.Class == 3]['precision']), max(df_metrics[df_metrics.Class == 3]['recall']), max(df_metrics[df_metrics.Class == 3]['f1_score'])]
        best_metrics_df = pd.DataFrame([best_metrics_c0, best_metrics_c1, best_metrics_c2, best_metrics_c3], columns=cols, index=[0, 1, 2, 3])
        best_metrics_df.index.name = 'Class'
        best_metrics_df.to_csv('output_assets_model/best_metrics_val(during_training)_ResUnet.csv', index=True)

        # Save parameters:
        parameters = {'Num Epochs': NUM_EPOCHS, 'Learning Rate': LEARNING_RATE, 'Batch Size': BATCH_SIZE, 'Image Height': IMAGE_HEIGHT, 'Image Width': IMAGE_WIDTH, 'Device': str(DEVICE), 'Num Workers': NUM_WORKERS, 'Pin Memory': PIN_MEMORY, 'Load Model': LOAD_MODEL, 'Save Images': SAVE_IMS, 'Train Image Dir': TRAIN_IMG_DIR, 'Val Image Dir': VAL_IMG_DIR, 'Elapsed Time[m]': round((end_time - start_time)/60, 4), 'Best_model_epoch': best_model_epoch, 'Patience (early_stop)': PATIENCE, 'Dropout_p' : p_dropout}  
        pd.DataFrame(parameters, index=[0]).to_csv('output_assets_model/parameters_ResUnet.csv', index=False)    # Guardar los parámetros en un archivo CSV
            # Guardar los parámetros como un archivo .json:
        with open('output_assets_model/parameters_ResUnet.json', 'w') as json_file:
            json.dump(parameters, json_file, indent=4)

        print('Best model epoch:', best_model_epoch)
        print("Metrics saved successfully!")
    return model


# ------------------- Entrenamiento -------------------
num_ep = input("Enter # of epochs: ")
# Modl = main(NUM_EPOCHS=int(num_ep))
try:
    print("Training for ", num_ep, " epochs.")
    Modl = main(NUM_EPOCHS=int(num_ep))
except NameError:
    print("Número inválido. Se entrenará con 10 épocas por default.")
    Modl = main()