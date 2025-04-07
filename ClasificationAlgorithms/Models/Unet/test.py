import json
import torch
import pandas as pd
from main import UNET #, ResUnet, etc.
import json
import torch
import pandas as pd
from main import UNET #, ResUnet, etc.
from metrics import calculate_double_metrics
import segmentation_models_pytorch as smp

# ------------- Parámetros ----------------
## Get best Optuna hyperparameters to train:
with open('output_assets_model/Optuna/optuna_best_hyp_w_balanced_clas.json', 'r') as f: # Load best hyperparameters from JSON file
    best_hyperparams = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = best_hyperparams['params']['batch_size']
w_dice = best_hyperparams['params']['w_dice']
w_focal = 1.0 - w_dice
img_size_for_test = 256
test_image_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/test/images"
test_mask_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/test/masks"

# ----- Cargamos el modelo entrenado con las mejores métricas ----------
checkpoint1 = torch.load("output_assets_model/best_model_checkpoint_Unet_smp.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
# Definir el modelo smp.Unet
encoder_name = 'resnet34'  # Puedes cambiarlo a otro como 'efficientnet-b0' o 'resnet50'
model1 = smp.Unet(
        encoder_name=encoder_name,        # Codificador preentrenado
        encoder_weights='imagenet',       # Pesos preentrenados en ImageNet
        in_channels=3,                    # Canales de entrada (RGB)
        classes=4,                        # Número de clases en las máscaras
        activation=None,                  # Sin activación para devolver logits
    ).to(DEVICE)
model1.load_state_dict(checkpoint1["state_dict"])
model1.eval()

checkpoint2 = torch.load("output_assets_model/best_model_checkpoint_Unet_smp.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
encoder_name = 'resnet34'  # Puedes cambiarlo a otro como 'efficientnet-b0' o 'resnet50'
model2 = smp.Unet(
        encoder_name=encoder_name,        # Codificador preentrenado
        encoder_weights='imagenet',       # Pesos preentrenados en ImageNet
        in_channels=3,                    # Canales de entrada (RGB)
        classes=4,                        # Número de clases en las máscaras
        activation=None,                  # Sin activación para devolver logits
    ).to(DEVICE)
model2.load_state_dict(checkpoint2["state_dict"])
model2.eval()

# ----- Calculamos las métricas --------------

print("Calculating test metrics...")
dict_test_metrics = calculate_double_metrics(test_image_dir, test_mask_dir, model1, model2, num_classes=4, device=DEVICE, image_height=img_size_for_test, image_width=img_size_for_test, num_workers=0, batch_size=batch_size, pin_memory=True, encoder_name=encoder_name)

# ----- Guardamos las métricas en un archivo .csv --------------
df_test_metrics = pd.DataFrame(dict_test_metrics, index=[0,1,2,3])
df_test_metrics.index.name = 'Class'
df_test_metrics.to_csv("output_assets_model/test_metrics_Unet_smp.csv", index=True) # Sin índices.
# # Guardar las métricas en un archivo JSON
# with open("output_assets_model/test_metrics.json", "w") as outfile:
#     json.dump(test_metrics, outfile)
df_test_mean_metrics = pd.DataFrame(dict_test_metrics).mean()
df_test_mean_metrics.to_csv("output_assets_model/test_mean_metrics_Unet_smp.csv", index=True) # Sin índices.

# ----- Imprimimos las métricas (opcional) --------------
print("Métricas calculadas para el test set:")
print(pd.DataFrame(dict_test_metrics))
print("---- Métricas promedio ----")
print(df_test_mean_metrics)


# ------------------- Comparación del cálculo de métricas de validación (desp. del entrenamiento) -------------------
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/val/images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded_stratified/val/masks"
print('========\n', 'Métricas de validación (después del entrenamiento, con el mejor estado del modelo)\n', '=====================')
dict_val_metrics = calculate_double_metrics(VAL_IMG_DIR, VAL_MASK_DIR, model1, model2, num_classes=4, device=DEVICE, image_height=img_size_for_test, image_width=img_size_for_test, num_workers=0, batch_size=batch_size, pin_memory=True)
print(pd.DataFrame(dict_val_metrics))
print("---- Métricas promedio ----")
print(pd.DataFrame(dict_val_metrics).mean())