import json
import torch
import pandas as pd
import json
import torch
import pandas as pd
from main import ResUnet #, ResUnet, etc.
from metrics import calculate_double_metrics

# ------------- Parámetros ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
img_size_for_test = 240
test_image_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/test_images"
test_mask_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/test_masks"

# ----- Cargamos el modelo entrenado con las mejores métricas ----------
checkpoint1 = torch.load("output_assets_model/best_model_checkpoint_ResUnet.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model1 = ResUnet(in_channels=3, out_channels=4).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model1.load_state_dict(checkpoint1["state_dict"])
model1.eval()

checkpoint2 = torch.load("output_assets_model/best_model_checkpoint_ResUnet.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model2 = ResUnet(in_channels=3, out_channels=4).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model2.load_state_dict(checkpoint2["state_dict"])
model2.eval()

# ----- Calculamos las métricas --------------

print("Calculating test metrics...")
dict_test_metrics = calculate_double_metrics(test_image_dir, test_mask_dir, model1, model2, num_classes=4, device=DEVICE, image_height=img_size_for_test, image_width=img_size_for_test)

# ----- Guardamos las métricas en un archivo .csv --------------
df_test_metrics = pd.DataFrame(dict_test_metrics, index=[0,1,2,3])
df_test_metrics.index.name = 'Class'
df_test_metrics.to_csv("output_assets_model/test_double_metrics_ResUnet.csv", index=True) # Sin índices.
# # Guardar las métricas en un archivo JSON
# with open("output_assets_model/test_metrics.json", "w") as outfile:
#     json.dump(test_metrics, outfile)
df_test_mean_metrics = pd.DataFrame(dict_test_metrics).mean()
df_test_mean_metrics.to_csv("output_assets_model/test_double_mean_metrics_ResUnet.csv", index=True) # Sin índices.

# ----- Imprimimos las métricas (opcional) --------------
print("Métricas calculadas para el test set:")
print(pd.DataFrame(dict_test_metrics))
print("---- Métricas promedio ----")
print(df_test_mean_metrics)


# ------------------- Comparación del cálculo de métricas de validación (desp. del entrenamiento) -------------------
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_masks"
print('========\n', 'Métricas de validación (después del entrenamiento, con el mejor estado del modelo)\n', '=====================')
dict_val_metrics = calculate_double_metrics(VAL_IMG_DIR, VAL_MASK_DIR, model1, model2, num_classes=4, device=DEVICE, image_height=img_size_for_test, image_width=img_size_for_test)
print(pd.DataFrame(dict_val_metrics))
print("---- Métricas promedio ----")
print(pd.DataFrame(dict_val_metrics).mean())