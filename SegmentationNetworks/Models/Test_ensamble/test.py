import json
import torch
import pandas as pd
from main_models import UNET, ResUnet, UnetPlusPlus, AttentionUnet
from metrics import calculate_metrics, calculate_double_metrics

# ------------- Parámetros ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
# test_image_dir = "/content/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/test_images"
# test_mask_dir = "/content/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/test_masks"
# test_image_dir = "/content/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68"  # colab
# test_mask_dir = "/content/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/masks_68"
test_image_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68"
test_mask_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/masks_68"

# ----- Cargamos el modelo entrenado con las mejores métricas ----------

# Este es bueno, pero hay que ajustar el peso ponderado a ver si mejora más:
checkpoint1 = torch.load("output_assets_model/best_model_checkpoint_ResUnetArtColab.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model1 = ResUnet(in_channels=3, out_channels=1).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model1.load_state_dict(checkpoint1["state_dict"])
model1.eval()

checkpoint2 = torch.load("output_assets_model/best_model_checkpoint_UnetPlusPlusArtColab.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model2 = UnetPlusPlus(in_channels=3, out_channels=1).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model2.load_state_dict(checkpoint2["state_dict"])
model2.eval()

# ----- Calculamos las métricas --------------

print("Calculating test metrics...")
# dice_coefficient, IoU, accuracy, precision, recall, f1_score = calculate_metrics(test_image_dir, test_mask_dir, model1, device=DEVICE, image_height=240, image_width=240)
dice_coefficient, IoU, accuracy, precision, recall, f1_score, hausdorff_dist, hausdorff_95_dist = calculate_double_metrics(test_image_dir, test_mask_dir, model1, model2, device=DEVICE, image_height=240, image_width=240)

# ----- Guardamos las métricas en un archivo .csv --------------
test_metrics = {
    "Dice Coefficient": dice_coefficient,
    "IoU": IoU,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_score,
    "Hausdorff Distance": hausdorff_dist,
    "Hausdorff 95 Distance": hausdorff_95_dist
}
pd.DataFrame(test_metrics, index=[0]).to_csv("output_assets_model/test_metrics.csv", index=False)
# with open("output_assets_model/test_metrics.json", "w") as outfile: # Guardar las métricas en un archivo JSON
#     json.dump(test_metrics, outfile)

# ----- Imprimimos las métricas (opcional) --------------
print("Métricas calculadas para el test set:")
print(pd.DataFrame(test_metrics, index=[0]))

# # ------------------- Comparación del cálculo de métricas de validación (desp. del entrenamiento) -------------------
# # VAL_IMG_DIR = "/content/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_images"  # colab
# # VAL_MASK_DIR = "/content/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_masks"
print('========\n', 'Comparación de métricas de validación (después del entrenamiento, con el mejor estado del modelo)\n', '=====================')
dice_coefficient, IoU, accuracy, precision, recall, f1_score, hausdorff_dist, hausdorff_95_dist = calculate_double_metrics(VAL_IMG_DIR, VAL_MASK_DIR, model1, model2, device=DEVICE, batch_size=4)
print(f"Dice Coefficient: {dice_coefficient:.4f}")
print(f"IoU: {IoU:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"Hausdorff Distance: {hausdorff_dist:.4f}")
print(f"Hausdorff 95 Distance: {hausdorff_95_dist:.4f}")