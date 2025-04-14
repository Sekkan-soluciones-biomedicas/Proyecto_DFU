import torch
import numpy as np
from utils import get_test_loader

def check_metrics(loader, model, prin=True, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds =  (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += (torch.numel(preds))
            dice_score += (2* (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            true_positive += ((preds == 1) & (y == 1)).sum()
            false_positive += ((preds == 1) & (y == 0)).sum()
            false_negative += ((preds == 0) & (y == 1)).sum()

    # Calculate metrics:
    dice_coefficient = dice_score / len(loader) if len(loader) > 0 else 0
    IoU = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0
    accuracy = num_correct / num_pixels if num_pixels > 0 else 0 # Calculate accuracy
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if prin:  # Print metrics:
        print(f"Acc: {accuracy*100:.3f}")
        print(f"Dice score: {dice_score/len(loader)}")
    model.train()

    dice_coefficient = dice_coefficient.item() if isinstance(dice_coefficient, torch.Tensor) else dice_coefficient
    IoU = IoU.item() if isinstance(IoU, torch.Tensor) else IoU
    accuracy = accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy
    precision = precision.item() if isinstance(precision, torch.Tensor) else precision
    recall = recall.item() if isinstance(recall, torch.Tensor) else recall
    f1_score = f1_score.item() if isinstance(f1_score, torch.Tensor) else f1_score

    return dice_coefficient, IoU, accuracy, precision, recall, f1_score
    # return dice_coefficient.item(), IoU.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item()


def calculate_metrics(test_image_dir, test_mask_dir, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), image_height=240, image_width=240, num_workers=0, batch_size=4, pin_memory=True):
    model.eval()
    loader = get_test_loader(test_image_dir, test_mask_dir, batch_size= batch_size,  image_height=image_height, image_width=image_width, num_workers=num_workers, pin_memory=pin_memory) # Load images and masks

    dice_coefficient, IoU, accuracy, precision, recall, f1_score = check_metrics(loader, model, prin=False, device=device) # Calculate metrics
    model.train() # regresarlo a su estado original si se quiere seguir entrenando el modelo.

    return dice_coefficient, IoU, accuracy, precision, recall, f1_score


## Ensemble ----------------------------------

def check_double_metrics(loader, model1, model2, prin=True, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    model1.eval()
    model2.eval()
    w_m1 = 0.50  # Para ResUnetArtColab + UnetPlusPlusArtColab (Usados en el ensamble usado en el artículo Italia).
    w_m2 = 1-w_m1
    print((w_m1, w_m2))

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds1 = torch.sigmoid(model1(x))
            preds2 = torch.sigmoid(model2(x))
            preds = (w_m1*preds1 + w_m2*preds2)  # Promedio de las predicciones
            # preds = (preds1 + preds2) / 2
            preds = (preds > 0.5).float()  # Binarización de las predicciones
            num_correct += (preds == y).sum()
            num_pixels += (torch.numel(preds))
            dice_score += (2* (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            true_positive += ((preds == 1) & (y == 1)).sum()
            false_positive += ((preds == 1) & (y == 0)).sum()
            false_negative += ((preds == 0) & (y == 1)).sum()

    # Calculate metrics:
    # dice_coefficient = dice_score / len(loader) if len(loader) > 0 else 0
    dice_coefficient = ( 2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-8) if (2 * true_positive + false_positive + false_negative) > 0 else 0
    IoU = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0
    accuracy = num_correct / num_pixels if num_pixels > 0 else 0 # Calculate accuracy
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if prin:  # Print metrics:
        print(f"Acc: {accuracy*100:.3f}")
        print(f"Dice score: {dice_score/len(loader)}")
    model1.train()
    model2.train()

    dice_coefficient = dice_coefficient.item() if isinstance(dice_coefficient, torch.Tensor) else dice_coefficient
    IoU = IoU.item() if isinstance(IoU, torch.Tensor) else IoU
    accuracy = accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy
    precision = precision.item() if isinstance(precision, torch.Tensor) else precision
    recall = recall.item() if isinstance(recall, torch.Tensor) else recall
    f1_score = f1_score.item() if isinstance(f1_score, torch.Tensor) else f1_score

    return dice_coefficient, IoU, accuracy, precision, recall, f1_score
    # return dice_coefficient.item(), IoU.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item()

def calculate_double_metrics(test_image_dir, test_mask_dir, model1, model2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), image_height=240, image_width=240, num_workers=0, batch_size=4, pin_memory=True):
    model1.eval()
    model2.eval()
    loader = get_test_loader(test_image_dir, test_mask_dir, batch_size= batch_size,  image_height=image_height, image_width=image_width, num_workers=num_workers, pin_memory=pin_memory) # Load images and masks

    dice_coefficient, IoU, accuracy, precision, recall, f1_score = check_double_metrics(loader, model1, model2, prin=False, device=device) # Calculate metrics
    model1.train() # regresarlo a su estado original si se quiere seguir entrenando el modelo.
    model2.train()

    return dice_coefficient, IoU, accuracy, precision, recall, f1_score


## ------------------------------------------.


def dice_loss(input, target):
    smooth = 1.0
    input = torch.sigmoid(input)  # Aplicar sigmoide para obtener probabilidades
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))