import torch
import numpy as np

def check_metrics(loader, model, device="cuda"):
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
            dice_score += (2* (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            true_positive += ((preds == 1) & (y == 1)).sum()
            false_positive += ((preds == 1) & (y == 0)).sum()
            false_negative += ((preds == 0) & (y == 1)).sum()

    # Calculate metrics:
    accuracy = num_correct / num_pixels if num_pixels > 0 else 0 # Calculate accuracy
    dice_coefficient = dice_score / len(loader) if len(loader) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics:
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy*100:.3f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return dice_coefficient, accuracy, precision, recall, f1_score


def dice_loss(input, target):
    smooth = 1.0
    input = torch.sigmoid(input)  # Aplicar sigmoide para obtener probabilidades
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))