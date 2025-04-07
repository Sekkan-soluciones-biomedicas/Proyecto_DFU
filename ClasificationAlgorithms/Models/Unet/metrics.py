import torch
import torch.nn.functional as F
import numpy as np
from utils import get_test_loader

def check_metrics(loader, model, num_classes = 4, prin=True, device="cuda"):
    """Calcula las métricas de evaluación de un modelo de segmentación multiclase (opcional) en un conjunto de datos de validación."""
    metrics = {
        "dice_coefficient": torch.zeros(num_classes, device=device),
        "IoU": torch.zeros(num_classes, device=device),
        "accuracy": torch.zeros(num_classes, device=device),
        "precision": torch.zeros(num_classes, device=device),
        "recall": torch.zeros(num_classes, device=device),
        "f1_score": torch.zeros(num_classes, device=device),
    }
    class_counts = torch.zeros(num_classes, device=device)
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # Asegúrate de que x sea float32
            y = y.to(device=device, dtype=torch.float32) 
            preds = torch.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1)

            for cls in range(num_classes):
                true_positive = ((preds == cls) & (y == cls)).sum().float()
                false_positive = ((preds == cls) & (y != cls)).sum().float()
                false_negative = ((preds != cls) & (y == cls)).sum().float()
                true_negative = ((preds != cls) & (y != cls)).sum().float()

                metrics["dice_coefficient"][cls] += (2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-8)
                metrics["IoU"][cls] += true_positive / (true_positive + false_positive + false_negative + 1e-8)
                metrics["accuracy"][cls] += (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + 1e-8)
                metrics["precision"][cls] += true_positive / (true_positive + false_positive + 1e-8)
                metrics["recall"][cls] += true_positive / (true_positive + false_negative + 1e-8)
                metrics["f1_score"][cls] += 2 * (metrics["precision"][cls] * metrics["recall"][cls]) / (metrics["precision"][cls] + metrics["recall"][cls] + 1e-8)
                class_counts[cls] += 1

    for key in metrics:
        metrics[key] /= class_counts

    if prin: # If print metrics:
        print(f"Classes:     {[cls for cls in range(num_classes)]}")
        print(f"Acc:         {[f'{acc:.4f}' for acc in metrics['accuracy']]}")
        print(f"Dice Coeff:  {[f'{dice:.4f}' for dice in metrics['dice_coefficient']]}")
        print(f"mean dice: {np.mean(metrics['dice_coefficient'].tolist())}")    ## To print the mean Dice.

    dict_metrics= {key: metrics[key].tolist() for key in metrics}
    model.train()
    return dict_metrics


def check_double_metrics(loader, model1, model2, num_classes=4, prin=True, device="cuda"):
    """Calcula las métricas de evaluación de un modelo de segmentación multiclase (opcional) en un conjunto de datos de validación."""
    metrics = {
        "dice_coefficient": torch.zeros(num_classes, device=device),
        "IoU": torch.zeros(num_classes, device=device),
        "accuracy": torch.zeros(num_classes, device=device),
        "precision": torch.zeros(num_classes, device=device),
        "recall": torch.zeros(num_classes, device=device),
        "f1_score": torch.zeros(num_classes, device=device),
    }
    class_counts = torch.zeros(num_classes, device=device)
    
    model1.eval()
    model2.eval()
    w_1 = 0.5
    w_2 = 1-w_1

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds1 = model1(x)
            preds2 = model2(x)
            preds_mean = w_1*preds1 + w_2*preds2
            preds = torch.softmax(preds_mean, dim=1)
            preds = torch.argmax(preds, dim=1)

            for cls in range(num_classes):
                true_positive = ((preds == cls) & (y == cls)).sum().float()
                false_positive = ((preds == cls) & (y != cls)).sum().float()
                false_negative = ((preds != cls) & (y == cls)).sum().float()
                true_negative = ((preds != cls) & (y != cls)).sum().float()

                metrics["dice_coefficient"][cls] += (2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-8)
                metrics["IoU"][cls] += true_positive / (true_positive + false_positive + false_negative + 1e-8)
                metrics["accuracy"][cls] += (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + 1e-8)
                precision = true_positive / (true_positive + false_positive + 1e-8)
                recall = true_positive / (true_positive + false_negative + 1e-8)
                metrics["precision"][cls] += precision
                metrics["recall"][cls] += recall
                metrics["f1_score"][cls] += 2 * (precision * recall) / (precision + recall + 1e-8)
                class_counts[cls] += 1

    for key in metrics:
        metrics[key] /= class_counts

    if prin:  # If print metrics:
        print(f"Classes:     {[cls for cls in range(num_classes)]}")
        print(f"Acc:         {[f'{acc:.4f}' for acc in metrics['accuracy']]}")
        print(f"Dice Coeff:  {[f'{dice:.4f}' for dice in metrics['dice_coefficient']]}")
        print(f"mean dice: {np.mean(metrics['dice_coefficient'].tolist())}")    ## To print the mean Dice.

    dict_metrics = {key: metrics[key].tolist() for key in metrics}
    model1.train()
    model2.train()
    return dict_metrics

def calculate_double_metrics(test_image_dir, test_mask_dir, model1, model2, num_classes=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), image_height=240, image_width=240, num_workers=0, batch_size=4, pin_memory=True):
    """Hace lo mismo que check metrics, pero en este se usa el test_loader para el cálculo de las métricas del test."""
    model1.eval()
    model2.eval()

    loader = get_test_loader(test_image_dir, test_mask_dir, batch_size= batch_size,  image_height=image_height, image_width=image_width, num_workers=num_workers, pin_memory=pin_memory)   # Cargar los datos.

    dict_metrics = check_double_metrics(loader, model1, model2, num_classes=num_classes, prin=False, device=device)  # Calcular las métricas.
    model1.train() # regresarlo a su estado original si se quiere seguir entrenando el modelo.
    model2.train()

    return dict_metrics

def calculate_metrics(test_image_dir, test_mask_dir, model, num_classes=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), image_height=240, image_width=240, num_workers=0, batch_size=4, pin_memory=True):
    """Hace lo mismo que check metrics, pero en este se usa el test_loader para el cálculo de las métricas del test."""
    model.eval()
    loader = get_test_loader(test_image_dir, test_mask_dir, batch_size= batch_size,  image_height=image_height, image_width=image_width, num_workers=num_workers, pin_memory=pin_memory)   # Cargar los datos.

    dict_metrics = check_metrics(loader, model, num_classes=num_classes, prin=False, device=device)  # Calcular las métricas.
    model.train() # regresarlo a su estado original si se quiere seguir entrenando el modelo.

    return dict_metrics
    # return dice_coefficient.item(), IoU.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item()

def dice_loss(input, target):
    smooth = 1.0
    input = torch.sigmoid(input)  # Aplicar sigmoide para obtener probabilidades
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def dice_loss_multiclass(pred, target, epsilon=1e-6):
    """
    Calcula la Dice Loss para segmentación multiclase.

    Args:
        pred (torch.Tensor): Salidas del modelo (logits) de tamaño (batch_size, num_classes, H, W).
        target (torch.Tensor): Etiquetas verdaderas de tamaño (batch_size, H, W).
        epsilon (float): Pequeño valor para evitar división por cero.

    Returns:
        torch.Tensor: Valor escalar de la pérdida Dice.
    """
    # Asegurarse de que las etiquetas estén en el tipo correcto
    if target.dtype != torch.long:
        target = target.long()

    # Aplicar Softmax a las predicciones para convertir logits a probabilidades
    pred = F.softmax(pred, dim=1)

    # Convertir las etiquetas a one-hot encoding y ajustar dimensiones
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1])  # (batch_size, H, W, num_classes)
    target_one_hot = target_one_hot.squeeze(1).permute(0, 3, 1, 2).float()  # (batch_size, num_classes, H, W)

    # Calcular el Dice para cada clase
    intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice.mean()

    return dice_loss


### ----- Actualización de ResUnet --------


import torch
import torch.nn as nn
import torch.nn.functional as F
# from functools import partial

# Definimos las clases base necesarias (simplificadas para este ejemplo)
class Loss(nn.Module):
    def __add__(self, other):
        if isinstance(other, Loss):
            return WeightedSumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

# class SumOfLosses(Loss):
#     def __init__(self, l1, l2):
#         super().__init__()
#         self.l1 = l1
#         self.l2 = l2

#     def forward(self, pred, target):
#         return self.l1(pred, target) + self.l2(pred, target)
class WeightedSumOfLosses(Loss):
    def __init__(self, l1, l2, w1=1.0, w2=1.0):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.w1 = w1
        self.w2 = w2

    def forward(self, pred, target):
        return self.w1 * self.l1(pred, target) + self.w2 * self.l2(pred, target)

# Definimos DiceLoss ajustado
class DiceLoss(Loss):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Asegurarse de que target sea de tipo long
        if target.dtype != torch.long:
            target = target.long()

        # Aplicar softmax a las predicciones (como en tu función original)
        pred = F.softmax(pred, dim=1)
        # print(pred.shape)

        # Convertir target a one-hot y ajustar dimensiones
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1)  # (batch_size, H, W, num_classes)
        # print(target_one_hot.shape)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (batch_size, num_classes, H, W)
        # print(target_one_hot.shape)

        # target_one_hot = F.one_hot(target, num_classes=pred.shape[1])  # (batch_size, H, W, num_classes)
        # target_one_hot = target_one_hot.squeeze(1).permute(0, 3, 1, 2).float()  # (batch_size, num_classes, H, W)

        # Calcular Dice Loss
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()

# Definimos FocalLoss ajustado para multiclass
class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: (batch_size, num_classes, H, W)
        # target: (batch_size, 1, H, W) o (batch_size, H, W)
        
        # Asegurarse de que target sea de tipo long y eliminar dimensión singleton si existe
        if target.dtype != torch.long:
            target = target.long()
        if target.dim() == 4 and target.size(1) == 1:  # Si target tiene forma [batch_size, 1, H, W]
            target = target.squeeze(1)  # Convertir a [batch_size, H, W]

        # Calcular la pérdida de entropía cruzada sin reducción
        ce_loss = F.cross_entropy(pred, target, reduction="none")  # [batch_size, H, W]
        
        # Calcular el término focal
        pt = torch.exp(-ce_loss)  # Probabilidades exponenciales
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Aplicar el factor focal

        # Aplicar alpha si está definido
        if self.alpha is not None:
            # Asegurarse de que alpha sea un tensor con pesos por clase
            alpha_t = self.alpha[target]  # alpha[target] selecciona el peso correspondiente a cada clase en target
            focal_loss = alpha_t * focal_loss

        # Aplicar la reducción
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# Definimos FocalLoss ajustado para multiclass (-VERSION ANTERIOR, LA DEL ARTÍCULO DFUTissueSegNet)
# class FocalLoss(Loss):
#     def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, pred, target):
#         # Asegurarse de que target sea de tipo long
#         if target.dtype != torch.long:
#             target = target.long()

#         # Calcular la pérdida focal para multiclass
#         num_classes = pred.size(1)
#         loss = 0
#         for cls in range(num_classes):
#             cls_y_true = (target == cls).float().squeeze(1)  # Convertir a máscara binaria por clase
#             # print(cls_y_true.shape)
#             cls_y_pred = pred[:, cls, ...]       # Logits de la clase actual
#             # print(cls_y_pred.shape)
#             logpt = F.binary_cross_entropy_with_logits(cls_y_pred, cls_y_true, reduction="none")
#             pt = torch.exp(-logpt)
#             focal_term = (1.0 - pt).pow(self.gamma)
#             loss_cls = focal_term * logpt
#             if self.alpha is not None:
#                 loss_cls *= self.alpha * cls_y_true + (1 - self.alpha) * (1 - cls_y_true)
#             loss += loss_cls.mean() if self.reduction == "mean" else loss_cls.sum()
#         return loss / num_classes if self.reduction == "mean" else loss

# Definimos la suma de pérdidas
dice_loss = DiceLoss(eps=1e-6)
focal_loss = FocalLoss(gamma=2.0, alpha=None, reduction="mean")
sum_of_losses = WeightedSumOfLosses(dice_loss, focal_loss, w1=0.5, w2=0.5)


