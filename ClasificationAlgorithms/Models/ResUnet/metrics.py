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
            x = x.to(device)
            y = y.to(device)
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

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds1 = model1(x)
            preds2 = model2(x)
            preds_mean = (preds1 + preds2) / 2
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


## Metrics from DFUTissueSegNet code:




# import re
# import torch.nn as nn

# ### ----- DFUTissueSegNet/Codes/segmentation_models_pytorch/utils/base.py ----
# class BaseObject(nn.Module):
#     def __init__(self, name=None):
#         super().__init__()
#         self._name = name

#     @property
#     def __name__(self):
#         if self._name is None:
#             name = self.__class__.__name__
#             s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
#             return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
#         else:
#             return self._name



# class Loss(BaseObject):
#     def __add__(self, other):
#         if isinstance(other, Loss):
#             return SumOfLosses(self, other)
#         else:
#             raise ValueError("Loss should be inherited from `Loss` class")

#     def __radd__(self, other):
#         return self.__add__(other)

#     def __mul__(self, value):
#         if isinstance(value, (int, float)):
#             return MultipliedLoss(self, value)
#         else:
#             raise ValueError("Loss should be inherited from `BaseLoss` class")

#     def __rmul__(self, other):
#         return self.__mul__(other)
    
# class SumOfLosses(Loss):
#     def __init__(self, l1, l2):
#         name = "{} + {}".format(l1.__name__, l2.__name__)
#         super().__init__(name=name)
#         self.l1 = l1
#         self.l2 = l2

#     def __call__(self, *inputs):
#         return self.l1.forward(*inputs) + self.l2.forward(*inputs)


# # class HybridLoss(Loss):
# #     def __init__(self, l1, l2, l3, w_list): # weight list, w_list = [w1, w2, w3]
# #         name = "{} + {} + {}".format(l1.__name__, l2.__name__, l3.__name__)
# #         super().__init__(name=name)
# #         self.l1 = l1
# #         self.l2 = l2
# #         self.l3 = l3
# #         self.w1 = w_list[0]
# #         self.w2 = w_list[1]
# #         self.w3 = w_list[2]

#     def __call__(self, *inputs):
#         return self.w1 * self.l1.forward(*inputs) + self.w2 * self.l2.forward(*inputs) \
#             + self.w3 * self.l3.forward(*inputs)

# class MultipliedLoss(Loss):
#     def __init__(self, loss, multiplier):

#         # resolve name
#         if len(loss.__name__.split("+")) > 1:
#             name = "{} * ({})".format(multiplier, loss.__name__)
#         else:
#             name = "{} * {}".format(multiplier, loss.__name__)
#         super().__init__(name=name)
#         self.loss = loss
#         self.multiplier = multiplier

#     def __call__(self, *inputs):
#         return self.multiplier * self.loss.forward(*inputs)
    
# ### ----- DFUTissueSegNet/Codes/segmentation_models_pytorch/base/modules.py ----
# class ArgMax(nn.Module):
#     def __init__(self, dim=None):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         return torch.argmax(x, dim=self.dim)


# class Clamp(nn.Module):
#     def __init__(self, min=0, max=1):
#         super().__init__()
#         self.min, self.max = min, max

#     def forward(self, x):
#         return torch.clamp(x, self.min, self.max)
        

# class Activation(nn.Module):
#     def __init__(self, name, **params):

#         super().__init__()

#         if name is None or name == "identity":
#             self.activation = nn.Identity(**params)
#         elif name == "sigmoid":
#             self.activation = nn.Sigmoid()
#         elif name == "softmax2d":
#             self.activation = nn.Softmax(dim=1, **params)
#         elif name == "softmax":
#             self.activation = nn.Softmax(**params)
#         elif name == "logsoftmax":
#             self.activation = nn.LogSoftmax(**params)
#         elif name == "tanh":
#             self.activation = nn.Tanh()
#         elif name == "argmax":
#             self.activation = ArgMax(**params)
#         elif name == "argmax2d":
#             self.activation = ArgMax(dim=1, **params)
#         elif name == "clamp":
#             self.activation = Clamp(**params)
#         elif callable(name):
#             self.activation = name(**params)
#         else:
#             raise ValueError(
#                 f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
#                 f"argmax/argmax2d/clamp/None; got {name}"
#             )

#     def forward(self, x):
#         return self.activation(x)
    
# ## ------- DFUTissueSegNet/Codes/segmentation_models_pytorch/utils/losses.py -------

# class DiceLoss(Loss):   ## Dice Loss
#     def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.f_score(
#             y_pr,
#             y_gt,
#             beta=self.beta,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )
    
# ## Focal loss:

# from typing import Optional
# from functools import partial

# # import torch
# from torch.nn.modules.loss import _Loss
# # from _functional import focal_loss_with_logits
# # from constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

# ## ---- DFUTissueSegNet/Codes/segmentation_models_pytorch/losses/_functional.py -----

# def focal_loss_with_logits(
#     output: torch.Tensor,
#     target: torch.Tensor,
#     gamma: float = 2.0,
#     alpha: Optional[float] = 0.25,
#     reduction: str = "mean",
#     normalized: bool = False,
#     reduced_threshold: Optional[float] = None,
#     eps: float = 1e-6,
# ) -> torch.Tensor:
#     """Compute binary focal loss between target and output logits.
#     See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

#     Args:
#         output: Tensor of arbitrary shape (predictions of the model)
#         target: Tensor of the same shape as input
#         gamma: Focal loss power factor
#         alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
#             high values will give more weight to positive class.
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
#             'mean': the sum of the output will be divided by the number of
#             elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
#             and :attr:`reduce` are in the process of being deprecated, and in the meantime,
#             specifying either of those two args will override :attr:`reduction`.
#             'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
#         normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
#         reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

#     References:
#         https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
#     """
#     target = target.type(output.type())

#     logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none") 
#     pt = torch.exp(-logpt)

#     # compute the loss
#     if reduced_threshold is None:
#         focal_term = (1.0 - pt).pow(gamma)
#     else:
#         focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
#         focal_term[pt < reduced_threshold] = 1

#     loss = focal_term * logpt

#     if alpha is not None:
#         loss *= alpha * target + (1 - alpha) * (1 - target)

#     if normalized:
#         norm_factor = focal_term.sum().clamp_min(eps)
#         loss /= norm_factor

#     if reduction == "mean":
#         loss = loss.mean()
#     if reduction == "sum":
#         loss = loss.sum()
#     if reduction == "batchwise_mean":
#         loss = loss.sum(0)

#     return loss

# ## ---- DFUTissueSegNet/Codes/segmentation_models_pytorch/losses/_functional.py -----

# #: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
# #: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
# #: classes are not mutually exclusive and each class have its own *channel*,
# #: pixels in each channel which are not belong to class labeled as **0**.
# #: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
# BINARY_MODE: str = "binary"
# MULTICLASS_MODE: str = "multiclass"
# MULTILABEL_MODE: str = "multilabel"


# ## ------- DFUTissueSegNet/Codes/segmentation_models_pytorch/utils/losses.py -------

# __all__ = ["FocalLoss"]

# class FocalLoss(_Loss, Loss):
#     def __init__(
#         self,
#         mode: str = 'binary',
#         alpha: Optional[float] = None,
#         gamma: Optional[float] = 2.0,
#         ignore_index: Optional[int] = None,
#         reduction: Optional[str] = "mean",
#         normalized: bool = False,
#         reduced_threshold: Optional[float] = None,
#     ):
#         """Compute Focal loss
#         Args:
#             mode: Loss mode 'binary', 'multiclass' or 'multilabel'
#             alpha: Prior probability of having positive value in target.
#             gamma: Power factor for dampening weight (focal strength).
#             ignore_index: If not None, targets may contain values to be ignored.
#                 Target values equal to ignore_index will be ignored from loss computation.
#             normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
#             reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
#                 should use `reduction="sum"`.
#         Shape
#              - **y_pred** - torch.Tensor of shape (N, C, H, W)
#              - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
#         Reference
#             https://github.com/BloodAxe/pytorch-toolbelt
#         """
#         assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
#         super().__init__()

#         self.mode = mode
#         self.ignore_index = ignore_index
#         self.focal_loss_fn = partial(
#             focal_loss_with_logits,
#             alpha=alpha,
#             gamma=gamma,
#             reduced_threshold=reduced_threshold,
#             reduction=reduction,
#             normalized=normalized,
#         )

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        

#         if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
#             y_true = y_true.view(-1)
#             y_pred = y_pred.view(-1)

#             if self.ignore_index is not None:
#                 # Filter predictions with ignore label from loss computation
#                 not_ignored = y_true != self.ignore_index
#                 y_pred = y_pred[not_ignored]
#                 y_true = y_true[not_ignored]

#             loss = self.focal_loss_fn(y_pred, y_true)

#         elif self.mode == MULTICLASS_MODE:

            

#             num_classes = y_pred.size(1)
#             loss = 0

#             # Filter anchors with -1 label from loss computation
#             if self.ignore_index is not None:
#                 not_ignored = y_true != self.ignore_index

#             for cls in range(num_classes):
#                 cls_y_true = (y_true == cls).long() # mkd commented

#                 # cls_y_true = y_true[:, cls, ...] # mkd added
#                 cls_y_pred = y_pred[:, cls, ...]

#                 if self.ignore_index is not None:
#                     cls_y_true = cls_y_true[not_ignored]
#                     cls_y_pred = cls_y_pred[not_ignored]

#                 loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

#         return loss
    
# ##### DEFINIMOS SUM OF LOSSES: #####
# # Loss function
# dice_loss = DiceLoss()
# focal_loss = FocalLoss()
# sum_of_losses = SumOfLosses(dice_loss, focal_loss)


import torch
import torch.nn as nn
import torch.nn.functional as F
# from functools import partial

# Definimos las clases base necesarias (simplificadas para este ejemplo)
class Loss(nn.Module):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def forward(self, pred, target):
        return self.l1(pred, target) + self.l2(pred, target)

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
        # Asegurarse de que target sea de tipo long
        if target.dtype != torch.long:
            target = target.long()

        # Calcular la pérdida focal para multiclass
        num_classes = pred.size(1)
        loss = 0
        for cls in range(num_classes):
            cls_y_true = (target == cls).float().squeeze(1)  # Convertir a máscara binaria por clase
            # print(cls_y_true.shape)
            cls_y_pred = pred[:, cls, ...]       # Logits de la clase actual
            # print(cls_y_pred.shape)
            logpt = F.binary_cross_entropy_with_logits(cls_y_pred, cls_y_true, reduction="none")
            pt = torch.exp(-logpt)
            focal_term = (1.0 - pt).pow(self.gamma)
            loss_cls = focal_term * logpt
            if self.alpha is not None:
                loss_cls *= self.alpha * cls_y_true + (1 - self.alpha) * (1 - cls_y_true)
            loss += loss_cls.mean() if self.reduction == "mean" else loss_cls.sum()
        return loss / num_classes if self.reduction == "mean" else loss

# Definimos la suma de pérdidas
dice_loss = DiceLoss(eps=1e-6)
focal_loss = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
sum_of_losses = SumOfLosses(dice_loss, focal_loss)