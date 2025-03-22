import torch
import time
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

### ------------------ ResUnet con Dropout -------------------

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=0.5):
        super().__init__()
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.dropout1 = nn.Dropout2d(p=dropout)  # Dropout agregado
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.dropout2 = nn.Dropout2d(p=dropout)  # Dropout agregado
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.dropout1(x)  # Aplicar Dropout después de la primera convolución
        x = self.b2(x)
        x = self.c2(x)
        x = self.dropout2(x)  # Aplicar Dropout después de la segunda convolución
        s = self.s(inputs)
        skip = x + s
        return skip

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.5):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dropout = nn.Dropout2d(p=dropout)  # Dropout antes de la convolución
        self.r = residual_block(in_c + out_c, out_c, dropout=dropout)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.dropout(x)  # Aplicar Dropout después de la concatenación
        x = self.r(x)
        return x

class ResUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, dropout=0.5, n_filters=64):
        super().__init__()
        # Usamos n_filters como base y escalamos las capas
        f = n_filters  # Número inicial de filtros

        # Encoder
        self.c11 = nn.Conv2d(in_channels, f, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(f)
        self.c12 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_channels, f, kernel_size=1, padding=0)

        self.r2 = residual_block(f, f * 2, stride=2, dropout=0.0)      # f -> 2f
        self.r3 = residual_block(f * 2, f * 4, stride=2, dropout=0.0)  # 2f -> 4f
        self.r4 = residual_block(f * 4, f * 8, stride=2, dropout=0.0)  # 4f -> 8f

        # Decoder
        self.d1 = decoder_block(f * 8, f * 4, dropout=dropout)  # 8f -> 4f
        self.d2 = decoder_block(f * 4, f * 2, dropout=dropout)  # 4f -> 2f
        self.d3 = decoder_block(f * 2, f, dropout=dropout)      # 2f -> f

        # Capa final
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, inputs):
        # Encoder
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)
        b = self.r4(skip3)

        # Decoder
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        # Salida
        output = self.final_conv(d3)
        return output
    

# Función para probar el modelo

def test_model(n_filters=64, dropout=0.5):
    batch_size = 2  # Número de imágenes en el lote
    in_channels = 3  # Canales de entrada (RGB)
    height, width = 224, 224  # Tamaño de las imágenes de entrada
    num_classes = 4  # Número de clases para segmentación

    # Crear un tensor aleatorio para simular las imágenes de entrada
    x = torch.randn(batch_size, in_channels, height, width)

    # Instanciar el modelo con los hiperparámetros
    model = ResUnet(
        in_channels=in_channels,
        out_channels=num_classes,
        dropout=dropout,
        n_filters=n_filters
    )

    # Pasar el tensor de entrada a través del modelo
    outputs = model(x)

    # Imprimir las dimensiones de la salida
    print(f"Dimensiones de salida: {outputs.shape}")
    print(f"Esperado: ({batch_size}, {num_classes}, {height}, {width})")

    # Verificar que las dimensiones sean las correctas
    assert outputs.shape == (batch_size, num_classes, height, width), \
        "Error: Las dimensiones de la salida no son correctas."

    print(f"El modelo funciona correctamente con n_filters={n_filters} y dropout={dropout}.")

# Llamar a la función de prueba
if __name__ == "__main__":
    print("Probando el modelo...")
    start = time.time()
    test_model(n_filters=32, dropout=0.3)  # Ejemplo con valores específicos
    print("(Tiempo de 2 imágenes 224x224: ", round(time.time() - start, 2), "s)\n\n")