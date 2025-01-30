import torch
import torch.nn as nn
import torch.nn.functional as F
import  torchvision.transforms.functional as TF


## ------------------ Unet -------------------

# Bloque de código de cada capa de la Unet (donde se aplican las dos convoluciones, Además se aplica el btchnorm):
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_prob > 0.0 and dropout_prob < 1.0:
            layers.append(nn.Dropout(dropout_prob))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
        dropout_prob=0.0
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)

        # Down part of Unet:
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_prob=dropout_prob))
            in_channels = feature

        # Up part of Unet:
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature, dropout_prob=dropout_prob))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_prob=dropout_prob) # Medium part
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) #Final part


    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

test()