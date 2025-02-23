import torch
import torch.nn as nn
import torch.nn.functional as F

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn_relu = batchnorm_relu(out_c)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn_relu(x)
        x = self.dropout(x)
        return x
    
class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsample para g

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Aplicar upsampling a g1 para que coincida con las dimensiones de x1
        g1 = self.upsample(g1)
        
        psi = F.relu(g1 + x1)  # Sumar g1 y x1
        psi = self.psi(psi)
        return x * psi


class AttentionUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout=0.5):
        super(AttentionUnet, self).__init__()
        self.enc1 = conv_block(in_channels, 64, dropout)
        self.enc2 = conv_block(64, 128, dropout)
        self.enc3 = conv_block(128, 256, dropout)
        self.enc4 = conv_block(256, 512, dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = conv_block(512, 1024, dropout)

        # Ajustar F_g para que coincida con el número de canales de center (1024)
        self.att4 = attention_block(F_g=1024, F_l=512, F_int=512)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512, dropout)

        self.att3 = attention_block(F_g=512, F_l=256, F_int=256)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256, dropout)

        self.att2 = attention_block(F_g=256, F_l=128, F_int=128)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128, dropout)

        self.att1 = attention_block(F_g=128, F_l=64, F_int=64)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64, dropout)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, inputs):
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        att4 = self.att4(g=center, x=enc4)
        up4 = self.up4(center)
        merge4 = torch.cat([up4, att4], dim=1)
        dec4 = self.dec4(merge4)

        att3 = self.att3(g=dec4, x=enc3)
        up3 = self.up3(dec4)
        merge3 = torch.cat([up3, att3], dim=1)
        dec3 = self.dec3(merge3)

        att2 = self.att2(g=dec3, x=enc2)
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, att2], dim=1)
        dec2 = self.dec2(merge2)

        att1 = self.att1(g=dec2, x=enc1)
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, att1], dim=1)
        dec1 = self.dec1(merge1)

        output = self.final_conv(dec1)
        return output

def test():
    x = torch.randn((3, 1, 160, 160))  # Dimensiones deben ser pares
    model = AttentionUnet(in_channels=1, out_channels=1, dropout=0.5)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape , f"Expected (3, 1, 160, 160), but got {preds.shape}" # Esto fallará si no se restaura la resolución espacial

test()
