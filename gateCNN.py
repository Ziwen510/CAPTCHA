import torch
import torch.nn as nn
import torch.nn.functional as F

class Point_DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, activation=nn.ELU()):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.pointwise(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class GateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.P1 = Point_DepthwiseConv(in_channels, out_channels // 2, kernel_size=3, activation=nn.ELU())
        self.P2 = Point_DepthwiseConv(out_channels // 2, out_channels // 2 * 3, kernel_size=3, activation=nn.Identity())
        self.H = nn.Tanh()
        self.T = nn.Sigmoid()
        self.P3 = Point_DepthwiseConv(out_channels // 2, out_channels, kernel_size=3, activation=nn.ELU())
        self.residule = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        y = self.P1(x)
        y_3 = self.P2(y)
        a, b, c = torch.chunk(y_3, 3, dim=1)
        h1 = self.H(a)
        h2 = self.H(b)
        t  = self.T(c)
        out = (h1 - h2) * t
        out = self.P3(out)
        return out + self.residule(x)

# Model
class GateCNN(nn.Module):
    def __init__(self, num_classes, num_gateblocks=8, input_channels=3, mid1_channels=32, mid2_channels=128):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.preprocess_depthwise = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.dropout1 = nn.Dropout2d(0.1)
        self.concat_channels = 16 + input_channels

        gateblocks = []
        gateblocks.append(GateBlock(self.concat_channels, mid1_channels))
        gateblocks.append(GateBlock(mid1_channels, mid1_channels))
        gateblocks.append(GateBlock(mid1_channels, mid2_channels))
        for _ in range(3, num_gateblocks):
            gateblocks.append(GateBlock(mid2_channels, mid2_channels))
        self.gateblocks = nn.Sequential(*gateblocks)

        self.dropout2 = nn.Dropout2d(0.1)
        self.classifier = nn.Sequential(
            nn.Conv2d(mid2_channels, num_classes, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_classes)
        )
    
    def forward(self, x):
        x_proj = self.initial_conv(x)  # (B, 16, H, W)

        # 13x13 depthwise conv + BN.
        x_pre = self.preprocess_depthwise(x_proj)  # (B, 16, H, W)
        x_pre = x_pre.contiguous()  # ensure contiguous memory before dropout
        x_drop = self.dropout1(x_pre)  # (B, 16, H, W)
        x_cat = torch.cat([x_drop, x], dim=1)  # (B, 16+input_channels, H, W)

        x_gate = self.gateblocks(x_cat)

        x_final = self.dropout2(x_gate)
        logits = self.classifier(x_final)  # (B, num_classes, H, W)
        logits = logits.mean(dim=2, keepdim=True)  # (B, num_classes, 1, W)
        logits = logits.squeeze(2)  # (B, num_classes, W)
        return logits


