import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CRNN(nn.Module):
    def __init__(
        self,
        num_chars: int,
        hidden_size: int = 256,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_lstm_layers: int = 2
    ):
        super(CRNN, self).__init__()     
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone} not supported.")
        
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        if backbone in ["resnet18", "resnet34"]:
            cnn_out_channels = 512
        else:
            cnn_out_channels = 2048

        self.projector = nn.Conv2d(cnn_out_channels, hidden_size, kernel_size=1, stride=1)
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))  
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        # x shape: [batch, 3, H, W]
        features = self.cnn_backbone(x) # shape: [batch, cnn_out_channels, H', W']
        features = self.projector(features) # shape: [batch, hidden_size, H', W']
        features = self.height_pool(features) # shape: [batch, hidden_size, 1, W']
        features = features.squeeze(2)  # shape: [batch, hidden_size, W']
        features = features.permute(0, 2, 1).contiguous()  # shape: [batch, W', hidden_size]
        lstm_out, _ = self.lstm(features)  # shape: [batch, W', 2*hidden_size]
        logits = self.classifier(lstm_out)  # [batch, W', num_chars]
        logits = logits.permute(0, 2, 1)  # [batch, num_chars, 'W']
        return logits


