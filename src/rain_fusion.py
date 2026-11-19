"""Point rainfall prediction: CSI temporal encoder + U-Net contracting path."""
import torch
from torch import nn


class CloudEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
        self.network = nn.Sequential(block(channels, 16), nn.MaxPool2d(2),
                                     block(16, 32), nn.MaxPool2d(2), block(32, 64),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 128))

    def forward(self, x):
        return self.network(x)


class RainFusion(nn.Module):
    def __init__(self, csi_features, cloud_channels, mode='fusion'):
        super().__init__()
        if mode not in ('csi', 'cloud', 'fusion'):
            raise ValueError('mode must be csi, cloud or fusion')
        self.mode = mode
        self.csi_encoder = nn.Sequential(nn.Conv1d(csi_features, 32, 3, padding=1),
            nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 128)) if mode != 'cloud' else None
        self.cloud_encoder = CloudEncoder(cloud_channels) if mode != 'csi' else None
        self.head = nn.Sequential(nn.Linear(256 if mode == 'fusion' else 128, 64),
                                  nn.ReLU(), nn.Linear(64, 2))

    def forward(self, csi, cloud):
        features = []
        if self.csi_encoder is not None:
            features.append(self.csi_encoder(csi.transpose(1, 2)))
        if self.cloud_encoder is not None:
            features.append(self.cloud_encoder(cloud))
        raw = self.head(torch.cat(features, dim=1))
        # First output is an occurrence logit; second is nonnegative log1p(rate).
        return raw[:, 0], torch.nn.functional.softplus(raw[:, 1])
