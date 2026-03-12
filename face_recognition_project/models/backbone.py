import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class FaceNet(nn.Module):
    def __init__(self, embedding_size=512, pretrained=False):
        super().__init__()

        self.backbone = resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x
