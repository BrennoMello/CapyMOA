from capymoa.stream._stream import Schema
from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as models


class _TorchResNet(nn.Module):
    def __init__(self, model: Callable[..., models.ResNet], schema: Schema, img_size: tuple[int, int], pretrained=False, freeze_backbone=False):
        super().__init__()

        if pretrained:
            self.backbone = model(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        else:
            self.backbone = model(weights=None)

        if min(img_size) < 32:
            m = min(img_size)
            img_size = map(lambda x: 32*x//m, img_size)

        self.img_shape = (3, *img_size)

        num_classes = schema.get_num_classes()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, *self.img_shape))
        return self.backbone(x)


def resnet18(schema: Schema, img_size: tuple[int, int], pretrained: bool = False, freeze_backbone: bool = False):
    return _TorchResNet(models.resnet18, schema, img_size, pretrained, freeze_backbone)