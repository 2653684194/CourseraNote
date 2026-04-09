"""
PyTorch replica of the backbone from ML_Note13_2 (3 ResBlocks) + VOC multi-label head.
Backbone: ResBlock1 -> ResBlock2 -> ResBlock3 (same structure as notebook).
Head: Global Average Pooling -> FC -> 20 classes, sigmoid.
"""
import torch
import torch.nn as nn
from config import NUM_CLASSES


def conv3x3(in_c, out_c, stride=1, same_padding=True):
    p = 1 if same_padding else 0
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=p)


def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride=stride)


class ResBlock(nn.Module):
    """ResBlock: main path (sequence of conv+act+pool) + optional shortcut (1x1 or identity)."""

    def __init__(self, layers, shortcut_proj=None):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.shortcut_proj = shortcut_proj  # None = identity

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut_proj is not None:
            out = out + self.shortcut_proj(x)
        return out


def make_backbone():
    """
    Backbone matching ML_Note13_2 backbone2:
    ResBlock1: Conv 7x7 s2(3->64) -> AvgPool2 -> LeakyReLU -> MaxPool2 -> Conv 3x3(64->192) -> LeakyReLU -> MaxPool2
    ResBlock2: 1x1(192->128) -> LReLU -> 3x3(128->256) -> LReLU -> 1x1(256->256) -> LReLU -> 3x3(256->512) -> LReLU -> MaxPool2
    ResBlock3: 1x1(512->256) -> ... -> 3x3(512->1024) -> LReLU -> MaxPool2
    """
    act = nn.LeakyReLU(0.1)

    # ResBlock 1: 224 -> 112 -> 56 -> 28 -> 14 (if input 224)
    block1_layers = [
        nn.Conv2d(3, 64, 7, stride=2, padding=3),   # same_padding for 7x7 s2
        nn.AvgPool2d(2, stride=2),
        act,
        nn.MaxPool2d(2, stride=2),
        conv3x3(64, 192, 1),
        act,
        nn.MaxPool2d(2, stride=2),
    ]
    block1 = ResBlock(block1_layers, shortcut_proj=None)  # first block no shortcut

    # ResBlock 2: 14 -> 7 (spatial) -> 28 -> 14 -> 7
    block2_layers = [
        conv1x1(192, 128),
        act,
        conv3x3(128, 256),
        act,
        conv1x1(256, 256),
        act,
        conv3x3(256, 512),
        act,
        nn.MaxPool2d(2, stride=2),
    ]
    # Shortcut: 192 ch -> 512 ch, same spatial
    block2_shortcut = nn.Sequential(
        conv1x1(192, 512),
        nn.MaxPool2d(2, stride=2),
    )
    block2 = ResBlock(block2_layers, shortcut_proj=block2_shortcut)

    # ResBlock 3: repeated 1x1->3x3 blocks then final 1x1->3x3 and MaxPool
    block3_layers = [
        conv1x1(512, 256),
        act,
        conv3x3(256, 512),
        act,
        conv1x1(512, 256),
        act,
        conv3x3(256, 512),
        act,
        conv1x1(512, 256),
        act,
        conv3x3(256, 512),
        act,
        conv1x1(512, 256),
        act,
        conv3x3(256, 512),
        act,
        conv1x1(512, 512),
        act,
        conv3x3(512, 1024),
        act,
        nn.MaxPool2d(2, stride=2),
    ]
    block3_shortcut = nn.Sequential(conv1x1(512, 1024), nn.MaxPool2d(2, stride=2))
    block3 = ResBlock(block3_layers, shortcut_proj=block3_shortcut)

    return nn.Sequential(block1, block2, block3)


class VOCBackboneHead(nn.Module):
    """Backbone (3 ResBlocks) + GAP + FC -> num_classes, sigmoid for multi-label."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = make_backbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return torch.sigmoid(x)


def build_model(num_classes=NUM_CLASSES):
    return VOCBackboneHead(num_classes=num_classes)
