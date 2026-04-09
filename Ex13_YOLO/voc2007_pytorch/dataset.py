"""
VOC 2007 dataset for multi-label classification.
Reads ImageSets and Annotations; returns image + 20-dim binary labels.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from config import (
    VOC_IMAGES, VOC_ANNOTATIONS, VOC_CLASSES, NUM_CLASSES,
    TRAINVAL_LIST, TRAIN_LIST, VAL_LIST, IMAGE_SIZE
)


def parse_voc_xml(ann_path, class_to_idx):
    """Parse VOC XML; return set of present class indices (ignore difficult)."""
    tree = ET.parse(ann_path)
    root = tree.getroot()
    labels = set()
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue
        name = obj.find("name").text
        if name in class_to_idx:
            labels.add(class_to_idx[name])
    return labels


def build_image_list(list_path):
    """Return list of image ids (no extension) from trainval.txt / train.txt / val.txt."""
    if not os.path.isfile(list_path):
        return []
    with open(list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class VOC2007Dataset(Dataset):
    """VOC 2007 multi-label classification."""

    def __init__(self, image_ids, transform=None, class_to_idx=None):
        self.image_ids = image_ids
        self.transform = transform
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(VOC_CLASSES)}
        self.num_classes = NUM_CLASSES

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(VOC_IMAGES, img_id + ".jpg")
        ann_path = os.path.join(VOC_ANNOTATIONS, img_id + ".xml")

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        labels = set()
        if os.path.isfile(ann_path):
            labels = parse_voc_xml(ann_path, self.class_to_idx)
        y = np.zeros(self.num_classes, dtype=np.float32)
        for i in labels:
            y[i] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(y)


def get_train_val_loaders(batch_size=32, num_workers=0, val_ratio=0.1, use_cuda=True):
    """Build train/val DataLoaders. Uses trainval.txt; splits by val_ratio. use_cuda=True enables pin_memory for GPU."""
    from torchvision import transforms

    all_ids = build_image_list(TRAINVAL_LIST)
    if not all_ids:
        raise FileNotFoundError(f"Image list not found: {TRAINVAL_LIST}")

    n = len(all_ids)
    np.random.seed(42)
    perm = np.random.permutation(n)
    nval = max(1, int(n * val_ratio))
    val_ids = [all_ids[i] for i in perm[:nval]]
    train_ids = [all_ids[i] for i in perm[nval:]]

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = VOC2007Dataset(train_ids, transform=transform_train)
    val_set = VOC2007Dataset(val_ids, transform=transform_val)

    pin = use_cuda
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader
