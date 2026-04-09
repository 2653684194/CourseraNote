# VOC2007 PyTorch project config (do not modify original workspace files)
import os

# Paths relative to Ex13_YOLO (parent of this folder)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOC_ROOT = os.path.join(_ROOT, "archive_2007", "VOCtrainval_06-Nov-2007")
VOC_IMAGES = os.path.join(VOC_ROOT, "JPEGImages")
VOC_ANNOTATIONS = os.path.join(VOC_ROOT, "Annotations")
VOC_IMAGESETS = os.path.join(VOC_ROOT, "ImageSets", "Main")

# Use trainval for training, optional val split from file
TRAINVAL_LIST = os.path.join(VOC_IMAGESETS, "trainval.txt")
TRAIN_LIST = os.path.join(VOC_IMAGESETS, "train.txt")
VAL_LIST = os.path.join(VOC_IMAGESETS, "val.txt")

# Outputs (all under voc2007_pytorch)
OUT_DIR = os.path.join(_ROOT, "voc2007_pytorch")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR = os.path.join(OUT_DIR, "visualizations")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# VOC 20 classes (alphabetical, same as PASCAL)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(VOC_CLASSES)

# Training
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 5e-4
