# YOLO v1 实现 - 使用说明

基于 CNN_v4_cupy.py 框架实现的 YOLO v1 目标检测模型。

## 论文参考
**You Only Look Once: Unified, Real-Time Object Detection**

## 文件结构

```
Ex13_YOLO/
├── CNN_v4_cupy.py          # 基础CNN框架（已存在）
├── YOLO_v1.py              # YOLO v1 核心实现
├── VOC_dataset.py          # PASCAL VOC 数据集加载器
├── train_yolo.py           # 训练脚本
├── test_yolo.py            # 测试和可视化脚本
├── YOLO_example.py         # 使用示例
└── models/                 # 模型保存目录
```

## 核心组件

### 1. YOLO_v1.py

**主要类:**
- `YOLOLoss` - YOLO损失函数（坐标损失、置信度损失、分类损失）
- `YOLOOutput` - 输出层，将FC输出reshape为YOLO格式
- `YOLOTargetEncoder` - 目标编码器（边界框 → YOLO格式）
- `YOLOPostProcessor` - 后处理器（解码 + NMS）
- `YOLOTrainer` - 训练器

**主要函数:**
- `create_yolo_v1_network()` - 标准YOLO v1（24层卷积）
- `create_tiny_yolo_v1()` - Tiny YOLO（9层卷积，更快）

### 2. VOC_dataset.py

**主要类:**
- `VOCDataset` - VOC数据集加载器
  - 支持真实VOC数据和模拟数据
  - `create_simulated_dataset()` - 创建模拟数据集
  - `load_real_voc_dataset()` - 加载真实VOC数据
- `VOCDatasetLoader` - 批量数据加载器

### 3. train_yolo.py

**主要类:**
- `YOLOTrainer` - 训练器，包含训练和验证逻辑

**主要函数:**
- `train_yolo_on_voc()` - 在VOC数据集上训练
- `evaluate_model()` - 模型评估

### 4. test_yolo.py

**主要函数:**
- `visualize_detection()` - 可视化检测结果
- `visualize_comparison()` - 对比真实标注和检测结果
- `test_on_simulated_data()` - 在模拟数据上测试
- `demo_detection_process()` - 演示完整检测流程

## 快速开始

### 1. 运行演示

```bash
# 检测流程演示
python test_yolo.py --mode demo

# 在模拟数据上测试
python test_yolo.py --mode test --num_samples 5
```

### 2. 训练模型

```bash
# 完整训练
python train_yolo.py

# 或使用Python API
from train_yolo import train_yolo_on_voc

model, trainer, train_dataset, val_dataset = train_yolo_on_voc(
    num_train_samples=100,    # 训练样本数
    num_val_samples=20,       # 验证样本数
    batch_size=4,             # 批次大小
    epochs=10,                # 训练轮数
    learning_rate=0.001,      # 学习率
    save_dir='./models'       # 保存目录
)
```

### 3. 使用模型

```python
from YOLO_v1 import create_tiny_yolo_v1, YOLOPostProcessor
from VOC_dataset import VOCDataset

# 创建模型
model = create_tiny_yolo_v1(S=7, B=2, C=20)

# 创建后处理器
post_processor = YOLOPostProcessor(
    S=7, B=2, C=20,
    image_size=448,
    conf_threshold=0.3,
    nms_threshold=0.4
)

# 准备图像
image = ...  # (3, 448, 448)
image_batch = image[np.newaxis, ...]  # 添加batch维度

# 前向传播
predictions = model.forward(image_batch, training=False)

# 后处理
detected_boxes = post_processor.process(predictions)
```

## 模型架构

### Tiny YOLO v1

```
输入: 448 × 448 × 3

卷积层 1: Conv 7×7×64, stride=2 → MaxPool 2×2
卷积层 2: Conv 3×3×192 → MaxPool 2×2
卷积层 3: Conv 1×1×128 → Conv 3×3×256 → MaxPool 2×2
卷积层 4: Conv 1×1×256 → Conv 3×3×512 → MaxPool 2×2
卷积层 5: Conv 1×1×512 → Conv 3×3×1024 → MaxPool 2×2

全连接层:
  FC: 50176 → 256
  FC: 256 → 4096
  FC: 4096 → 1470

输出: 7 × 7 × 30
  - 30 = 2×5 + 20 (B=2个框, C=20个类别)
  - 每个框: [x, y, w, h, confidence]
  - 加上20个类别概率

总计: 9个卷积层 + 3个全连接层
```

## 输出格式

YOLO输出是一个 `7 × 7 × 30` 的张量：

- **S = 7**: 图像被分成7×7的网格
- **B = 2**: 每个网格预测2个边界框
- **C = 20**: 20个类别（PASCAL VOC）

每个网格单元包含：
- 边界框1: [x, y, w, h, confidence] (5个值)
- 边界框2: [x, y, w, h, confidence] (5个值)
- 类别概率: [p(class_1), ..., p(class_20)] (20个值)

总计: 5 + 5 + 20 = 30

## 损失函数

YOLO损失由四部分组成：

```
Loss = λ_coord × 坐标损失
     + 置信度损失(有目标)
     + λ_noobj × 置信度损失(无目标)
     + 分类损失
```

其中：
- `λ_coord = 5`: 坐标损失权重
- `λ_noobj = 0.5`: 无目标置信度损失权重

## 类别列表 (PASCAL VOC)

```python
['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
 'bus', 'car', 'cat', 'chair', 'cow',
 'diningtable', 'dog', 'horse', 'motorbike', 'person',
 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

## 参数说明

### YOLO 参数
- `S`: 网格大小（默认7）
- `B`: 每个网格的边界框数（默认2）
- `C`: 类别数（默认20）

### 训练参数
- `learning_rate`: 学习率（默认0.001）
- `batch_size`: 批次大小（根据GPU内存调整）
- `epochs`: 训练轮数
- `lambda_coord`: 坐标损失权重（默认5.0）
- `lambda_noobj`: 无目标置信度损失权重（默认0.5）

### 后处理参数
- `conf_threshold`: 置信度阈值（默认0.3）
- `nms_threshold`: NMS IoU阈值（默认0.4）

## 注意事项

1. **GPU内存**: YOLO模型较大，如果GPU内存不足，请减小batch_size
2. **训练时间**: 完整训练需要较长时间，建议使用GPU加速
3. **模拟数据**: 当前使用模拟数据进行演示，真实VOC数据需要单独下载
4. **模型保存**: 训练过程中会自动保存模型，中断后可以继续训练

## 示例输出

训练输出示例：
```
============================================================
Epoch 0
============================================================
  Batch [2/10] Loss: 29.4137 (coord=20.355, conf_obj=3.103, conf_noobj=1.367, class=5.204) Time: 1.09s
  Batch [4/10] Loss: 21.7847 (coord=7.039, conf_obj=1.727, conf_noobj=0.212, class=1.463) Time: 1.09s

训练损失: 18.2341
  坐标: 10.234
  置信度(有): 2.345
  置信度(无): 0.456
  分类: 3.456

验证损失: 15.678
...
```

## 参考

- 论文: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- PASCAL VOC: http://host.robots.ox.ac.uk/pascal/VOC/
