"""
简化版Mask Dataset训练脚本
更稳定的配置，避免内存问题
"""
import os
import sys

os.environ['DISABLE_CUPY'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

sys.path.insert(0, '.')

from mask_dataset_loader import MaskDataset, MaskDatasetLoader
from YOLO_v1 import create_tiny_yolo_v1, YOLOLoss, YOLOTargetEncoder, to_cpu

print('='*70)
print('YOLO v1 Mask Detection - Simple Training')
print('='*70)

np.random.seed(42)

# 配置
NUM_CLASSES = 3
S, B = 7, 2
save_dir = './models_mask'
os.makedirs(save_dir, exist_ok=True)

# 加载数据
print("\nLoading datasets...")
train_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
val_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)

train_dataset.load_dataset(split='train')
val_dataset.load_dataset(split='val')

train_stats = train_dataset.get_statistics()
val_stats = val_dataset.get_statistics()

print(f"\nTrain: {train_stats['num_images']} images, {train_stats['num_objects']} objects")
print(f"Val: {val_stats['num_images']} images, {val_stats['num_objects']} objects")

# 数据加载器 - 小batch size
train_loader = MaskDatasetLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = MaskDatasetLoader(val_dataset, batch_size=4, shuffle=False)

# 创建模型
print("\nCreating model...")
model = create_tiny_yolo_v1(S=S, B=B, C=NUM_CLASSES, learning_rate=0.001, _Adam=True)

# 损失函数
loss_fn = YOLOLoss(S=S, B=B, C=NUM_CLASSES, lambda_coord=5.0, lambda_noobj=0.5)
target_encoder = YOLOTargetEncoder(S=S, B=B, C=NUM_CLASSES, image_size=448)

# 训练参数
epochs = 15
best_val_loss = float('inf')
train_losses = []
val_losses = []

print(f"\nTraining {epochs} epochs...")
print(f"Batches per epoch: {len(train_loader)}")

start_time = time.time()

for epoch in range(epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*70}")
    
    # 学习率调整
    lr = 0.001 if epoch < 8 else 0.0005 if epoch < 12 else 0.0001
    model.unified_hyperparam(learning_rate=lr)
    print(f"Learning rate: {lr}")
    
    # 训练
    epoch_losses = []
    for batch_idx, (images, boxes_list, labels_list) in enumerate(train_loader):
        targets = target_encoder.encode(boxes_list, labels_list)
        predictions = model.forward(images, training=True)
        loss, loss_dict = loss_fn.compute_loss(predictions, targets)
        grad = loss_fn.compute_gradient(predictions, targets)
        model.backward(grad)
        
        loss_val = float(to_cpu(loss))
        epoch_losses.append(loss_val)
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss_val:.4f}")
    
    avg_train_loss = np.mean(epoch_losses)
    train_losses.append(avg_train_loss)
    
    # 验证
    val_epoch_losses = []
    for images, boxes_list, labels_list in val_loader:
        targets = target_encoder.encode(boxes_list, labels_list)
        predictions = model.forward(images, training=False)
        loss, _ = loss_fn.compute_loss(predictions, targets)
        val_epoch_losses.append(float(to_cpu(loss)))
    
    avg_val_loss = np.mean(val_epoch_losses)
    val_losses.append(avg_val_loss)
    
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    
    # 保存模型
    model.save_model(os.path.join(save_dir, f'epoch_{epoch}.npz'))
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_model(os.path.join(save_dir, 'yolo_mask_best.npz'))
        print(f"  ✓ Best model saved!")

# 保存最终模型
model.save_model(os.path.join(save_dir, 'yolo_mask_final.npz'))

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', label='Train', linewidth=2)
plt.plot(val_losses, 'r-', label='Val', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=150)
plt.close()

# 报告
total_time = (time.time() - start_time) / 60
report = f"""
Training Report
{'='*50}
Time: {total_time:.1f} minutes
Epochs: {len(train_losses)}
Best Val Loss: {best_val_loss:.4f}
Final Train Loss: {train_losses[-1]:.4f}
Final Val Loss: {val_losses[-1]:.4f}
"""

with open(os.path.join(save_dir, 'report.txt'), 'w') as f:
    f.write(report)

print(f"\n{'='*70}")
print("Training completed!")
print(f"{'='*70}")
print(report)
print(f"Models saved to: {save_dir}")
