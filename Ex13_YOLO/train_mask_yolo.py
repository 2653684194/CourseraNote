"""
使用YOLO v1训练Mask Dataset
口罩检测目标检测
"""
import os
import sys

# 强制禁用CuPy
os.environ['DISABLE_CUPY'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mask_dataset_loader import MaskDataset, MaskDatasetLoader
from YOLO_v1 import create_tiny_yolo_v1, YOLOLoss, YOLOTargetEncoder, YOLOPostProcessor, to_cpu
from CNN_v4_cupy import CNN

print('='*70)
print('YOLO v1 Training on Mask Dataset')
print('Classes: with_mask, without_mask, mask_weared_incorrect')
print('='*70)

# 设置随机种子
np.random.seed(42)

# Mask Dataset 配置
MASK_CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
NUM_CLASSES = 3
S, B = 7, 2  # Grid size, boxes per cell

# 创建保存目录
save_dir = './models_mask'
os.makedirs(save_dir, exist_ok=True)

# 加载数据集
print("\nLoading Mask Dataset...")
train_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
val_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)

train_dataset.load_dataset(split='train')
val_dataset.load_dataset(split='val')

# 显示统计信息
train_stats = train_dataset.get_statistics()
val_stats = val_dataset.get_statistics()

print(f"\nDataset Statistics:")
print(f"  Training: {train_stats['num_images']} images, {train_stats['num_objects']} objects")
print(f"    Class distribution: {train_stats['class_counts']}")
print(f"  Validation: {val_stats['num_images']} images, {val_stats['num_objects']} objects")
print(f"    Class distribution: {val_stats['class_counts']}")

# 创建数据加载器
batch_size = 8
print(f"\nCreating data loaders (batch_size={batch_size})...")
train_loader = MaskDatasetLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = MaskDatasetLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 创建模型 - 针对3个类别
print("\nCreating Tiny YOLO model for mask detection...")
model = create_tiny_yolo_v1(S=S, B=B, C=NUM_CLASSES, learning_rate=0.001, _Adam=True)

# 创建损失函数 - 针对mask数据集调参
# 由于mask数据集类别不平衡，调整权重
loss_fn = YOLOLoss(
    S=S, B=B, C=NUM_CLASSES,
    lambda_coord=5.0,      # 坐标损失权重
    lambda_noobj=0.5       # 无物体损失权重
)

# 创建目标编码器
target_encoder = YOLOTargetEncoder(S=S, B=B, C=NUM_CLASSES, image_size=448)

# 训练配置
epochs = 20
patience = 5  # 早停耐心值
min_delta = 0.001

print(f"\nTraining Configuration:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: 0.001 (with decay)")
print(f"  Early stopping patience: {patience}")
print(f"  Lambda coord: {loss_fn.lambda_coord}")
print(f"  Lambda noobj: {loss_fn.lambda_noobj}")

# 训练循环
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0
best_epoch = 0

print(f"\n{'='*70}")
print("Starting training...")
print(f"{'='*70}")

start_time = time.time()

try:
    for epoch in range(epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{epochs}")
        
        # 学习率衰减
        if epoch < 10:
            lr = 0.001
        elif epoch < 15:
            lr = 0.0005
        else:
            lr = 0.0001
        
        model.unified_hyperparam(learning_rate=lr)
        
        # 训练阶段
        epoch_losses = []
        num_batches = len(train_loader)
        
        for batch_idx, (images, boxes_list, labels_list) in enumerate(train_loader):
            batch_start = time.time()
            
            # 编码目标
            targets = target_encoder.encode(boxes_list, labels_list)
            
            # 前向传播
            predictions = model.forward(images, training=True)
            
            # 计算损失
            loss, loss_dict = loss_fn.compute_loss(predictions, targets)
            
            # 计算梯度
            grad = loss_fn.compute_gradient(predictions, targets)
            
            # 反向传播
            model.backward(grad)
            
            # 记录损失
            loss_val = float(to_cpu(loss))
            epoch_losses.append(loss_val)
            
            batch_time = time.time() - batch_start
            
            # 打印进度
            if (batch_idx + 1) % max(1, num_batches // 3) == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
                print(f"  Batch [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f} "
                      f"(coord={loss_dict['coord']:.2f}, conf={loss_dict['conf_obj']:.2f}, "
                      f"class={loss_dict['class']:.2f}) Time: {batch_time:.2f}s")
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        print("  Validating...")
        val_epoch_losses = []
        for images, boxes_list, labels_list in val_loader:
            targets = target_encoder.encode(boxes_list, labels_list)
            predictions = model.forward(images, training=False)
            loss, _ = loss_fn.compute_loss(predictions, targets)
            val_epoch_losses.append(float(to_cpu(loss)))
        
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"  Epoch {epoch} Summary:")
        print(f"    Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}min)")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss: {avg_val_loss:.4f}")
        
        # 保存模型
        model_path = os.path.join(save_dir, f'yolo_mask_epoch_{epoch}.npz')
        model.save_model(model_path)
        
        # 检查最佳模型
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            best_model_path = os.path.join(save_dir, 'yolo_mask_best.npz')
            model.save_model(best_model_path)
            print(f"    ✓✓✓ NEW BEST MODEL! (val_loss: {avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    No improvement ({epochs_no_improve}/{patience}) Best: {best_val_loss:.4f} @ epoch {best_epoch}")
        
        # 早停检查
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered!")
            break

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")

# 保存最终模型
final_model_path = os.path.join(save_dir, 'yolo_mask_final.npz')
model.save_model(final_model_path)

# 绘制损失曲线
print("\n" + "="*70)
print("Training completed!")
print("="*70)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(len(train_losses)), train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(range(len(val_losses)), val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('YOLO Mask Detection - Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(len(train_losses)), train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(range(len(val_losses)), val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
loss_plot_path = os.path.join(save_dir, 'training_loss.png')
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
plt.close()

# 保存训练报告
report = f"""
YOLO v1 Mask Detection Training Report
{'='*70}
Training Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Dataset:
  Classes: {MASK_CLASSES}
  Training: {train_stats['num_images']} images, {train_stats['num_objects']} objects
  Validation: {val_stats['num_images']} images, {val_stats['num_objects']} objects

Configuration:
  Epochs: {len(train_losses)} (max: {epochs})
  Batch size: {batch_size}
  Grid size: {S}x{S}
  Boxes per cell: {B}
  Lambda coord: {loss_fn.lambda_coord}
  Lambda noobj: {loss_fn.lambda_noobj}

Results:
  Best epoch: {best_epoch}
  Best validation loss: {best_val_loss:.4f}
  Final training loss: {train_losses[-1]:.4f}
  Final validation loss: {val_losses[-1]:.4f}
  Total training time: {(time.time() - start_time)/60:.1f} minutes

Saved Models:
  - yolo_mask_best.npz (best model)
  - yolo_mask_final.npz (final model)
  - training_loss.png (loss curve)
"""

with open(os.path.join(save_dir, 'training_report.txt'), 'w') as f:
    f.write(report)

print(f"\n{report}")
print(f"\nAll files saved to: {os.path.abspath(save_dir)}")
