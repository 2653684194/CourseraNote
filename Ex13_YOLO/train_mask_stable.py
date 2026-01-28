"""
稳定的Mask Dataset训练脚本
修复数值溢出和NaN问题
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
print('YOLO v1 Mask Detection - Stable Training')
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

# 数据加载器
train_loader = MaskDatasetLoader(train_dataset, batch_size=2, shuffle=True)  # 更小的batch
val_loader = MaskDatasetLoader(val_dataset, batch_size=2, shuffle=False)

# 创建模型 - 使用更小的学习率
print("\nCreating model...")
model = create_tiny_yolo_v1(S=S, B=B, C=NUM_CLASSES, learning_rate=0.0001, _Adam=True)

# 损失函数 - 调整权重
loss_fn = YOLOLoss(S=S, B=B, C=NUM_CLASSES, lambda_coord=2.0, lambda_noobj=0.2)
target_encoder = YOLOTargetEncoder(S=S, B=B, C=NUM_CLASSES, image_size=448)

# 梯度裁剪函数
def clip_gradient(grad, max_norm=1.0):
    """裁剪梯度，防止爆炸"""
    grad_norm = np.sqrt(np.sum(grad ** 2))
    if grad_norm > max_norm:
        grad = grad * (max_norm / grad_norm)
    return grad

# 检查数值有效性
def check_nan_inf(arr, name="array"):
    """检查数组中是否有NaN或Inf"""
    if np.isnan(arr).any():
        print(f"  ⚠️  Warning: NaN detected in {name}")
        return False
    if np.isinf(arr).any():
        print(f"  ⚠️  Warning: Inf detected in {name}")
        return False
    return True

# 训练参数
epochs = 10  # 减少epoch数，先测试稳定性
best_val_loss = float('inf')
train_losses = []
val_losses = []

print(f"\nTraining {epochs} epochs...")
print(f"Batches per epoch: {len(train_loader)}")
print(f"Learning rate: 0.0001 (small for stability)")
print(f"Gradient clipping: enabled (max_norm=1.0)")

start_time = time.time()

for epoch in range(epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*70}")
    
    # 训练
    epoch_losses = []
    nan_count = 0
    
    for batch_idx, (images, boxes_list, labels_list) in enumerate(train_loader):
        # 检查输入数据
        if not check_nan_inf(images, "input images"):
            print(f"  Skipping batch {batch_idx+1} due to invalid input")
            continue
        
        # 编码目标
        targets = target_encoder.encode(boxes_list, labels_list)
        
        if not check_nan_inf(targets, "targets"):
            print(f"  Skipping batch {batch_idx+1} due to invalid targets")
            continue
        
        # 前向传播
        predictions = model.forward(images, training=True)
        
        if not check_nan_inf(predictions, "predictions"):
            print(f"  ⚠️  NaN in predictions, resetting...")
            # 重新初始化模型
            model = create_tiny_yolo_v1(S=S, B=B, C=NUM_CLASSES, learning_rate=0.0001, _Adam=True)
            continue
        
        # 计算损失
        loss, loss_dict = loss_fn.compute_loss(predictions, targets)
        
        if np.isnan(float(to_cpu(loss))) or np.isinf(float(to_cpu(loss))):
            print(f"  ⚠️  NaN/Inf loss at batch {batch_idx+1}, skipping...")
            nan_count += 1
            if nan_count > 5:
                print("  Too many NaN losses, stopping training")
                break
            continue
        
        # 计算梯度
        grad = loss_fn.compute_gradient(predictions, targets)
        
        # 裁剪梯度
        grad = clip_gradient(grad, max_norm=1.0)
        
        if not check_nan_inf(grad, "gradient"):
            print(f"  ⚠️  NaN in gradient, skipping batch...")
            continue
        
        # 反向传播
        model.backward(grad)
        
        loss_val = float(to_cpu(loss))
        epoch_losses.append(loss_val)
        
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss_val:.4f} "
                  f"(coord={loss_dict['coord']:.2f}, conf={loss_dict['conf_obj']:.2f}, "
                  f"class={loss_dict['class']:.2f})")
    
    if len(epoch_losses) == 0:
        print("  No valid losses in this epoch, skipping...")
        continue
    
    avg_train_loss = np.mean(epoch_losses)
    train_losses.append(avg_train_loss)
    
    # 验证
    val_epoch_losses = []
    for images, boxes_list, labels_list in val_loader:
        targets = target_encoder.encode(boxes_list, labels_list)
        predictions = model.forward(images, training=False)
        loss, _ = loss_fn.compute_loss(predictions, targets)
        loss_val = float(to_cpu(loss))
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            val_epoch_losses.append(loss_val)
    
    avg_val_loss = np.mean(val_epoch_losses) if val_epoch_losses else float('inf')
    val_losses.append(avg_val_loss)
    
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    
    # 保存模型
    model.save_model(os.path.join(save_dir, f'epoch_{epoch}.npz'))
    
    if avg_val_loss < best_val_loss and not (np.isnan(avg_val_loss) or np.isinf(avg_val_loss)):
        best_val_loss = avg_val_loss
        model.save_model(os.path.join(save_dir, 'yolo_mask_best.npz'))
        print(f"  ✓ Best model saved!")

# 保存最终模型
model.save_model(os.path.join(save_dir, 'yolo_mask_final.npz'))

# 绘制损失曲线
if len(train_losses) > 0:
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
Epochs completed: {len(train_losses)}
Best Val Loss: {best_val_loss:.4f}
Final Train Loss: {train_losses[-1]:.4f if train_losses else 'N/A'}
Final Val Loss: {val_losses[-1]:.4f if val_losses else 'N/A'}
"""

with open(os.path.join(save_dir, 'report.txt'), 'w') as f:
    f.write(report)

print(f"\n{'='*70}")
print("Training completed!")
print(f"{'='*70}")
print(report)
print(f"Models saved to: {save_dir}")
