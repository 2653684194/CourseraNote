"""
评估演示脚本 - 展示如何可视化和衡量结果
"""
import os
import sys

os.environ['DISABLE_CUPY'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mask_dataset_loader import MaskDataset
from PIL import Image

# 类别定义
MASK_CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
COLORS = ['#2ecc71', '#e74c3c', '#f39c12']


def visualize_dataset_samples(num_samples=5):
    """
    可视化数据集样本 - 展示真实标注
    """
    print('='*70)
    print('Visualizing Mask Dataset Samples')
    print('='*70)
    
    # 加载数据集
    dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
    dataset.load_dataset(split='train')
    
    if len(dataset) == 0:
        print("No data found!")
        return
    
    # 创建输出目录
    output_dir = './dataset_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        image_path = dataset.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label_data = dataset.labels[idx]
        
        ax = axes[i]
        ax.imshow(image)
        
        # 绘制边界框
        for box, label in zip(label_data['boxes'], label_data['labels']):
            x, y, w, h = box
            color = COLORS[label]
            
            rect = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x - w/2, y - h/2 - 5, MASK_CLASSES[label],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                   fontsize=9, color='white', weight='bold')
        
        ax.set_title(f'Sample {i+1}: {os.path.basename(image_path)}\n'
                    f'{len(label_data["labels"])} objects', fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Mask Dataset Samples with Ground Truth Annotations', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'dataset_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    # 显示统计信息
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['num_images']}")
    print(f"  Total objects: {stats['num_objects']}")
    print(f"  Class distribution:")
    for cls, count in stats['class_counts'].items():
        percentage = count / stats['num_objects'] * 100
        print(f"    {cls}: {count} ({percentage:.1f}%)")


def plot_class_distribution():
    """
    绘制类别分布图
    """
    print('\n' + '='*70)
    print('Plotting Class Distribution')
    print('='*70)
    
    # 加载训练集和验证集
    train_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
    val_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
    
    train_dataset.load_dataset(split='train')
    val_dataset.load_dataset(split='val')
    
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练集分布
    train_counts = [train_stats['class_counts'].get(cls, 0) for cls in MASK_CLASSES]
    axes[0].bar(MASK_CLASSES, train_counts, color=COLORS, alpha=0.7)
    axes[0].set_title(f'Training Set Distribution\n({train_stats["num_images"]} images, {train_stats["num_objects"]} objects)',
                     fontsize=12, weight='bold')
    axes[0].set_ylabel('Number of Objects')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(train_counts):
        axes[0].text(i, v + max(train_counts)*0.01, str(v), ha='center', fontsize=10, weight='bold')
    
    # 验证集分布
    val_counts = [val_stats['class_counts'].get(cls, 0) for cls in MASK_CLASSES]
    axes[1].bar(MASK_CLASSES, val_counts, color=COLORS, alpha=0.7)
    axes[1].set_title(f'Validation Set Distribution\n({val_stats["num_images"]} images, {val_stats["num_objects"]} objects)',
                     fontsize=12, weight='bold')
    axes[1].set_ylabel('Number of Objects')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(val_counts):
        axes[1].text(i, v + max(val_counts)*0.01, str(v), ha='center', fontsize=10, weight='bold')
    
    plt.tight_layout()
    
    output_dir = './dataset_visualization'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution saved to: {output_path}")


def demonstrate_metrics_calculation():
    """
    演示指标计算过程
    """
    print('\n' + '='*70)
    print('Demonstrating Metrics Calculation')
    print('='*70)
    
    # 模拟检测结果
    print("\nExample Detection Results:")
    print("-" * 50)
    
    # 模拟数据
    examples = [
        {
            'image': 'sample_001.jpg',
            'gt': [('with_mask', [100, 100, 50, 50]), ('without_mask', [200, 150, 60, 60])],
            'pred': [('with_mask', 0.85, [105, 102, 48, 52]), ('without_mask', 0.92, [198, 148, 62, 58])]
        },
        {
            'image': 'sample_002.jpg',
            'gt': [('with_mask', [150, 200, 80, 80])],
            'pred': [('with_mask', 0.78, [155, 205, 75, 75]), ('with_mask', 0.45, [300, 300, 50, 50])]
        }
    ]
    
    total_tp = {0: 0, 1: 0, 2: 0}
    total_fp = {0: 0, 1: 0, 2: 0}
    total_fn = {0: 0, 1: 0, 2: 0}
    
    class_map = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
    
    for ex in examples:
        print(f"\nImage: {ex['image']}")
        print(f"  Ground Truth: {len(ex['gt'])} objects")
        for cls, box in ex['gt']:
            print(f"    - {cls} at {box}")
        
        print(f"  Predictions: {len(ex['pred'])} objects")
        for cls, conf, box in ex['pred']:
            status = "✓" if conf > 0.5 else "✗ (low confidence)"
            print(f"    - {cls} ({conf:.2f}) at {box} {status}")
        
        # 简化的TP/FP/FN计算
        matched = set()
        for pred_cls, pred_conf, pred_box in ex['pred']:
            if pred_conf < 0.5:
                continue
            
            pred_cls_id = class_map[pred_cls]
            best_iou = 0
            best_gt = None
            
            for i, (gt_cls, gt_box) in enumerate(ex['gt']):
                if i in matched or gt_cls != pred_cls:
                    continue
                # 简化的IoU计算
                iou = 0.7  # 假设
                if iou > best_iou:
                    best_iou = iou
                    best_gt = i
            
            if best_iou > 0.5 and best_gt is not None:
                total_tp[pred_cls_id] += 1
                matched.add(best_gt)
            else:
                total_fp[pred_cls_id] += 1
        
        for i, (gt_cls, _) in enumerate(ex['gt']):
            if i not in matched:
                total_fn[class_map[gt_cls]] += 1
    
    print("\n" + "="*50)
    print("Summary Statistics:")
    print("="*50)
    
    for i, cls in enumerate(MASK_CLASSES):
        tp = total_tp[i]
        fp = total_fp[i]
        fn = total_fn[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{cls}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")


def create_evaluation_guide():
    """
    创建评估指南
    """
    guide = """
# YOLO Mask Detection 评估指南

## 评估指标说明

### 1. Precision (精确率)
- 定义：检测出的框中，正确的比例
- 公式：Precision = TP / (TP + FP)
- 意义：越高表示误检越少

### 2. Recall (召回率)
- 定义：真实物体中，被检测出的比例
- 公式：Recall = TP / (TP + FN)
- 意义：越高表示漏检越少

### 3. F1-Score
- 定义：Precision和Recall的调和平均
- 公式：F1 = 2 * (Precision * Recall) / (Precision + Recall)
- 意义：综合考虑精确率和召回率

### 4. mAP (mean Average Precision)
- 定义：所有类别AP的平均值
- 意义：目标检测的标准评估指标

## 评估结果解读

### 优秀结果
- mAP > 0.8
- 各类别F1 > 0.75

### 良好结果
- mAP > 0.6
- 各类别F1 > 0.6

### 需要改进
- mAP < 0.5
- 某些类别F1 < 0.5

## 可视化结果

1. **对比图** (comparisons/)
   - 左侧：真实标注 (Ground Truth)
   - 右侧：模型预测 (Predictions)
   - 绿色：with_mask
   - 红色：without_mask
   - 橙色：mask_weared_incorrect

2. **指标图** (metrics.png)
   - Precision、Recall、F1柱状图
   - 详细指标表格

## 运行评估

```bash
# 完整评估
python evaluate_mask_model.py

# 指定置信度阈值
python evaluate_mask_model.py --conf 0.5

# 只评估部分样本
python evaluate_mask_model.py --samples 20
```
"""
    
    output_path = './evaluation_results/EVALUATION_GUIDE.md'
    os.makedirs('./evaluation_results', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(guide)
    
    print(f"\nEvaluation guide saved to: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Mask Detection Evaluation Demo")
    print("="*70)
    
    # 1. 可视化数据集样本
    visualize_dataset_samples(num_samples=6)
    
    # 2. 绘制类别分布
    plot_class_distribution()
    
    # 3. 演示指标计算
    demonstrate_metrics_calculation()
    
    # 4. 创建评估指南
    create_evaluation_guide()
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - dataset_visualization/dataset_samples.png")
    print("  - dataset_visualization/class_distribution.png")
    print("  - evaluation_results/EVALUATION_GUIDE.md")
    print("\nTo run full evaluation after training:")
    print("  python evaluate_mask_model.py")
