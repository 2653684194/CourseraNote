"""
全面评估YOLO口罩检测模型
包含可视化、指标计算和错误分析
"""
import os
import sys

os.environ['DISABLE_CUPY'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mask_dataset_loader import MaskDataset
from YOLO_v1 import YOLOPostProcessor, to_cpu, YOLOTargetEncoder
from CNN_v4_cupy import CNN
from PIL import Image

# 类别定义
MASK_CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
NUM_CLASSES = 3
COLORS = ['#2ecc71', '#e74c3c', '#f39c12']  # 绿色、红色、橙色


def compute_iou(box1, box2):
    """
    计算两个边界框的IoU
    box: [x, y, w, h] (中心坐标, 宽高)
    """
    # 转换为角点坐标
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2
    
    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2
    
    # 交集区域
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 并集区域
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    union_area = b1_area + b2_area - inter_area
    
    iou = inter_area / (union_area + 1e-8)
    return iou


def evaluate_detection(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """
    评估单张图像的检测结果
    
    Returns:
        tp: 每个类别的True Positives
        fp: 每个类别的False Positives
        fn: 每个类别的False Negatives
        precision: 每个类别的Precision
        recall: 每个类别的Recall
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    # 标记已匹配的GT
    matched_gt = set()
    
    # 对每个预测框
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        best_iou = 0
        best_gt_idx = -1
        
        # 找到最佳匹配的GT
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in matched_gt:
                continue
            if gt_label != pred_label:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # 判断是否匹配成功
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[pred_label] += 1
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_label] += 1
    
    # 计算FN
    for gt_idx, gt_label in enumerate(gt_labels):
        if gt_idx not in matched_gt:
            fn[gt_label] += 1
    
    return tp, fp, fn


def calculate_metrics(all_tp, all_fp, all_fn):
    """计算整体指标"""
    metrics = {}
    
    for class_id in range(NUM_CLASSES):
        tp = sum(tp_dict.get(class_id, 0) for tp_dict in all_tp)
        fp = sum(fp_dict.get(class_id, 0) for fp_dict in all_fp)
        fn = sum(fn_dict.get(class_id, 0) for fn_dict in all_fn)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics[MASK_CLASSES[class_id]] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    
    # 计算mAP
    map_value = np.mean([m['Precision'] for m in metrics.values()])
    metrics['mAP'] = map_value
    
    return metrics


def visualize_comparison(image_path, gt_data, predictions, output_path, orig_w, orig_h):
    """
    可视化对比：真实标注 vs 预测结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 左图：真实标注
    ax1.imshow(image)
    ax1.set_title('Ground Truth', fontsize=14, weight='bold')
    
    gt_boxes = gt_data['boxes']
    gt_labels = gt_data['labels']
    
    for box, label in zip(gt_boxes, gt_labels):
        x, y, w, h = box
        color = COLORS[label]
        
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=2, edgecolor=color, facecolor='none', linestyle='--'
        )
        ax1.add_patch(rect)
        ax1.text(x - w/2, y - h/2 - 5, MASK_CLASSES[label],
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                fontsize=9, color='white', weight='bold')
    
    ax1.axis('off')
    
    # 右图：预测结果
    ax2.imshow(image)
    ax2.set_title('Predictions', fontsize=14, weight='bold')
    
    for det in predictions:
        x, y, w, h, conf, class_id, class_prob = det
        
        # 映射回原始尺寸
        x_orig = x * orig_w / 448
        y_orig = y * orig_h / 448
        w_orig = w * orig_w / 448
        h_orig = h * orig_h / 448
        
        x1 = x_orig - w_orig / 2
        y1 = y_orig - h_orig / 2
        
        color = COLORS[class_id]
        
        rect = patches.Rectangle(
            (x1, y1), w_orig, h_orig,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 5, f"{MASK_CLASSES[class_id]}: {conf:.2f}",
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                fontsize=9, color='white', weight='bold')
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics(metrics, output_path):
    """绘制评估指标图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    classes = list(MASK_CLASSES)
    precision = [metrics[c]['Precision'] for c in classes]
    recall = [metrics[c]['Recall'] for c in classes]
    f1 = [metrics[c]['F1-Score'] for c in classes]
    
    # Precision
    axes[0, 0].bar(classes, precision, color=COLORS, alpha=0.7)
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision by Class')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(precision):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Recall
    axes[0, 1].bar(classes, recall, color=COLORS, alpha=0.7)
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall by Class')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(recall):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # F1-Score
    axes[1, 0].bar(classes, f1, color=COLORS, alpha=0.7)
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score by Class')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # 综合指标表格
    axes[1, 1].axis('off')
    table_data = []
    for cls in classes:
        m = metrics[cls]
        table_data.append([
            cls,
            f"{m['TP']}",
            f"{m['FP']}",
            f"{m['FN']}",
            f"{m['Precision']:.3f}",
            f"{m['Recall']:.3f}",
            f"{m['F1-Score']:.3f}"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Detailed Metrics', fontsize=12, weight='bold', pad=20)
    
    plt.suptitle(f'Mask Detection Evaluation\nmAP: {metrics["mAP"]:.4f}', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def full_evaluation(model_path='./models_mask/yolo_mask_best.npz', 
                   conf_threshold=0.3, 
                   iou_threshold=0.5,
                   max_samples=None):
    """
    完整评估流程
    """
    print('='*70)
    print('YOLO Mask Detection - Full Evaluation')
    print('='*70)
    
    # 加载模型
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"\nLoading model: {model_path}")
    model = CNN.load_model(model_path)
    
    # 加载验证集
    print("\nLoading validation dataset...")
    val_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
    val_dataset.load_dataset(split='val')
    
    if len(val_dataset) == 0:
        print("No validation data found!")
        return
    
    num_samples = min(max_samples, len(val_dataset)) if max_samples else len(val_dataset)
    print(f"Evaluating on {num_samples} samples...")
    
    # 创建输出目录
    output_dir = './evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    # 初始化统计
    all_tp = []
    all_fp = []
    all_fn = []
    
    # 创建后处理器
    post_processor = YOLOPostProcessor(
        S=7, B=2, C=NUM_CLASSES,
        image_size=448,
        conf_threshold=conf_threshold,
        nms_threshold=0.5
    )
    
    # 目标编码器（用于获取GT的target格式）
    target_encoder = YOLOTargetEncoder(S=7, B=2, C=NUM_CLASSES, image_size=448)
    
    # 评估每张图像
    for idx in range(num_samples):
        image, label_data = val_dataset[idx]
        image_path = val_dataset.image_paths[idx]
        
        # 获取GT
        gt_boxes = label_data['boxes']
        gt_labels = label_data['labels']
        
        # 预处理图像
        image_batch = image[np.newaxis, ...]
        
        # 预测
        predictions = model.forward(image_batch, training=False)
        detections = post_processor.process(predictions)
        pred_dets = detections[0]
        
        # 提取预测框
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        
        for det in pred_dets:
            x, y, w, h, conf, class_id, class_prob = det
            pred_boxes.append([x, y, w, h])
            pred_labels.append(class_id)
            pred_scores.append(conf)
        
        # 评估
        tp, fp, fn = evaluate_detection(
            gt_boxes, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            iou_threshold
        )
        
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        
        # 可视化对比（前10张）
        if idx < 10:
            orig_image = Image.open(image_path)
            orig_w, orig_h = orig_image.size
            
            output_path = os.path.join(output_dir, 'comparisons', f'comparison_{idx:03d}.png')
            visualize_comparison(image_path, label_data, pred_dets, output_path, orig_w, orig_h)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{num_samples} images")
    
    # 计算指标
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_tp, all_fp, all_fn)
    
    # 打印结果
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    
    for cls in MASK_CLASSES:
        m = metrics[cls]
        print(f"\n{cls}:")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall:    {m['Recall']:.4f}")
        print(f"  F1-Score:  {m['F1-Score']:.4f}")
        print(f"  TP: {m['TP']}, FP: {m['FP']}, FN: {m['FN']}")
    
    print(f"\nmAP: {metrics['mAP']:.4f}")
    
    # 保存指标图表
    metrics_path = os.path.join(output_dir, 'metrics.png')
    plot_metrics(metrics, metrics_path)
    print(f"\nMetrics visualization saved to: {metrics_path}")
    
    # 保存详细报告
    report = {
        'model_path': model_path,
        'num_samples': num_samples,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Detailed report saved to: {report_path}")
    print(f"Comparison images saved to: {os.path.join(output_dir, 'comparisons')}")
    print("="*70)
    
    return metrics


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO Mask Detection Model')
    parser.add_argument('--model', type=str, default='./models_mask/yolo_mask_best.npz',
                       help='Path to model file')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for evaluation')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    metrics = full_evaluation(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_samples=args.samples
    )
