"""
使用训练好的YOLO模型检测口罩
"""
import os
import sys

os.environ['DISABLE_CUPY'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mask_dataset_loader import MaskDataset
from YOLO_v1 import YOLOPostProcessor, to_cpu
from CNN_v4_cupy import CNN
from PIL import Image

# 类别定义
MASK_CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
NUM_CLASSES = 3
COLORS = ['green', 'red', 'orange']  # 对应三个类别

def load_model(model_path='./models_mask/yolo_mask_best.npz'):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    print(f"Loading model from {model_path}")
    model = CNN.load_model(model_path)
    return model

def detect_image(model, image_path, conf_threshold=0.3, nms_threshold=0.5):
    """
    检测单张图像
    
    Args:
        model: 训练好的模型
        image_path: 图像路径
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
    
    Returns:
        detections: 检测结果列表
        original_image: 原始图像
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    
    # 预处理
    image_resized = image.resize((448, 448))
    image_array = np.array(image_resized).astype(np.float64) / 255.0
    image_array = (image_array - 0.5) * 2.0
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # 添加batch维度
    input_batch = image_array[np.newaxis, ...]
    
    # 前向传播
    predictions = model.forward(input_batch, training=False)
    
    # 后处理
    post_processor = YOLOPostProcessor(
        S=7, B=2, C=NUM_CLASSES, 
        image_size=448,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )
    
    detections = post_processor.process(predictions)
    
    return detections[0], image, orig_w, orig_h

def visualize_detection(image_path, output_path=None, model_path='./models_mask/yolo_mask_best.npz'):
    """
    可视化检测结果
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
        model_path: 模型路径
    """
    # 加载模型
    model = load_model(model_path)
    if model is None:
        return
    
    # 检测
    detections, original_image, orig_w, orig_h = detect_image(model, image_path)
    
    # 创建可视化
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(original_image)
    
    print(f"\nDetected {len(detections)} objects:")
    
    for i, det in enumerate(detections):
        x, y, w, h, conf, class_id, class_prob = det
        class_name = MASK_CLASSES[class_id]
        color = COLORS[class_id]
        
        # 映射回原始图像尺寸
        x_orig = x * orig_w / 448
        y_orig = y * orig_h / 448
        w_orig = w * orig_w / 448
        h_orig = h * orig_h / 448
        
        x1 = x_orig - w_orig / 2
        y1 = y_orig - h_orig / 2
        
        # 绘制边界框
        rect = patches.Rectangle(
            (x1, y1), w_orig, h_orig,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        label = f"{class_name}: {conf:.2f}"
        ax.text(
            x1, y1 - 5,
            label,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=10,
            color='white',
            weight='bold'
        )
        
        print(f"  {i+1}. {class_name} (conf: {conf:.2f}) at ({x1:.0f}, {y1:.0f}, {w_orig:.0f}, {h_orig:.0f})")
    
    ax.set_title(f'Mask Detection - {os.path.basename(image_path)}\n{len(detections)} objects detected', 
                 fontsize=12, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nResult saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return detections

def evaluate_on_val_set(model_path='./models_mask/yolo_mask_best.npz', num_samples=10):
    """
    在验证集上评估模型
    
    Args:
        model_path: 模型路径
        num_samples: 评估样本数
    """
    # 加载模型
    model = load_model(model_path)
    if model is None:
        return
    
    # 加载验证集
    val_dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
    val_dataset.load_dataset(split='val')
    
    if len(val_dataset) == 0:
        print("No validation data found")
        return
    
    print(f"\nEvaluating on {min(num_samples, len(val_dataset))} validation samples...")
    
    # 创建输出目录
    output_dir = './mask_detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    for idx in indices:
        image_path = val_dataset.image_paths[idx]
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f'detection_{image_name}')
        
        print(f"\nProcessing: {image_name}")
        
        try:
            detections = visualize_detection(image_path, output_path, model_path)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mask Detection using YOLO v1')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='./models_mask/yolo_mask_best.npz', 
                        help='Path to model file')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate on validation set')
    parser.add_argument('--samples', type=int, default=10, 
                        help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # 在验证集上评估
        evaluate_on_val_set(args.model, args.samples)
    elif args.image:
        # 检测单张图像
        output_dir = './mask_detection_results'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'result_{os.path.basename(args.image)}')
        visualize_detection(args.image, output_path, args.model)
    else:
        # 默认：评估验证集
        print("No image specified. Evaluating on validation set...")
        evaluate_on_val_set(args.model, args.samples)
