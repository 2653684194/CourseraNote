"""
PASCAL VOC 数据集加载器和模拟器
基于 YOLO v1 实现
"""

import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
from typing import List, Tuple, Dict, Optional
import sys

# 导入 YOLO 实现
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from YOLO_v1 import xp, to_cpu, to_gpu, PASCAL_VOC_CLASSES


class VOCDataset:
    """
    PASCAL VOC 数据集加载器
    支持真实 VOC 数据和模拟数据
    """
    
    def __init__(self, data_dir: str, image_size: int = 448, S: int = 7, B: int = 2, C: int = 20):
        self.data_dir = data_dir
        self.image_size = image_size
        self.S = S
        self.B = B
        self.C = C
        self.classes = PASCAL_VOC_CLASSES
        
        # 类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 数据存储
        self.images = []
        self.annotations = []
        self.image_paths = []
        
    def create_simulated_dataset(self, num_samples: int = 100, split: str = 'train'):
        """
        创建模拟的 VOC 数据集
        
        Args:
            num_samples: 样本数量
            split: 数据集分割 ('train', 'val', 'test')
        """
        print(f"创建模拟 {split} 数据集: {num_samples} 张图像")
        
        self.images = []
        self.annotations = []
        self.image_paths = []
        
        for i in range(num_samples):
            # 创建模拟图像 (3, H, W)
            image = self._create_simulated_image(i)
            
            # 创建模拟标注
            annotation = self._create_simulated_annotation(i)
            
            self.images.append(image)
            self.annotations.append(annotation)
            self.image_paths.append(f"simulated_{split}_{i:06d}.jpg")
            
            if (i + 1) % 20 == 0:
                print(f"  已创建 {i + 1}/{num_samples} 张图像")
        
        print(f"模拟数据集创建完成: {len(self.images)} 张图像")
        
    def _create_simulated_image(self, index: int) -> np.ndarray:
        """创建模拟图像"""
        # 创建基础图像 (3, 448, 448)
        image = np.random.randn(3, self.image_size, self.image_size).astype(np.float32) * 0.1
        
        # 添加一些结构化内容来模拟真实图像
        # 创建一些简单的几何形状作为"物体"
        num_objects = random.randint(1, 4)
        
        for obj_idx in range(num_objects):
            # 随机选择物体类型（矩形、圆形等）
            obj_type = random.choice(['rectangle', 'circle', 'ellipse'])
            
            # 随机位置和大小
            x = random.randint(50, self.image_size - 50)
            y = random.randint(50, self.image_size - 50)
            w = random.randint(30, 150)
            h = random.randint(30, 150)
            
            # 随机颜色（简单的RGB值）
            color = np.random.rand(3) * 0.8 + 0.2
            
            if obj_type == 'rectangle':
                self._add_rectangle(image, x, y, w, h, color)
            elif obj_type == 'circle':
                self._add_circle(image, x, y, min(w, h), color)
            else:  # ellipse
                self._add_ellipse(image, x, y, w, h, color)
        
        # 添加一些噪声
        image += np.random.randn(*image.shape) * 0.02
        
        # 确保值在合理范围内
        image = np.clip(image, -1.0, 1.0)
        
        return image
    
    def _add_rectangle(self, image: np.ndarray, x: int, y: int, w: int, h: int, color: np.ndarray):
        """在图像上添加矩形"""
        x1, x2 = max(0, x - w//2), min(self.image_size, x + w//2)
        y1, y2 = max(0, y - h//2), min(self.image_size, y + h//2)
        
        for c in range(3):
            image[c, y1:y2, x1:x2] = color[c]
    
    def _add_circle(self, image: np.ndarray, x: int, y: int, r: int, color: np.ndarray):
        """在图像上添加圆形"""
        Y, X = np.ogrid[:self.image_size, :self.image_size]
        mask = (X - x)**2 + (Y - y)**2 <= r**2
        
        for c in range(3):
            image[c][mask] = color[c]
    
    def _add_ellipse(self, image: np.ndarray, x: int, y: int, w: int, h: int, color: np.ndarray):
        """在图像上添加椭圆"""
        Y, X = np.ogrid[:self.image_size, :self.image_size]
        mask = ((X - x)**2 / (w/2)**2 + (Y - y)**2 / (h/2)**2) <= 1
        
        for c in range(3):
            image[c][mask] = color[c]
    
    def _create_simulated_annotation(self, index: int) -> Dict:
        """创建模拟标注"""
        annotation = {
            'filename': f"simulated_{index:06d}.jpg",
            'size': {
                'width': self.image_size,
                'height': self.image_size,
                'depth': 3
            },
            'objects': []
        }
        
        # 随机物体数量
        num_objects = random.randint(1, 4)
        
        for obj_idx in range(num_objects):
            # 随机选择类别
            class_name = random.choice(self.classes)
            
            # 随机边界框（确保在图像范围内）
            x = random.randint(50, self.image_size - 50)
            y = random.randint(50, self.image_size - 50)
            w = random.randint(30, 150)
            h = random.randint(30, 150)
            
            # 确保边界框不超出图像
            x1 = max(0, x - w//2)
            y1 = max(0, y - h//2)
            x2 = min(self.image_size, x + w//2)
            y2 = min(self.image_size, y + h//2)
            
            # 重新计算中心坐标和宽高
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            obj = {
                'name': class_name,
                'difficult': random.choice([0, 1]),
                'bndbox': {
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'x': center_x,  # 中心坐标
                    'y': center_y,
                    'w': width,
                    'h': height
                }
            }
            
            annotation['objects'].append(obj)
        
        return annotation
    
    def load_real_voc_dataset(self, image_set: str = 'train', max_samples: int = None):
        """
        加载真实的 VOC 数据集
        
        Args:
            image_set: 数据集分割 ('train', 'val', 'trainval', 'test')
            max_samples: 最大加载样本数（None表示加载全部）
        """
        print(f"加载真实 VOC 数据集: {image_set}")
        
        # VOC 数据集的路径结构（根据实际目录结构调整）
        # 尝试两种可能的路径结构
        if image_set == 'test':
            voc_dir = os.path.join(self.data_dir, 'VOCtest_06-Nov-2007')
        else:
            voc_dir = os.path.join(self.data_dir, 'VOCtrainval_06-Nov-2007')
        
        images_dir = os.path.join(voc_dir, 'JPEGImages')
        annotations_dir = os.path.join(voc_dir, 'Annotations')
        image_sets_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
        
        # 检查路径是否存在
        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            # 尝试备用路径结构
            images_dir = os.path.join(self.data_dir, 'VOCdevkit', 'VOC2007', 'JPEGImages')
            annotations_dir = os.path.join(self.data_dir, 'VOCdevkit', 'VOC2007', 'Annotations')
            image_sets_dir = os.path.join(self.data_dir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main')
            
            if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
                print(f"真实 VOC 数据集未找到，使用模拟数据")
                print(f"  尝试路径: {voc_dir}")
                return False
        
        # 读取图像列表
        image_set_file = os.path.join(image_sets_dir, f'{image_set}.txt')
        if not os.path.exists(image_set_file):
            print(f"图像集文件 {image_set_file} 未找到")
            return False
        
        with open(image_set_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
        
        # 限制样本数
        if max_samples is not None:
            image_ids = image_ids[:max_samples]
        
        print(f"找到 {len(image_ids)} 张图像")
        
        # 加载图像和标注
        for i, image_id in enumerate(image_ids):
            # 图像路径
            image_path = os.path.join(images_dir, f'{image_id}.jpg')
            annotation_path = os.path.join(annotations_dir, f'{image_id}.xml')
            
            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                continue
            
            try:
                # 加载图像
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self.image_size, self.image_size))
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) * 2.0  # 归一化到 [-1, 1]
                image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                
                # 加载标注
                annotation = self._parse_voc_annotation(annotation_path)
                
                self.images.append(image)
                self.annotations.append(annotation)
                self.image_paths.append(image_path)
                
            except Exception as e:
                print(f"加载 {image_id} 失败: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                print(f"  已加载 {i + 1}/{len(image_ids)} 张图像")
        
        print(f"成功加载 {len(self.images)} 张图像")
        return True
    
    def _parse_voc_annotation(self, annotation_path: str) -> Dict:
        """解析 VOC XML 标注文件"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        annotation = {
            'filename': root.find('filename').text,
            'size': {
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'depth': int(root.find('size/depth').text)
            },
            'objects': []
        }
        
        # 解析所有物体
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            
            # 获取边界框
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # 计算中心坐标和宽高
            x = (xmin + xmax) // 2
            y = (ymin + ymax) // 2
            w = xmax - xmin
            h = ymax - ymin
            
            obj = {
                'name': obj_name,
                'difficult': difficult,
                'bndbox': {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                }
            }
            
            annotation['objects'].append(obj)
        
        return annotation
    
    def get_boxes_and_labels(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定索引的边界框和标签
        
        Returns:
            boxes: (num_objects, 4) [x_center, y_center, width, height] in image coordinates
            labels: (num_objects,) class indices
        """
        annotation = self.annotations[index]
        objects = annotation['objects']
        
        boxes = []
        labels = []
        
        for obj in objects:
            # 获取边界框信息
            bbox = obj['bndbox']
            x = bbox['x']
            y = bbox['y']
            w = bbox['w']
            h = bbox['h']
            
            # 获取类别索引
            class_name = obj['name']
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                
                boxes.append([x, y, w, h])
                labels.append(class_idx)
        
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,), dtype=int)
        
        return np.array(boxes), np.array(labels)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取单个样本
        
        Returns:
            image: (3, H, W) 图像数据
            boxes: (num_objects, 4) 边界框 [x, y, w, h]
            labels: (num_objects,) 类别标签
        """
        image = self.images[index]
        boxes, labels = self.get_boxes_and_labels(index)
        
        return image, boxes, labels
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        获取批次数据
        
        Returns:
            images: (batch_size, 3, H, W)
            boxes_list: List of (num_objects, 4) arrays
            labels_list: List of (num_objects,) arrays
        """
        images = []
        boxes_list = []
        labels_list = []
        
        for idx in indices:
            image, boxes, labels = self[idx]
            images.append(image)
            boxes_list.append(boxes)
            labels_list.append(labels)
        
        # 转换为numpy数组
        images = np.array(images)
        
        return images, boxes_list, labels_list
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        total_objects = 0
        class_counts = {cls: 0 for cls in self.classes}
        
        for annotation in self.annotations:
            for obj in annotation['objects']:
                class_name = obj['name']
                if class_name in class_counts:
                    class_counts[class_name] += 1
                    total_objects += 1
        
        return {
            'num_images': len(self.images),
            'num_objects': total_objects,
            'objects_per_image': total_objects / len(self.images) if len(self.images) > 0 else 0,
            'class_distribution': class_counts
        }


class VOCDatasetLoader:
    """VOC数据集加载器，支持批量加载"""
    
    def __init__(self, dataset: VOCDataset, batch_size: int = 16, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        """迭代器"""
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield self.dataset.get_batch(batch_indices)
    
    def __len__(self) -> int:
        """返回批次数量"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    # 测试数据集
    print("测试 VOC 数据集加载器...")
    
    # 创建模拟数据集
    dataset = VOCDataset(data_dir="./data", image_size=448)
    dataset.create_simulated_dataset(num_samples=50, split='train')
    
    # 获取统计信息
    stats = dataset.get_statistics()
    print(f"\n数据集统计:")
    print(f"  图像数量: {stats['num_images']}")
    print(f"  物体总数: {stats['num_objects']}")
    print(f"  平均每张图像物体数: {stats['objects_per_image']:.2f}")
    
    # 显示类别分布
    print(f"\n类别分布:")
    for cls, count in stats['class_distribution'].items():
        if count > 0:
            print(f"  {cls}: {count}")
    
    # 测试数据加载
    print(f"\n测试数据加载...")
    image, boxes, labels = dataset[0]
    print(f"第一张图像:")
    print(f"  图像形状: {image.shape}")
    print(f"  边界框数量: {len(boxes)}")
    print(f"  边界框: {boxes}")
    print(f"  标签: {labels}")
    
    # 测试批量加载
    print(f"\n测试批量加载...")
    loader = VOCDatasetLoader(dataset, batch_size=4)
    for images, boxes_list, labels_list in loader:
        print(f"批次:")
        print(f"  图像形状: {images.shape}")
        print(f"  边界框列表长度: {len(boxes_list)}")
        print(f"  标签列表长度: {len(labels_list)}")
        break
    
    print("\n测试完成!")