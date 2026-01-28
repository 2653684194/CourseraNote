"""
Mask Dataset Loader
支持YOLO格式的目标检测数据集
"""
import numpy as np
import os
from PIL import Image
from typing import List, Tuple, Dict
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CNN_v4_cupy import xp, to_cpu


class MaskDataset:
    """
    Mask Dataset 加载器
    支持YOLO格式标注 (class_id, x_center, y_center, width, height)
    """
    
    def __init__(self, data_dir: str = './mask_dataset', image_size: int = 448):
        """
        Args:
            data_dir: 数据集根目录
            image_size: 图像尺寸
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # 类别定义 (根据你的mask数据集)
        self.classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        self.num_classes = len(self.classes)
        
        # 数据存储
        self.images = []
        self.labels = []
        self.image_paths = []
        
    def load_dataset(self, split: str = 'train'):
        """
        加载数据集
        
        Args:
            split: 'train' 或 'val'
        """
        images_dir = os.path.join(self.data_dir, 'images', split)
        labels_dir = os.path.join(self.data_dir, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"Error: Images directory not found: {images_dir}")
            return False
        
        if not os.path.exists(labels_dir):
            print(f"Error: Labels directory not found: {labels_dir}")
            return False
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_files.sort()
        
        print(f"Loading {split} dataset...")
        print(f"Found {len(image_files)} images")
        
        loaded = 0
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # 检查标注文件是否存在
            if not os.path.exists(label_path):
                continue
            
            try:
                # 加载图像
                image = Image.open(img_path).convert('RGB')
                orig_w, orig_h = image.size
                
                # Resize到目标尺寸
                image_resized = image.resize((self.image_size, self.image_size))
                image_array = np.array(image_resized).astype(np.float64) / 255.0
                
                # 归一化到[-1, 1]
                image_array = (image_array - 0.5) * 2.0
                
                # 转换为 (C, H, W)
                image_array = np.transpose(image_array, (2, 0, 1))
                
                # 加载标注
                boxes = []
                labels = []
                
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为绝对坐标 (基于原始图像尺寸)
                        x = x_center * orig_w
                        y = y_center * orig_h
                        w = width * orig_w
                        h = height * orig_h
                        
                        boxes.append([x, y, w, h])
                        labels.append(class_id)
                
                if len(boxes) > 0:
                    self.images.append(image_array)
                    self.labels.append({'boxes': np.array(boxes), 'labels': np.array(labels)})
                    self.image_paths.append(img_path)
                    loaded += 1
                    
                    if loaded % 100 == 0:
                        print(f"  Loaded {loaded}/{len(image_files)} images")
                        
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
        
        print(f"Successfully loaded {loaded} images with annotations")
        return True
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def get_statistics(self):
        """获取数据集统计信息"""
        num_images = len(self.images)
        num_objects = sum(len(label['labels']) for label in self.labels)
        
        class_counts = {}
        for label in self.labels:
            for class_id in label['labels']:
                class_name = self.classes[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'num_images': num_images,
            'num_objects': num_objects,
            'class_counts': class_counts
        }


class MaskDatasetLoader:
    """
    Mask Dataset 批量加载器
    """
    
    def __init__(self, dataset: MaskDataset, batch_size: int = 8, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.batch_idx = 0
        return self
    
    def __next__(self):
        if self.batch_idx >= len(self):
            raise StopIteration
        
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_idx:end_idx]
        
        images = []
        boxes_list = []
        labels_list = []
        
        for idx in batch_indices:
            image, label = self.dataset[idx]
            images.append(image)
            boxes_list.append(label['boxes'])
            labels_list.append(label['labels'])
        
        # 堆叠图像
        images = np.stack(images, axis=0)
        
        self.batch_idx += 1
        return images, boxes_list, labels_list


if __name__ == "__main__":
    # 测试数据加载
    print("Testing Mask Dataset Loader...")
    
    dataset = MaskDataset(data_dir='./mask_dataset', image_size=448)
    
    # 加载训练集
    dataset.load_dataset(split='train')
    
    # 显示统计信息
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Images: {stats['num_images']}")
    print(f"  Objects: {stats['num_objects']}")
    print(f"  Class distribution: {stats['class_counts']}")
    
    # 测试数据加载器
    loader = MaskDatasetLoader(dataset, batch_size=4, shuffle=True)
    print(f"\nNumber of batches: {len(loader)}")
    
    # 加载一个batch
    for images, boxes_list, labels_list in loader:
        print(f"\nBatch shape: {images.shape}")
        print(f"Number of images in batch: {len(boxes_list)}")
        for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            print(f"  Image {i}: {len(boxes)} objects")
        break
    
    print("\nTest completed!")
