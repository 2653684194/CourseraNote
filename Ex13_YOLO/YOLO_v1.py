"""
YOLO v1 Implementation based on the paper:
"You Only Look Once: Unified, Real-Time Object Detection"

Architecture:
- 24 Convolutional layers for feature extraction
- 2 Fully connected layers for prediction
- Input: 448x448x3
- Output: 7x7x30 (S x S x (B*5 + C))
  - S = 7 (grid size)
  - B = 2 (bounding boxes per grid cell)
  - C = 20 (number of classes for PASCAL VOC)
  - 30 = B*5 + C = 2*5 + 20 = 10 + 20

Based on CNN_v4_cupy.py framework
"""

import numpy as np
import sys
import os

# Add the current directory to path to import CNN_v4_cupy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CNN_v4_cupy import (
    xp, to_cpu, to_gpu, Conv, BatchNorm, Activation, 
    Pooling, FC, layer, TrainableLayer, CNN
)


class YOLOLoss:
    """
    YOLO Loss Function
    
    Loss = λ_coord * Σ[1_obj * ((x-x̂)² + (y-ŷ)² + (√w-√ŵ)² + (√h-√ĥ)²)]
         + λ_noobj * Σ[1_noobj * (C-Ĉ)²]
         + Σ[1_obj * (C-Ĉ)²]
         + Σ[1_obj * (p(c)-p̂(c))²]
    
    Where:
    - λ_coord = 5 (coordinate loss weight)
    - λ_noobj = 0.5 (no-object confidence loss weight)
    - 1_obj = 1 if object exists in cell, 0 otherwise
    - 1_noobj = 1 if no object exists in cell, 0 otherwise
    """
    
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per cell
        self.C = C  # Number of classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def compute_loss(self, predictions, targets):
        """
        Compute YOLO loss
        
        Args:
            predictions: (N, S, S, B*5 + C) - raw network output
            targets: (N, S, S, B*5 + C) - ground truth
            
        Returns:
            loss: scalar loss value
            loss_dict: dictionary with individual loss components
        """
        N = predictions.shape[0]
        
        # Reshape predictions and targets
        pred = predictions.reshape(N, self.S, self.S, self.B * 5 + self.C)
        targ = targets.reshape(N, self.S, self.S, self.B * 5 + self.C)
        
        # Extract components from predictions
        # For each cell: [x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2, class_probs...]
        pred_boxes = []
        for b in range(self.B):
            start = b * 5
            pred_boxes.append({
                'x': pred[..., start + 0],
                'y': pred[..., start + 1],
                'w': pred[..., start + 2],
                'h': pred[..., start + 3],
                'conf': pred[..., start + 4]
            })
        pred_classes = pred[..., self.B * 5:]  # (N, S, S, C)
        
        # Extract components from targets
        targ_boxes = []
        for b in range(self.B):
            start = b * 5
            targ_boxes.append({
                'x': targ[..., start + 0],
                'y': targ[..., start + 1],
                'w': targ[..., start + 2],
                'h': targ[..., start + 3],
                'conf': targ[..., start + 4]
            })
        targ_classes = targ[..., self.B * 5:]  # (N, S, S, C)
        
        # Object mask: 1 if object exists in cell, 0 otherwise
        # We use the first box's confidence as indicator
        obj_mask = targ_boxes[0]['conf']  # (N, S, S)
        noobj_mask = 1 - obj_mask  # (N, S, S)
        
        # Expand masks for broadcasting
        obj_mask_expanded = obj_mask[..., None]  # (N, S, S, 1)
        noobj_mask_expanded = noobj_mask[..., None]  # (N, S, S, 1)
        
        # Coordinate loss (only for cells with objects)
        coord_loss = 0
        for b in range(self.B):
            # x, y loss
            coord_loss += xp.sum(obj_mask * ((pred_boxes[b]['x'] - targ_boxes[b]['x']) ** 2))
            coord_loss += xp.sum(obj_mask * ((pred_boxes[b]['y'] - targ_boxes[b]['y']) ** 2))
            
            # w, h loss (square root to penalize small errors in small boxes less)
            coord_loss += xp.sum(obj_mask * ((xp.sqrt(xp.abs(pred_boxes[b]['w']) + 1e-8) - 
                                               xp.sqrt(xp.abs(targ_boxes[b]['w']) + 1e-8)) ** 2))
            coord_loss += xp.sum(obj_mask * ((xp.sqrt(xp.abs(pred_boxes[b]['h']) + 1e-8) - 
                                               xp.sqrt(xp.abs(targ_boxes[b]['h']) + 1e-8)) ** 2))
        
        coord_loss = self.lambda_coord * coord_loss / N
        
        # Confidence loss
        conf_loss_obj = 0
        conf_loss_noobj = 0
        for b in range(self.B):
            # Confidence loss for cells with objects
            conf_loss_obj += xp.sum(obj_mask * ((pred_boxes[b]['conf'] - targ_boxes[b]['conf']) ** 2))
            # Confidence loss for cells without objects
            conf_loss_noobj += xp.sum(noobj_mask * ((pred_boxes[b]['conf'] - targ_boxes[b]['conf']) ** 2))
        
        conf_loss_obj = conf_loss_obj / N
        conf_loss_noobj = self.lambda_noobj * conf_loss_noobj / N
        
        # Classification loss (only for cells with objects)
        class_loss = xp.sum(obj_mask_expanded * ((pred_classes - targ_classes) ** 2)) / N
        
        # Total loss
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        
        loss_dict = {
            'total': float(to_cpu(total_loss)),
            'coord': float(to_cpu(coord_loss)),
            'conf_obj': float(to_cpu(conf_loss_obj)),
            'conf_noobj': float(to_cpu(conf_loss_noobj)),
            'class': float(to_cpu(class_loss))
        }
        
        return total_loss, loss_dict
    
    def compute_gradient(self, predictions, targets):
        """
        Compute gradient of YOLO loss w.r.t. predictions
        
        Args:
            predictions: (N, S, S, B*5 + C) - raw network output
            targets: (N, S, S, B*5 + C) - ground truth
            
        Returns:
            grad: (N, S, S, B*5 + C) gradient
        """
        N = predictions.shape[0]
        
        # Reshape
        pred = predictions.reshape(N, self.S, self.S, self.B * 5 + self.C)
        targ = targets.reshape(N, self.S, self.S, self.B * 5 + self.C)
        
        # Initialize gradient
        grad = xp.zeros_like(pred)
        
        # Object mask
        obj_mask = targ[..., 4]  # Use first box confidence as indicator
        noobj_mask = 1 - obj_mask
        
        # Compute gradients for each component
        for b in range(self.B):
            start = b * 5
            
            # x gradient
            grad[..., start + 0] = (2 * self.lambda_coord * obj_mask * 
                                    (pred[..., start + 0] - targ[..., start + 0])) / N
            
            # y gradient
            grad[..., start + 1] = (2 * self.lambda_coord * obj_mask * 
                                    (pred[..., start + 1] - targ[..., start + 1])) / N
            
            # w gradient (with sqrt)
            w_pred = pred[..., start + 2]
            w_targ = targ[..., start + 2]
            grad[..., start + 2] = (self.lambda_coord * obj_mask * 
                                    (xp.sqrt(xp.abs(w_pred) + 1e-8) - xp.sqrt(xp.abs(w_targ) + 1e-8)) * 
                                    xp.sign(w_pred) / xp.sqrt(xp.abs(w_pred) + 1e-8)) / N
            
            # h gradient (with sqrt)
            h_pred = pred[..., start + 3]
            h_targ = targ[..., start + 3]
            grad[..., start + 3] = (self.lambda_coord * obj_mask * 
                                    (xp.sqrt(xp.abs(h_pred) + 1e-8) - xp.sqrt(xp.abs(h_targ) + 1e-8)) * 
                                    xp.sign(h_pred) / xp.sqrt(xp.abs(h_pred) + 1e-8)) / N
            
            # Confidence gradient
            grad[..., start + 4] = (2 * obj_mask * (pred[..., start + 4] - targ[..., start + 4]) + 
                                    2 * self.lambda_noobj * noobj_mask * (pred[..., start + 4] - targ[..., start + 4])) / N
        
        # Class probabilities gradient
        class_start = self.B * 5
        grad[..., class_start:] = (2 * obj_mask[..., None] * 
                                   (pred[..., class_start:] - targ[..., class_start:])) / N
        
        return grad.reshape(N, -1)


class YOLOOutput(layer):
    """
    YOLO Output Layer
    
    Reshapes the FC output to YOLO format:
    Input: (N, S*S*(B*5+C))
    Output: (N, S, S, B*5+C)
    """
    
    def __init__(self, S=7, B=2, C=20):
        self.S = S
        self.B = B
        self.C = C
        self.output_dim = S * S * (B * 5 + C)
        
    def forward_prop(self, X):
        """
        X: (N, S*S*(B*5+C))
        Returns: (N, S, S, B*5+C)
        """
        N = X.shape[0]
        return X.reshape(N, self.S, self.S, self.B * 5 + self.C)
    
    def backward_prop(self, dY):
        """
        dY: (N, S, S, B*5+C)
        Returns: (N, S*S*(B*5+C))
        """
        N = dY.shape[0]
        return dY.reshape(N, -1)
    
    def get_config(self):
        return {
            'type': 'YOLOOutput',
            'S': self.S,
            'B': self.B,
            'C': self.C
        }
    
    def set_config(self, config):
        self.S = config['S']
        self.B = config['B']
        self.C = config['C']


def create_yolo_v1_network(S=7, B=2, C=20, input_channels=3, learning_rate=0.001, _Adam=True):
    """
    Create YOLO v1 Network Architecture
    
    Based on the paper:
    - 24 Convolutional layers
    - 2 Fully connected layers
    - Input: 448x448x3
    - Output: 7x7x30 (for PASCAL VOC: S=7, B=2, C=20)
    
    Architecture details:
    Layer 1: Conv 7x7x64, stride=2, pad=3 -> MaxPool 2x2, stride=2
    Layer 2: Conv 3x3x192, stride=1, pad=1 -> MaxPool 2x2, stride=2
    Layer 3: Conv 1x1x128 -> Conv 3x3x256 -> Conv 1x1x256 -> Conv 3x3x512 -> MaxPool 2x2, stride=2
    Layer 4: [Conv 1x1x256 -> Conv 3x3x512] x 4 -> Conv 1x1x512 -> Conv 3x3x1024 -> MaxPool 2x2, stride=2
    Layer 5: [Conv 1x1x512 -> Conv 3x3x1024] x 2 -> Conv 3x3x1024 -> Conv 3x3x1024, stride=2
    Layer 6: Conv 3x3x1024, stride=1 -> Conv 3x3x1024, stride=1
    FC: 4096 -> 4096 -> S*S*(B*5+C)
    """
    
    layers = []
    
    # Layer 1: Conv 7x7x64, stride=2 -> MaxPool 2x2
    layers.append(Conv(filter_num=64, filter_size=7, filter_channel=input_channels, 
                       stride=2, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 2: Conv 3x3x192, stride=1 -> MaxPool 2x2
    layers.append(Conv(filter_num=192, filter_size=3, filter_channel=64, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 3: Conv 1x1x128 -> Conv 3x3x256 -> Conv 1x1x256 -> Conv 3x3x512 -> MaxPool
    layers.append(Conv(filter_num=128, filter_size=1, filter_channel=192, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=256, filter_size=3, filter_channel=128, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=256, filter_size=1, filter_channel=256, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=512, filter_size=3, filter_channel=256, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 4: [Conv 1x1x256 -> Conv 3x3x512] x 4
    for _ in range(4):
        layers.append(Conv(filter_num=256, filter_size=1, filter_channel=512, 
                           stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
        layers.append(Activation('relu'))
        layers.append(Conv(filter_num=512, filter_size=3, filter_channel=256, 
                           stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
        layers.append(Activation('relu'))
    
    # Continue Layer 4: Conv 1x1x512 -> Conv 3x3x1024 -> MaxPool
    layers.append(Conv(filter_num=512, filter_size=1, filter_channel=512, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=1024, filter_size=3, filter_channel=512, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 5: [Conv 1x1x512 -> Conv 3x3x1024] x 2
    for _ in range(2):
        layers.append(Conv(filter_num=512, filter_size=1, filter_channel=1024, 
                           stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
        layers.append(Activation('relu'))
        layers.append(Conv(filter_num=1024, filter_size=3, filter_channel=512, 
                           stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
        layers.append(Activation('relu'))
    
    # Continue Layer 5: Conv 3x3x1024 -> Conv 3x3x1024, stride=2
    layers.append(Conv(filter_num=1024, filter_size=3, filter_channel=1024, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=1024, filter_size=3, filter_channel=1024, 
                       stride=2, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    
    # Layer 6: Conv 3x3x1024 x 2
    layers.append(Conv(filter_num=1024, filter_size=3, filter_channel=1024, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=1024, filter_size=3, filter_channel=1024, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    
    # FC Layers
    # Flatten and connect to 4096
    layers.append(FC(output_size=4096, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    # layers.append(Dropout(drop_rate=0.5))  # Optional dropout
    
    # Second FC layer
    layers.append(FC(output_size=4096, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    # layers.append(Dropout(drop_rate=0.5))  # Optional dropout
    
    # Final output layer: S*S*(B*5+C)
    output_dim = S * S * (B * 5 + C)
    layers.append(FC(output_size=output_dim, learning_rate=learning_rate, _Adam=_Adam))
    
    # Reshape to YOLO format
    layers.append(YOLOOutput(S=S, B=B, C=C))
    
    return CNN(layers=layers, learning_rate=learning_rate, _Adam=_Adam)


def create_tiny_yolo_v1(S=7, B=2, C=20, input_channels=3, learning_rate=0.001, _Adam=True):
    """
    Create Tiny YOLO v1 (Fast YOLO) - 9 convolutional layers instead of 24
    Faster but less accurate
    """
    layers = []
    
    # Layer 1: Conv 7x7x64, stride=2 -> MaxPool
    layers.append(Conv(filter_num=64, filter_size=7, filter_channel=input_channels, 
                       stride=2, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 2: Conv 3x3x192 -> MaxPool
    layers.append(Conv(filter_num=192, filter_size=3, filter_channel=64, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 3: Conv 1x1x128 -> Conv 3x3x256 -> MaxPool
    layers.append(Conv(filter_num=128, filter_size=1, filter_channel=192, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=256, filter_size=3, filter_channel=128, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 4: Conv 1x1x256 -> Conv 3x3x512 -> MaxPool
    layers.append(Conv(filter_num=256, filter_size=1, filter_channel=256, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=512, filter_size=3, filter_channel=256, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # Layer 5: Conv 1x1x512 -> Conv 3x3x1024 -> MaxPool
    layers.append(Conv(filter_num=512, filter_size=1, filter_channel=512, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Conv(filter_num=1024, filter_size=3, filter_channel=512, 
                       stride=1, same_padding=True, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    layers.append(Pooling(pool_size=2, stride=2, pool_type='max'))
    
    # FC Layers
    layers.append(FC(output_size=256, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    
    layers.append(FC(output_size=4096, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(Activation('relu'))
    
    # Output
    output_dim = S * S * (B * 5 + C)
    layers.append(FC(output_size=output_dim, learning_rate=learning_rate, _Adam=_Adam))
    layers.append(YOLOOutput(S=S, B=B, C=C))
    
    return CNN(layers=layers, learning_rate=learning_rate, _Adam=_Adam)


class YOLOPostProcessor:
    """
    Post-processing for YOLO predictions
    Convert network output to bounding boxes
    """
    
    def __init__(self, S=7, B=2, C=20, image_size=448, conf_threshold=0.2, nms_threshold=0.4):
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
    def decode_predictions(self, predictions):
        """
        Decode YOLO predictions to bounding boxes
        
        Args:
            predictions: (N, S, S, B*5+C) or (N, S*S*(B*5+C))
            
        Returns:
            boxes: list of lists, each containing [x, y, w, h, confidence, class_id, class_prob]
                   for each image in batch
        """
        N = predictions.shape[0]
        
        # Reshape if needed
        if len(predictions.shape) == 2:
            predictions = predictions.reshape(N, self.S, self.S, self.B * 5 + self.C)
        
        batch_boxes = []
        
        # Move predictions to CPU for post-processing
        predictions_cpu = to_cpu(predictions)
        
        for n in range(N):
            boxes = []
            pred = predictions_cpu[n]  # (S, S, B*5+C)
            
            for i in range(self.S):
                for j in range(self.S):
                    # Get class probabilities
                    class_probs = pred[i, j, self.B * 5:]  # (C,)
                    class_id = int(np.argmax(class_probs))
                    class_prob = float(class_probs[class_id])
                    
                    # Get bounding boxes for this cell
                    for b in range(self.B):
                        start = b * 5
                        x = pred[i, j, start + 0]
                        y = pred[i, j, start + 1]
                        w = pred[i, j, start + 2]
                        h = pred[i, j, start + 3]
                        conf = pred[i, j, start + 4]
                        
                        # Convert from grid coordinates to image coordinates
                        # x, y are relative to cell, add cell offset
                        x_img = (j + float(x)) / self.S * self.image_size
                        y_img = (i + float(y)) / self.S * self.image_size
                        w_img = float(w) * self.image_size
                        h_img = float(h) * self.image_size
                        
                        # Final confidence = box confidence * class probability
                        final_conf = float(conf) * class_prob
                        
                        if final_conf > self.conf_threshold:
                            boxes.append([x_img, y_img, w_img, h_img, final_conf, class_id, class_prob])
            
            batch_boxes.append(boxes)
        
        return batch_boxes
    
    def iou(self, box1, box2):
        """
        Compute IoU between two boxes
        Boxes are in format [x, y, w, h, ...]
        """
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        
        # Convert to corner format
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # Compute intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    def nms(self, boxes):
        """
        Apply Non-Maximum Suppression
        
        Args:
            boxes: list of [x, y, w, h, confidence, class_id, class_prob]
            
        Returns:
            filtered_boxes: list after NMS
        """
        if not boxes:
            return []
        
        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        keep = []
        while boxes:
            best = boxes[0]
            keep.append(best)
            boxes = [box for box in boxes[1:] if self.iou(best, box) < self.nms_threshold]
        
        return keep
    
    def process(self, predictions):
        """
        Full post-processing pipeline
        
        Args:
            predictions: network output
            
        Returns:
            final_boxes: list of filtered bounding boxes for each image
        """
        # Decode predictions
        batch_boxes = self.decode_predictions(predictions)
        
        # Apply NMS for each image
        final_boxes = []
        for boxes in batch_boxes:
            final_boxes.append(self.nms(boxes))
        
        return final_boxes


class YOLOTargetEncoder:
    """
    Encode ground truth bounding boxes to YOLO target format
    """
    
    def __init__(self, S=7, B=2, C=20, image_size=448):
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size
        
    def encode(self, boxes_list, labels_list):
        """
        Encode ground truth to YOLO format
        
        Args:
            boxes_list: list of N arrays, each (num_objects, 4) with [x, y, w, h] in image coordinates
            labels_list: list of N arrays, each (num_objects,) with class ids
            
        Returns:
            targets: (N, S, S, B*5+C) encoded targets
        """
        N = len(boxes_list)
        targets = xp.zeros((N, self.S, self.S, self.B * 5 + self.C))
        
        for n in range(N):
            boxes = boxes_list[n]  # (num_objects, 4)
            labels = labels_list[n]  # (num_objects,)
            
            if len(boxes) == 0:
                continue
            
            for obj_idx in range(len(boxes)):
                x, y, w, h = boxes[obj_idx]
                label = int(labels[obj_idx])
                
                # Normalize to [0, 1]
                x_norm = x / self.image_size
                y_norm = y / self.image_size
                w_norm = w / self.image_size
                h_norm = h / self.image_size
                
                # Find responsible grid cell
                grid_x = int(x_norm * self.S)
                grid_y = int(y_norm * self.S)
                grid_x = min(grid_x, self.S - 1)
                grid_y = min(grid_y, self.S - 1)
                
                # Relative position within cell
                x_cell = x_norm * self.S - grid_x
                y_cell = y_norm * self.S - grid_y
                
                # Find which bounding box to use (first available)
                for b in range(self.B):
                    start = b * 5
                    if targets[n, grid_y, grid_x, start + 4] == 0:  # If confidence is 0
                        targets[n, grid_y, grid_x, start + 0] = x_cell
                        targets[n, grid_y, grid_x, start + 1] = y_cell
                        targets[n, grid_y, grid_x, start + 2] = w_norm
                        targets[n, grid_y, grid_x, start + 3] = h_norm
                        targets[n, grid_y, grid_x, start + 4] = 1.0  # Confidence
                        break
                
                # Set class probability (only once per cell)
                targets[n, grid_y, grid_x, self.B * 5 + label] = 1.0
        
        return targets


class YOLOTrainer:
    """
    YOLO Training wrapper
    """
    
    def __init__(self, model, loss_fn, target_encoder, S=7, B=2, C=20):
        self.model = model
        self.loss_fn = loss_fn
        self.target_encoder = target_encoder
        self.S = S
        self.B = B
        self.C = C
        self.cost_history = []
        
    def train_step(self, images, boxes_list, labels_list):
        """
        Single training step
        
        Args:
            images: (N, C, H, W) input images
            boxes_list: list of ground truth boxes
            labels_list: list of ground truth labels
            
        Returns:
            loss: total loss value
            loss_dict: dictionary of loss components
        """
        # Encode targets
        targets = self.target_encoder.encode(boxes_list, labels_list)
        
        # Forward pass
        predictions = self.model.forward(images, training=True)
        
        # Compute loss
        loss, loss_dict = self.loss_fn.compute_loss(predictions, targets)
        
        # Compute gradient
        grad = self.loss_fn.compute_gradient(predictions, targets)
        
        # Backward pass
        self.model.backward(grad)
        
        return float(to_cpu(loss)), loss_dict
    
    def train(self, X, boxes_list, labels_list, epochs=100, batch_size=16, 
              print_cost=True, save_path=None):
        """
        Train YOLO model
        
        Args:
            X: (N, C, H, W) training images
            boxes_list: list of ground truth boxes for each image
            labels_list: list of ground truth labels for each image
            epochs: number of training epochs
            batch_size: batch size
            print_cost: whether to print cost
            save_path: path to save model
        """
        X = to_gpu(X)
        N = X.shape[0]
        num_batches = (N + batch_size - 1) // batch_size
        
        print(f"Training YOLO with {N} samples, batch_size={batch_size}, num_batches={num_batches}")
        
        try:
            for epoch in range(epochs):
                # Shuffle data
                indices = xp.random.permutation(N)
                X_shuffled = X[indices]
                boxes_shuffled = [boxes_list[i] for i in indices]
                labels_shuffled = [labels_list[i] for i in indices]
                
                epoch_loss = 0
                epoch_loss_dict = {'coord': 0, 'conf_obj': 0, 'conf_noobj': 0, 'class': 0}
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, N)
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    boxes_batch = boxes_shuffled[start_idx:end_idx]
                    labels_batch = labels_shuffled[start_idx:end_idx]
                    
                    # Training step
                    loss, loss_dict = self.train_step(X_batch, boxes_batch, labels_batch)
                    
                    epoch_loss += loss
                    for key in epoch_loss_dict:
                        epoch_loss_dict[key] += loss_dict[key]
                    
                    if print_cost and batch_idx % max(1, num_batches // 5) == 0:
                        print(f"Epoch {epoch} Batch {batch_idx}/{num_batches} Loss: {loss:.4f}")
                
                # Average losses
                epoch_loss /= num_batches
                for key in epoch_loss_dict:
                    epoch_loss_dict[key] /= num_batches
                
                self.cost_history.append(epoch_loss)
                
                if print_cost:
                    print(f"Epoch {epoch}: Loss={epoch_loss:.4f} "
                          f"(coord={epoch_loss_dict['coord']:.4f}, "
                          f"conf_obj={epoch_loss_dict['conf_obj']:.4f}, "
                          f"conf_noobj={epoch_loss_dict['conf_noobj']:.4f}, "
                          f"class={epoch_loss_dict['class']:.4f})")
                
                # Save model
                if save_path:
                    self.model.save_model(save_path)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            if save_path:
                self.model.save_model(save_path)
        
        return self.cost_history


# PASCAL VOC class names
PASCAL_VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


if __name__ == "__main__":
    # Test YOLO implementation
    print("Testing YOLO v1 Implementation...")
    
    # Create model
    S, B, C = 7, 2, 20
    model = create_yolo_v1_network(S=S, B=B, C=C, learning_rate=0.001, _Adam=True)
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_input = xp.random.randn(2, 3, 448, 448).astype(xp.float32)
    output = model.forward(test_input, training=False)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (2, {S}, {S}, {B*5+C})")
    
    # Test loss function
    print("\nTesting loss function...")
    loss_fn = YOLOLoss(S=S, B=B, C=C)
    predictions = xp.random.randn(2, S, S, B*5+C)
    targets = xp.zeros((2, S, S, B*5+C))
    # Add some dummy targets
    targets[0, 3, 3, 4] = 1  # Confidence for first box
    targets[0, 3, 3, 14] = 1  # Class 0
    
    loss, loss_dict = loss_fn.compute_loss(predictions, targets)
    print(f"Loss: {loss:.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test target encoder
    print("\nTesting target encoder...")
    encoder = YOLOTargetEncoder(S=S, B=B, C=C)
    boxes_list = [
        xp.array([[224, 224, 100, 100]]),  # Center of image
        xp.array([])  # No objects
    ]
    labels_list = [
        xp.array([0]),  # Class 0
        xp.array([])
    ]
    encoded_targets = encoder.encode(boxes_list, labels_list)
    print(f"Encoded targets shape: {encoded_targets.shape}")
    
    # Test post-processor
    print("\nTesting post-processor...")
    post_processor = YOLOPostProcessor(S=S, B=B, C=C)
    decoded_boxes = post_processor.decode_predictions(predictions)
    print(f"Decoded boxes for first image: {len(decoded_boxes[0])} boxes")
    
    print("\nAll tests passed!")
