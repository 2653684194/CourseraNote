import numpy as np
import matplotlib.pyplot as plt
import os

# root path
ROOT_PATH = os.path.dirname(__file__)


train_path = (os.path.join(ROOT_PATH, "mask_dataset/images/train"), os.path.join(ROOT_PATH, "mask_dataset/labels/train"))
test_path = (os.path.join(ROOT_PATH, "mask_dataset/images/test"), os.path.join(ROOT_PATH, "mask_dataset/labels/test"))

# Grids number (S_w, S_h)
S = (10, 10)

# previous img size
PRE_IMG_SIZE = (1080, 1920)

# img size
IMG_SIZE = (448, 448)

# read size of folder
def read_folder_size(path:str) -> int:
    return len(os.listdir(path))
Train_size = read_folder_size(train_path[0]) # 398
Test_size = read_folder_size(test_path[0]) # 102

# print(Train_size,Test_size) # 398,102

# 每个lable x,y表示预测框的中心


def load_data(
    path:tuple[str, str],
    S:tuple[int, int], # grid number
    imgsize:tuple[int, int],
    B:int = 2, # box number
    C:int = 1, # class number
    shift:bool=True
) -> tuple[np.ndarray, np.ndarray]:
    images_path, labels_path = path

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]) # list of imagesfile name str

    # 打印结果
    print("排序后的图像文件列表:")
    # for i, file in enumerate(image_files):
    #     print(f"{i+1}. {file}")
    print(f"总共有 {len(image_files)} 个图像文件")

    num_images = len(image_files)

    S_w, S_h = S
    X = np.zeros((num_images, imgsize[0], imgsize[1], 3), dtype=np.float32)
    Y = np.zeros((num_images, S_w, S_h, B * 5 + C), dtype=np.float32)

    

    from PIL import Image
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_path, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        # 从右侧开始分割字符串，按照'.'分割，之分割一次，得到["img_000001", "jpg"]，然后拼接成"img_000001.txt"
        label_path = os.path.join(labels_path, label_file)
        
        img_pil = Image.open(img_path).convert('RGB')
        img_resized = img_pil.resize((imgsize[1], imgsize[0]))
        X[i] = np.array(img_resized, dtype=np.float32) / 255.0
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f: # 语句块之外自动关闭文件
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"跳过无效行: {line.strip()}")
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # # pre calculate the grid size
                # grid_w = 1 / S[0]
                # grid_h = 1 / S[1]
                # grid_x = int(x_center / grid_w) 
                # grid_y = int(y_center / grid_h)
                # 简化计算同时避免溢出导致出错
                grid_x = int(x_center * S[0])
                grid_y = int(y_center * S[1])

                box_idx = 0
                while Y[i, grid_x, grid_y, box_idx * 5 + 4] == 1.0:
                    box_idx += 1
                if box_idx >= B:
                    print(f"跳过第 {i+1} 张图像的Grid({grid_x},{grid_y})的第 {box_idx+1} 个目标框: 已超出最大框数 {B}")
                    continue
                Y[i, grid_x, grid_y, box_idx * 5 + 0] = x_center
                Y[i, grid_x, grid_y, box_idx * 5 + 1] = y_center
                Y[i, grid_x, grid_y, box_idx * 5 + 2] = width
                Y[i, grid_x, grid_y, box_idx * 5 + 3] = height
                Y[i, grid_x, grid_y, box_idx * 5 + 4] = 1.0
                """
                这里必须做几点说明
                """

                Y[i, grid_x, grid_y, B*5 + class_id] = 1.0
            # 过滤掉空白标签样本
            if shift:
                BoxIndices = np.arange(B).astype(int)
                tmp = Y[..., BoxIndices*5+4].sum(axis=-1) # (S_w, S_h)
                tmp = tmp.reshape(Y.shape[0],-1).sum(axis=-1) >= 1

    return X[tmp], Y[tmp]