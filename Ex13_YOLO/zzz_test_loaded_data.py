import numpy as np
from LoadData import laod_data, test_path, S, IMG_SIZE

# 加载数据
X, Y = laod_data(test_path, S=S, imgsize=IMG_SIZE, B=2, C=3)

print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')
print(f'X dtype: {X.dtype}')
print(f'Y dtype: {Y.dtype}')
print(f'X min/max: {X.min():.4f}/{X.max():.4f}')
print(f'Y non-zero count: {np.count_nonzero(Y)}')

# 取出第一个完整的 y
image_index = 0
y = Y[image_index]
print(f'\n取出第 {image_index+1} 个图像的完整 y:')
print(f'y shape: {y.shape}')

# 校验 y 的结构
S_w, S_h = S
B = 2
C = 3

print(f'\n校验 y 的结构:')
print(f'网格数量: S_w={S_w}, S_h={S_h}')
print(f'锚框数量: B={B}')
print(f'类别数量: C={C}')
print(f'每个网格的维度: B*5+C={B*5+C}')

# 检查是否有有效的目标标记
print(f'\n检查目标标记:')
object_count = 0
for grid_x in range(S_w):
    for grid_y in range(S_h):
        # 检查是否有任何锚框被使用
        for box_idx in range(B):
            if y[grid_x, grid_y, box_idx * 5 + 4] == 1.0:
                object_count += 1
                x_center = y[grid_x, grid_y, box_idx * 5 + 0]
                y_center = y[grid_x, grid_y, box_idx * 5 + 1]
                width = y[grid_x, grid_y, box_idx * 5 + 2]
                height = y[grid_x, grid_y, box_idx * 5 + 3]
                
                # 检查类别
                class_id = np.argmax(y[grid_x, grid_y, B*5:])
                class_prob = y[grid_x, grid_y, B*5 + class_id]
                
                print(f'  网格 ({grid_x}, {grid_y}), 锚框 {box_idx}:')
                print(f'    中心: ({x_center}, {y_center})')
                print(f'    宽高: ({width}, {height})')
                print(f'    类别: {class_id}, 概率: {class_prob}')

print(f'\n总共找到 {object_count} 个目标')

# 校验数据范围
print(f'\n校验数据范围:')
print(f'x_center 范围: {y[..., 0].min():.4f} ~ {y[..., 0].max():.4f}')
print(f'y_center 范围: {y[..., 1].min():.4f} ~ {y[..., 1].max():.4f}')
print(f'width 范围: {y[..., 2].min():.4f} ~ {y[..., 2].max():.4f}')
print(f'height 范围: {y[..., 3].min():.4f} ~ {y[..., 3].max():.4f}')
print(f'置信度 范围: {y[..., 4].min():.4f} ~ {y[..., 4].max():.4f}')
print(f'类别概率 范围: {y[..., B*5:].min():.4f} ~ {y[..., B*5:].max():.4f}')

# 校验网格分配
print(f'\n校验网格分配:')
for grid_x in range(S_w):
    for grid_y in range(S_h):
        for box_idx in range(B):
            if y[grid_x, grid_y, box_idx * 5 + 4] == 1.0:
                x_center = y[grid_x, grid_y, box_idx * 5 + 0]
                y_center = y[grid_x, grid_y, box_idx * 5 + 1]
                
                # 计算应该属于的网格
                expected_grid_x = int(x_center * S_w)
                expected_grid_y = int(y_center * S_h)
                expected_grid_x = min(expected_grid_x, S_w - 1)
                expected_grid_y = min(expected_grid_y, S_h - 1)
                
                if grid_x == expected_grid_x and grid_y == expected_grid_y:
                    print(f'  网格 ({grid_x}, {grid_y}) 分配正确')
                else:
                    print(f'  网格 ({grid_x}, {grid_y}) 分配错误，应该是 ({expected_grid_x}, {expected_grid_y})')