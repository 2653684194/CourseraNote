import numpy as np

path = 'models/yolov1_mask_dectection.npz'


# 方法1：尝试用 numpy.load 但不要求 pickle
try:
    model = np.load(path, allow_pickle=False)
except Exception as e:
    print(f"方法1失败: {e}")

# 方法2：尝试用普通文件读取
try:
    with open(path, 'rb') as f:
        data = f.read()
    print(f"成功读取文件，大小: {len(data)} 字节")
except Exception as e:
    print(f"方法2失败: {e}")

# 方法3：尝试用 pickle 加载
import pickle
try:
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("成功用 pickle 加载")
except Exception as e:
    print(f"方法3失败: {e}")
