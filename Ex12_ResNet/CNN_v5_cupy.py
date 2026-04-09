import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import os

import site

# ================= CuPy DLL Fix for Windows =================
if os.name == 'nt':
    def _add_nvidia_dll_paths():
        try:
            # Find site-packages where nvidia libs are installed
            # We look for the 'nvidia' folder in site-packages
            site_packages = site.getsitepackages()
            nvidia_path = None
            for sp in site_packages:
                p = os.path.join(sp, 'nvidia')
                if os.path.isdir(p):
                    nvidia_path = p
                    break
            
            if nvidia_path:
                # List of nvidia modules that contain DLLs in 'bin'
                modules = ['cuda_runtime', 'cublas', 'cudnn', 'curand', 'cusolver', 'cusparse', 'cuda_nvrtc', 'nvjitlink']
                
                for mod in modules:
                    bin_path = os.path.join(nvidia_path, mod, 'bin')
                    if os.path.isdir(bin_path):
                        try:
                            os.add_dll_directory(bin_path)
                        except AttributeError:
                            pass # Python < 3.8
                        os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                        
                        # Set CUDA_PATH to silence CuPy warning if using pip-installed CUDA
                        if mod == 'cuda_runtime' and 'CUDA_PATH' not in os.environ:
                            os.environ['CUDA_PATH'] = os.path.dirname(bin_path)
        except Exception as e:
            print(f"Warning: Failed to add NVIDIA DLL paths: {e}")

    _add_nvidia_dll_paths()
# ===========================================================

# ================= CuPy Acceleration Setup =================
# Check if CuPy should be disabled via environment variable
if os.environ.get('DISABLE_CUPY') == '1':
    print("⚠️ [Backend] CuPy disabled by DISABLE_CUPY environment variable")
    print("⚠️ [Backend] Using NumPy (CPU)")
    from numpy.lib.stride_tricks import as_strided
    xp = np
    cp = None  # Explicitly set cp to None
else:
    try:
        import cupy as cp
        from cupy.lib.stride_tricks import as_strided
        
        # Check if CuPy and CUDA libraries (like curand) are working correctly
        # This catches "ImportError: DLL load failed" for curand/cublas
        cp.random.randn(1)
        
        xp = cp
        print("🚀 [Backend] Using CuPy (GPU Acceleration)")
    except (ImportError, OSError, Exception) as e:
        # Fallback to NumPy if CuPy is missing or broken (e.g. missing DLLs)
        print(f"⚠️ [Backend] CuPy initialization failed: {e}")
        print("⚠️ [Backend] Falling back to NumPy (CPU)")
        from numpy.lib.stride_tricks import as_strided
        xp = np
        cp = None


def to_gpu(array, dtype=cp.float32, copy=True):
    """
    安全的将数组转移到GPU
    确保返回纯净的CuPy数组
    """
    if array is None:
        return None
    
    # 如果已经是CuPy数组
    if isinstance(array, cp.ndarray):
        if copy:
            return array.copy()  # 创建副本避免共享内存
        return array
    
    # 如果是NumPy数组
    if isinstance(array, np.ndarray):
        # 确保使用正确的数据类型
        if dtype is not None:
            array = array.astype(dtype)
        # 使用cp.asarray创建新的CuPy数组
        return cp.asarray(array)
    
    # 如果是其他类型（如list），先转NumPy再转CuPy
    return cp.asarray(np.asarray(array), dtype=dtype)

# def to_cpu(array, dtype=np.float32, copy=True):
#     """
#     安全的将数组转移到CPU
#     确保返回纯净的NumPy数组，避免溢出和nan/inf
#     """
#     if array is None:
#         return None
#     # 如果是NumPy数组
#     if isinstance(array, np.ndarray):
#         arr = array.copy() if copy else array
#         # Clip防止溢出
#         if np.issubdtype(dtype, np.integer):
#             arr = np.clip(arr, np.iinfo(dtype).min, np.iinfo(dtype).max)
#         elif np.issubdtype(dtype, np.floating):
#             arr = np.clip(arr, -1e10, 1e10)
#         arr = arr.astype(dtype)
#         # nan/inf处理
#         arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
#         return arr
#     # 如果是CuPy数组
#     if isinstance(array, cp.ndarray):
#         cpu_array = array.get()
#         # Clip防止溢出
#         if np.issubdtype(dtype, np.integer):
#             cpu_array = np.clip(cpu_array, np.iinfo(dtype).min, np.iinfo(dtype).max)
#         elif np.issubdtype(dtype, np.floating):
#             cpu_array = np.clip(cpu_array, -1e10, 1e10)
#         cpu_array = cpu_array.astype(dtype)
#         # nan/inf处理
#         cpu_array = np.nan_to_num(cpu_array, nan=0.0, posinf=1e10, neginf=-1e10)
#         return cpu_array
#     # 其他类型
#     arr = np.asarray(array, dtype=dtype)
#     arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
#     return arr
# ===========================================================
# old version
def to_cpu(array, dtype=np.float32, copy=True):# float32 by default
    """
    安全的将数组转移到CPU
    确保返回纯净的NumPy数组
    """
    if array is None:
        return None
    
    # 如果是NumPy数组
    if isinstance(array, np.ndarray):
        if copy:
            return array.copy()
        return array
    
    # 如果是CuPy数组
    if isinstance(array, cp.ndarray):
        # 使用get()获取NumPy数组
        cpu_array = array.get()
        if dtype is not None:
            cpu_array = cpu_array.astype(dtype)
        return cpu_array
    
    # 如果是其他类型
    return np.asarray(array, dtype=dtype)
# # ===========================================================

def interrupt():
    path = 'AAA_Interrupt_assist.txt'
    if not os.path.exists(path):
        return 0
    with open(path, 'r') as f:
        line = f.readline().strip()
        if line.startswith("Interrupt:"):
            try:
                return int(line.split(":")[1].strip()) == 1
            except ValueError:
                return 0
    # path = 'interrupt.txt'
    # if os.path.exists(path):
    #     return 1


def img2col(X:xp.ndarray, filter_size:int, stride:int = 1, padding:tuple[int,int,int,int]=(0,0,0,0), dilation:int = 1)->xp.ndarray:
    '''
    X: (N,X_C,X_H,X_W)
    filter_size: (f_H,f_W) or f_size
    stride: f_s
    padding: (p_H1,p_H2,p_W1,p_W2) - asymmetric padding for height (top, bottom) and width (left, right)
    # return (N * h_out * w_out, X_C * filter_size * filter_size)
    return (N,h_out,w_out,C,filter_size,filter_size)
    '''
    N,C,H,W = X.shape # C == filter_c

#----------------------------强制复制到gpu确保数组不是混杂numpy，保证pad和asstride能正常使用-----------------------------------#
    X = cp.array(X, copy=True)  # 强制复制到GPU，确保不共享内存
    
    # Handle both old format (p_h, p_w) and new format (p_H1, p_H2, p_W1, p_W2)
    if len(padding) == 2:
        # Old format: (p_h, p_w) -> convert to (p_h, p_h, p_w, p_w)
        p_H1, p_H2 = padding[0], padding[0]
        p_W1, p_W2 = padding[1], padding[1]
    elif len(padding) == 4:
        # New format: (p_H1, p_H2, p_W1, p_W2)
        p_H1, p_H2, p_W1, p_W2 = padding
    else:
        raise ValueError(f"img2col: padding must be tuple of length 2 or 4, got {len(padding)}")
    
    # Calculate output dimensions
    h_out = (H + p_H1 + p_H2 - filter_size) // stride + 1
    w_out = (W + p_W1 + p_W2 - filter_size) // stride + 1
    
   

    # Apply asymmetric padding
    if p_H1 > 0 or p_H2 > 0 or p_W1 > 0 or p_W2 > 0:
        # Use NumPy for padding to avoid CuPy compatibility issues
        # if isinstance(X, np.ndarray):
        #     X_paded = np.pad(X, [(0,0), (0,0), (p_H1, p_H2), (p_W1, p_W2)], mode='constant')
        # else:
        #     # For CuPy arrays, move to CPU, pad, then move back
        #     X_cpu = X.get()
        #     X_paded_cpu = np.pad(X_cpu, [(0,0), (0,0), (p_H1, p_H2), (p_W1, p_W2)], mode='constant')
        #     X_paded = to_gpu(X_paded_cpu)
        if hasattr(xp, 'pad'):
            X_paded = xp.pad(X, [(0,0), (0,0), (p_H1, p_H2), (p_W1, p_W2)], mode='constant', constant_values=0)
        else:
            print("Warning: xp does not have pad function, falling back to CPU padding")
    else:
        X_paded = X

    # Validate output dimensions
    if h_out <= 0 or w_out <= 0:
        raise ValueError(f"img2col: Invalid output dimensions h_out={h_out}, w_out={w_out}. "
                        f"Input shape={X.shape}, filter_size={filter_size}, stride={stride}, padding={padding}")
    
    # Validate that we can create the view without accessing invalid memory
    H_padded, W_padded = X_paded.shape[2], X_paded.shape[3]
    max_h_idx = (h_out - 1) * stride + filter_size - 1
    max_w_idx = (w_out - 1) * stride + filter_size - 1
    if max_h_idx >= H_padded or max_w_idx >= W_padded:
        raise ValueError(f"img2col: View would access invalid memory. "
                        f"max_h_idx={max_h_idx} >= H_padded={H_padded}, "
                        f"max_w_idx={max_w_idx} >= W_padded={W_padded}. "
                        f"h_out={h_out}, w_out={w_out}, stride={stride}, filter_size={filter_size}")

    new_shape = (N,h_out,w_out,C,filter_size,filter_size)

    # s_n = C*H*W, s_c = H*W, s_h = W, s_w = 1
    s_n,s_c,s_h,s_w = X_paded.strides
    new_strides = (s_n, # N
                   s_h*stride, # h_out
                   s_w*stride, # w_out
                   s_c, # C
                   s_h, # f_H
                   s_w) # f_W
    
 
    last_h_idx = (h_out - 1) * stride + (filter_size - 1)
    last_w_idx = (w_out - 1) * stride + (filter_size - 1)
    
    
    if last_h_idx >= H_padded or last_w_idx >= W_padded:
        raise ValueError(f"img2col: as_strided would access invalid memory. "
                        f"last_h_idx={last_h_idx} >= H_padded={H_padded} or "
                        f"last_w_idx={last_w_idx} >= W_padded={W_padded}")
    

    try:
        # Use NumPy's as_strided for compatibility, then convert back to CuPy if needed
        # if isinstance(X_paded, np.ndarray):
        #     X_out = np.lib.stride_tricks.as_strided(X_paded, shape=new_shape, strides=new_strides)
        # else:
        #     # For CuPy arrays, use NumPy as_strided on CPU then transfer back
        #     X_paded_cpu = X_paded.get()
        #     X_out_cpu = np.lib.stride_tricks.as_strided(X_paded_cpu, shape=new_shape, strides=new_strides)
        #     X_out = xp.asarray(X_out_cpu)
        X_out = as_strided(X_paded, shape=new_shape, strides=new_strides)
    except Exception as e:
        raise
    return X_out



def col2img(X_col:xp.ndarray, stride:int=1, padding:tuple=(0,0,0,0))->xp.ndarray:
    '''
    Vectorized col2img (optimized loop over filter dimensions only)
    X_col: (N, h_out, w_out, X_C, filter_size, filter_size)
    padding: (p_H1, p_H2, p_W1, p_W2) or (p_h, p_w) for backward compatibility
    return (N, X_C, X_H, X_W)
    '''
    N, h_out, w_out, C, filter_size, _ = X_col.shape
    
    # Handle both old format (p_h, p_w) and new format (p_H1, p_H2, p_W1, p_W2)
    if len(padding) == 2:
        # Old format: (p_h, p_w) -> convert to (p_h, p_h, p_w, p_w)
        p_H1, p_H2 = padding[0], padding[0]
        p_W1, p_W2 = padding[1], padding[1]
    elif len(padding) == 4:
        # New format: (p_H1, p_H2, p_W1, p_W2)
        p_H1, p_H2, p_W1, p_W2 = padding
    else:
        raise ValueError(f"col2img: padding must be tuple of length 2 or 4, got {len(padding)}")
    
    # Calculate output dimensions (original, unpadded)
    H = (h_out - 1) * stride - p_H1 - p_H2 + filter_size
    W = (w_out - 1) * stride - p_W1 - p_W2 + filter_size
    
    # Calculate padded dimensions
    # 卷积操作关键在于卷积核的移动，所以添加padding相当于调整卷积核移动的范围，关键的逆操作是对于卷积核的移动而言的，所以把paded看成被卷积的对象
    H_padded = H + p_H1 + p_H2
    W_padded = W + p_W1 + p_W2
    
    # Initialize output array (padded)
    X_padded = xp.zeros((N, C, H_padded, W_padded), dtype=X_col.dtype)
    
    # Loop only over filter dimensions (e.g. 2x2 or 3x3), which is very small
    for fh in range(filter_size):
        for fw in range(filter_size):
            # Extract values for this filter position across all images and all sliding windows
            val = X_col[:, :, :, :, fh, fw] # (N, h_out, w_out, C)
            
            # Calculate the corresponding slice in the output image
            h_start = fh
            h_end = fh + h_out * stride
            w_start = fw
            w_end = fw + w_out * stride
            
            # Add gradients to the corresponding positions
            # We transpose val to (N, C, h_out, w_out) to match X_padded layout
            X_padded[:, :, h_start:h_end:stride, w_start:w_end:stride] += val.transpose(0, 3, 1, 2)

    # Remove padding
    if p_H1 > 0 or p_H2 > 0 or p_W1 > 0 or p_W2 > 0:
        return X_padded[:, :, p_H1:p_H1+H, p_W1:p_W1+W]
    else:
        return X_padded


def calculate_dynamic_padding(input_size: int, filter_size: int, stride: int, 
                              same_padding: bool = False) -> tuple[int, int]:
    """
    计算动态 padding (p1, p2)，使得：
    1. (input_size + p1 + p2 - filter_size) % stride == 0
    2. (input_size + p1 + p2 - filter_size) // stride + 1 > 0
    
    Args:
        input_size: 输入尺寸 (H 或 W)
        filter_size: 卷积核或池化核大小
        stride: 步长
        same_padding: 是否使用 same padding（输出尺寸 = 输入尺寸）
    
    Returns:
        (p1, p2): 不对称 padding，优先对称分配
    """
    if same_padding:
        # Same padding: output_size = input_size
        # (input_size + p1 + p2 - filter_size) // stride + 1 = input_size
        # => p1 + p2 = (input_size - 1) * stride + filter_size - input_size
        total_p = (input_size - 1) * stride + filter_size - input_size
        
        if total_p < 0:
            # 无法实现 same padding，回退到最小 padding
            return calculate_dynamic_padding(input_size, filter_size, stride, same_padding=False)
        
        # 分配 padding（优先对称）
        if total_p % 2 == 0:
            return (total_p // 2, total_p // 2)
        else:
            # 奇数时，上/左多 1
            return ((total_p + 1) // 2, total_p // 2)
    
    # 动态 padding：计算最小 padding 使得可被 stride 整除且输出 > 0
    # 需要满足：(input_size + total_p - filter_size) % stride == 0
    # 需要满足：(input_size + total_p - filter_size) // stride + 1 > 0
    
    # 计算余数
    remainder = (input_size - filter_size) % stride
    
    # 计算使可被 stride 整除的最小 padding
    if remainder == 0:
        total_p_min = 0
    else:
        total_p_min = stride - remainder
    
    # 确保输出维度 > 0: input_size + total_p - filter_size >= 0
    # => total_p >= filter_size - input_size
    total_p_required = max(0, filter_size - input_size)
    
    # 取两者最大值，并确保是 stride 的倍数（如果需要）
    total_p = max(total_p_min, total_p_required)
    
    # 如果 total_p 不满足整除条件，向上取整到下一个 stride 的倍数
    if (input_size + total_p - filter_size) % stride != 0:
        # 计算需要增加多少才能被 stride 整除
        current_remainder = (input_size + total_p - filter_size) % stride
        total_p += stride - current_remainder
    
    # 验证输出维度 > 0
    h_out = (input_size + total_p - filter_size) // stride + 1
    if h_out <= 0:
        # 如果仍然 <= 0，增加一个 stride
        total_p += stride
        h_out = (input_size + total_p - filter_size) // stride + 1
    
    # 分配 padding（优先对称）
    if total_p % 2 == 0:
        return (total_p // 2, total_p // 2)
    else:
        # 奇数时，上/左多 1
        return ((total_p + 1) // 2, total_p // 2)

    # # ========== 旧的 padding 计算代码（已注释） ==========
        # # Calculate minimum padding needed to ensure positive output dimensions
        # # and that (H + p_H1 + p_H2 - filter_size) is divisible by stride
        # max_padding = max(H, W) + 10  # Safety limit
        # 
        # # For height dimension
        # p_H1, p_H2 = 0, 0
        # total_p_h = 0
        # while (H + total_p_h - self.filter_size) % self.stride != 0 or \
        #     (H + total_p_h - self.filter_size) // self.stride + 1 <= 0:
        #     total_p_h += 1
        #     if total_p_h > max_padding:
        #         raise ValueError(f"Conv: Failed to find valid padding for H={H}, filter_size={self.filter_size}, stride={self.stride}")
        # 
        # # Distribute padding symmetrically if possible, otherwise asymmetrically
        # if total_p_h % 2 == 0:
        #     p_H1 = p_H2 = total_p_h // 2
        # else:
        #     # For odd total padding, prefer more padding on top (p_H1)
        #     p_H1 = (total_p_h + 1) // 2
        #     p_H2 = total_p_h - p_H1
        # 
        # # For width dimension
        # p_W1, p_W2 = 0, 0
        # total_p_w = 0
        # while (W + total_p_w - self.filter_size) % self.stride != 0 or \
        #     (W + total_p_w - self.filter_size) // self.stride + 1 <= 0:
        #     total_p_w += 1
        #     if total_p_w > max_padding:
        #         raise ValueError(f"Conv: Failed to find valid padding for W={W}, filter_size={self.filter_size}, stride={self.stride}")
        # 
        # # Distribute padding symmetrically if possible, otherwise asymmetrically
        # if total_p_w % 2 == 0:
        #     p_W1 = p_W2 = total_p_w // 2
        # else:
        #     # For odd total padding, prefer more padding on left (p_W1)
        #     p_W1 = (total_p_w + 1) // 2
        #     p_W2 = total_p_w - p_W1
        # 
        # # Apply same_padding if requested
        # if (self.same_padding):
        #     # Calculate same padding: output size = input size
        #     # (H + p_H1 + p_H2 - filter_size) // stride + 1 = H
        #     # => H + p_H1 + p_H2 - filter_size = (H - 1) * stride
        #     # => p_H1 + p_H2 = (H - 1) * self.stride + self.filter_size - H
        #     total_p_h_same = (H - 1) * self.stride + self.filter_size - H
        #     if total_p_h_same >= 0 and (H + total_p_h_same - self.filter_size) // self.stride + 1 > 0:
        #         if total_p_h_same % 2 == 0:
        #             p_H1 = p_H2 = total_p_h_same // 2
        #         else:
        #             p_H1 = (total_p_h_same + 1) // 2
        #             p_H2 = total_p_h_same - p_H1
        #     else:
        #         print(f"(H - 1) * self.stride + self.filter_size - H) < 0 or invalid, H={H}, stride={self.stride}, filter_size={self.filter_size}, keep normal padding")
        #     
        #     total_p_w_same = (W - 1) * self.stride + self.filter_size - W
        #     if total_p_w_same >= 0 and (W + total_p_w_same - self.filter_size) // self.stride + 1 > 0:
        #         if total_p_w_same % 2 == 0:
        #             p_W1 = p_W2 = total_p_w_same // 2
        #         else:
        #             p_W1 = (total_p_w_same + 1) // 2
        #             p_W2 = total_p_w_same - p_W1
        #     else:
        #         print(f"(W - 1) * self.stride + self.filter_size - W) < 0 or invalid, W={W}, stride={self.stride}, filter_size={self.filter_size}, keep normal padding")
        # 
        # # Store as (p_H1, p_H2, p_W1, p_W2)
        # self.padding = (p_H1, p_H2, p_W1, p_W2)
        # # ========== 旧的 padding 计算代码结束 ==========



class layer:
    def forward_prop(self, X: xp.ndarray, training:bool=True) -> xp.ndarray:
        pass
    def backward_prop(self, dY: xp.ndarray) -> xp.ndarray:
        pass
    def modified_hyperparam(self, learn_rate=None, _Adam=None, beta1=None, beta2=None, epsilon=None):
        pass # 无参数层不需要更新超参数
    def get_config(self): return None
    def set_config(self, config:dict): pass
    def get_weights(self): return None
    def set_weights(self, weights:dict): pass


class TrainableLayer(layer):
    def __init__(self, learning_rate=0.001, _Adam=False, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self._Adam = _Adam
        self.Adam_beta1 = Adam_beta1
        self.Adam_beta2 = Adam_beta2
        self.epsilon = epsilon

    def modified_hyperparam(self, learn_rate=None, _Adam=None, beta1=None, beta2=None, epsilon=None):
        self.learning_rate = learn_rate
        if _Adam is not None: self._Adam = _Adam
        if beta1 is not None: self.Adam_beta1 = beta1
        if beta2 is not None: self.Adam_beta2 = beta2
        if epsilon is not None: self.epsilon = epsilon



class Conv(TrainableLayer):
    def __init__(
            self, filter_num, filter_size, filter_channel, stride=1, 
            _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
            learning_rate=0.001, same_padding:bool=False):
        TrainableLayer.__init__(self, learning_rate=learning_rate, _Adam=_Adam, Adam_beta1=Adam_beta1, Adam_beta2=Adam_beta2, epsilon=epsilon)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.filter_channel = filter_channel
        self.stride = stride

        """ 
        (H + 2p - f) ≡ 0 mod s
        (W + 2p - f) ≡ 0 mod s
        推出：
        H - f ≡ -2p mod s
        W - f ≡ -2p mod s
        说明H和W必须mod s同余
        设计者必须仔细设计s以保证输出为整数
        或者，选择长宽不同的filter或者padding!!!
        """
        self.padding = None
        self.same_padding = same_padding

    #--------tmp array, transfer to cpu--------------#
        # initialize filters and bias
        self.F = np.random.randn(filter_channel * filter_size * filter_size, filter_num) / np.sqrt(filter_size * filter_size * filter_channel)
            # maybe more way to initialize filters
        self.bias = np.zeros((1, filter_num))  # (1, f_n^{l})
        # update everytime forward_prop is called
        self.X_col = None 
    #------------------------------------------------#
        self.col_shape = None

    #--------tmp array, transfer to cpu--------------#
        # update params
        self.S_F = np.zeros_like(self.F)
        self.V_F = np.zeros_like(self.F)
        self.S_bias = np.zeros_like(self.bias)
        self.V_bias = np.zeros_like(self.bias)
    #------------------------------------------------#
    

    def get_config(self):
        return {
            'type': 'Conv',
            'filter_num': self.filter_num,
            'filter_size': self.filter_size,
            'filter_channel': self.filter_channel,
            'stride': self.stride,

            'learning_rate': self.learning_rate,
            '_Adam': self._Adam,
            'Adam_beta1': self.Adam_beta1,
            'Adam_beta2': self.Adam_beta2,
            'epsilon': self.epsilon,
            'same_padding': self.same_padding
        }
    def get_weights(self):
        return {
            'F': to_cpu(self.F),
            'bias': to_cpu(self.bias),

            'S_F': to_cpu(self.S_F),
            'V_F': to_cpu(self.V_F),
            'S_bias': to_cpu(self.S_bias),
            'V_bias': to_cpu(self.V_bias),
        }
    
    def set_config(self, config:dict):
        self.filter_num = config['filter_num']
        self.filter_size = config['filter_size']
        self.filter_channel = config['filter_channel']
        self.stride = config['stride']
    
        self.learning_rate = config['learning_rate']
        self._Adam = config['_Adam']
        self.Adam_beta1 = config['Adam_beta1']
        self.Adam_beta2 = config['Adam_beta2']
        self.epsilon = config['epsilon']
        self.same_padding = config['same_padding']
    def set_weights(self, weights:dict):
        self.F = to_gpu(weights['F'])
        self.bias = to_gpu(weights['bias'])
        self.S_F = to_gpu(weights['S_F'])
        self.V_F = to_gpu(weights['V_F'])
        self.S_bias = to_gpu(weights['S_bias'])
        self.V_bias = to_gpu(weights['V_bias'])
       
        
    def forward_prop(self, X:xp.ndarray)->xp.ndarray:
        ''' 
        X: (N, X_C^{l-1}, X_H^{l-1}, X_W^{l-1})
        F: (f_c^{l}*f_size^{l}*f_size^{l})
        Returns: (N, f_n^{l}, Z_H^{l}, Z_W^{l})
        '''
    #--------tmp array, transfer to gpu--------------#
        self.F = to_gpu(self.F)
        self.bias = to_gpu(self.bias)
        if hasattr(self, 'X_col') and self.X_col is not None:
            self.X_col = to_gpu(self.X_col) # X_col 只有在第一次未初始化是None
    #------------------------------------------------#


        N,C,H,W = X.shape
        # adjust padding dynamically with flexible asymmetric padding
        if self.padding is None:
  
            # 使用改进的 O(1) 算法计算 padding
            p_H1, p_H2 = calculate_dynamic_padding(H, self.filter_size, self.stride, self.same_padding)
            p_W1, p_W2 = calculate_dynamic_padding(W, self.filter_size, self.stride, self.same_padding)
            self.padding = (p_H1, p_H2, p_W1, p_W2)
        
        self.X_col = img2col(X,filter_size=self.filter_size,stride=self.stride,padding=self.padding)
        self.col_shape = self.X_col.shape # 每次forward_prop都要更新col_shape
        self.X_col = self.X_col.reshape(self.col_shape[0]*self.col_shape[1]*self.col_shape[2],-1) # (N * h_out * w_out, X_C^{l-1} * filter_size^{l} * filter_size^{l})

        # Debug/validation: ensure matmul inner dimensions match
        if self.X_col.shape[1] != self.F.shape[0]:
            raise ValueError(
                f"Conv.forward_prop shape mismatch: "
                f"X.shape={X.shape}, X_col.shape={self.X_col.shape}, "
                f"F.shape={self.F.shape}, "
                f"expected inner dim={self.F.shape[0]} (= filter_channel * filter_size^2), "
                f"got {self.X_col.shape[1]}"
            )

        Z = self.X_col @ self.F + self.bias # (N * Z_H^{l} * Z_W^{l}, f_n^{l})
        Z = Z.reshape(self.col_shape[0],self.col_shape[1],self.col_shape[2],self.filter_num)# (N, Z_H^{l}, Z_W^{l}, f_n^{l})
        Z = Z.transpose(0,3,1,2)# (N, f_n^{l}, Z_H^{l}, Z_W^{l})

    #--------tmp array, transfer to cpu--------------#
        self.F = to_cpu(self.F)
        self.bias = to_cpu(self.bias)
        self.X_col = to_cpu(self.X_col)
    #------------------------------------------------#
        return Z
    

    def backward_prop(self, d_Z:xp.ndarray)->xp.ndarray:
        '''
        d_Z: (N, f_n^{l}, Z_H^{l}, Z_W^{l}) (统一维度设计)
        F: (X_C^{l-1} * filter_size^{l} * filter_size^{l}, f_n^{l})
        Returns: (N, X_C^{l-1}, X_H^{l-1}, X_W^{l-1})
        '''
    #--------tmp array, transfer to gpu--------------#
        self.F = to_gpu(self.F)
        self.bias = to_gpu(self.bias)
        self.X_col = to_gpu(self.X_col)

        self.S_F = to_gpu(self.S_F)
        self.V_F = to_gpu(self.V_F)
        self.S_bias = to_gpu(self.S_bias)
        self.V_bias = to_gpu(self.V_bias)
    #------------------------------------------------#

        d_Z = d_Z.transpose(0,2,3,1).reshape(-1,self.filter_num) # (N * Z_H^{l} * Z_W^{l}, f_n^{l})
        
        d_F = self.X_col.T @ d_Z # (X_C * filter_size * filter_size, f_n^{l})
        d_bias = d_Z.sum(axis=0,keepdims=True).reshape(1,-1) # (1,f_n^{l})
        
        # Gradient Clipping to prevent explosion
        d_F = xp.clip(d_F, -5.0, 5.0)
        d_bias = xp.clip(d_bias, -5.0, 5.0)
        
        d_X_col = d_Z @ self.F.T # (N * Z_H^{l} * Z_W^{l}, X_C * filter_size * filter_size)
        # calculating gradient must be earlier than updating params
        if self._Adam:
            self.V_F = self.Adam_beta1 * self.V_F + (1-self.Adam_beta1) * d_F # (X_C * filter_size * filter_size, f_n^{l})
            self.S_F = self.Adam_beta2 * self.S_F + (1-self.Adam_beta2) * xp.power(d_F,2) # (X_C * filter_size * filter_size, f_n^{l})
            self.F = self.F - self.learning_rate * self.V_F / (xp.sqrt(self.S_F) + self.epsilon) # (X_C * filter_size * filter_size, f_n^{l})
            self.V_bias = self.Adam_beta1 * self.V_bias + (1-self.Adam_beta1) * d_bias # (1,f_n^{l})
            self.S_bias = self.Adam_beta2 * self.S_bias + (1-self.Adam_beta2) * xp.power(d_bias,2) # (1,f_n^{l})
            self.bias = self.bias - self.learning_rate * self.V_bias / (xp.sqrt(self.S_bias) + self.epsilon) # (1,f_n^{l})
        else:
            self.F = self.F - self.learning_rate * d_F # (X_C * filter_size * filter_size, f_n^{l})
            self.bias = self.bias - self.learning_rate * d_bias # (1,f_n^{l})  
        d_X_col = d_X_col.reshape(self.col_shape) # (N, h_out, w_out, filter_c, filter_size, filter_size)
        d_X = col2img(d_X_col,stride=self.stride, padding=self.padding) # (N,X_C^{l-1}, X_H^{l-1}, X_W^{l-1})

    #--------tmp array, transfer to cpu--------------#
        self.F = to_cpu(self.F)
        self.bias = to_cpu(self.bias)
                        # self.X_col = to_cpu(self.X_col)
        # clear X_col from memory since it's only needed during forward/backward pass and can be large
        del self.X_col
        cp.get_default_memory_pool().free_all_blocks()
        # self.X_col = None

        self.S_F = to_cpu(self.S_F)
        self.V_F = to_cpu(self.V_F)
        self.S_bias = to_cpu(self.S_bias)
        self.V_bias = to_cpu(self.V_bias)
    #------------------------------------------------#
        return d_X
        
        


class BatchNorm(TrainableLayer):
    def __init__(self,
            _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
            momentum=0.8,
            learning_rate=0.001):
        TrainableLayer.__init__(self, learning_rate=learning_rate, _Adam=_Adam, Adam_beta1=Adam_beta1, Adam_beta2=Adam_beta2, epsilon=epsilon)

        self.input_shape = None

    #--------tmp array, transfer to cpu--------------#
        # tmp params
        self.mu = None
        self.sigma = None
        self.y_hat = None
        # self.y_tilde = None 没用上

        # Running statistics for inference
        self.running_mean = None
        self.running_var = None
    #------------------------------------------------#

        self.momentum = momentum

    #--------tmp array, transfer to cpu--------------#
        # learnable parameters
        self.gamma = None
        self.beta = None

        self.S_gamma = None
        self.V_gamma = None
        self.S_beta = None
        self.V_beta = None
    #------------------------------------------------#
        
    def get_config(self):
        return {
            'type': 'BatchNorm',

            '_Adam': self._Adam,
            'Adam_beta1': self.Adam_beta1,
            'Adam_beta2': self.Adam_beta2,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'learning_rate': self.learning_rate,
        }
    def get_weights(self):
        return {
            'gamma': to_cpu(self.gamma),
            'beta': to_cpu(self.beta),
            # 保存推理用的统计量
            'running_mean': to_cpu(self.running_mean),
            'running_var': to_cpu(self.running_var),
            
            'S_gamma': to_cpu(self.S_gamma),
            'V_gamma': to_cpu(self.V_gamma),
            'S_beta': to_cpu(self.S_beta),
            'V_beta': to_cpu(self.V_beta),
        }

    def set_config(self, config:dict):
        self._Adam = config['_Adam']
        self.Adam_beta1 = config['Adam_beta1']
        self.Adam_beta2 = config['Adam_beta2']
        self.epsilon = config['epsilon']
        self.momentum = config['momentum']
        self.learning_rate = config['learning_rate']
    def set_weights(self, weights:dict):

        self.gamma = to_gpu(weights.get('gamma', self.gamma))
        self.beta = to_gpu(weights.get('beta', self.beta))
        self.running_mean = to_gpu(weights.get('running_mean', self.running_mean))
        self.running_var = to_gpu(weights.get('running_var', self.running_var))

        self.S_gamma = to_gpu(weights.get('S_gamma', self.S_gamma))
        self.V_gamma = to_gpu(weights.get('V_gamma', self.V_gamma))
        self.S_beta = to_gpu(weights.get('S_beta', self.S_beta))
        self.V_beta = to_gpu(weights.get('V_beta', self.V_beta))

        
    def forward_prop(self, Z:xp.ndarray, training:bool=True)->xp.ndarray:

    #--------tmp array, transfer to gpu--------------#
    # 其实在反向传播用完就可以丢弃了，所以在这个位置应该是None的
        # self.mu = to_gpu(self.mu)
        # self.sigma = to_gpu(self.sigma)
        # self.y_hat = to_gpu(self.y_hat) 

        self.running_mean = to_gpu(self.running_mean)
        self.running_var = to_gpu(self.running_var)
        if hasattr(self, 'gamma') and self.gamma is not None:
            self.gamma = to_gpu(self.gamma)
        if hasattr(self, 'beta') and self.beta is not None:
            self.beta = to_gpu(self.beta)
        
    #------------------------------------------------#
        
        if self.input_shape is None:
            self.input_shape = tuple([1]+list(Z.shape[1:]))
            
        if self.gamma is None:
            self.gamma = xp.ones(self.input_shape)
            self.S_gamma = np.zeros_like(self.gamma)
            self.V_gamma = np.zeros_like(self.gamma)
        if self.beta is None:
            self.beta = xp.zeros(self.input_shape)
            self.S_beta = np.zeros_like(self.beta)
            self.V_beta = np.zeros_like(self.beta)

        # 初始化 running statistics
        if self.running_mean is None:
            self.running_mean = xp.zeros(self.input_shape)
        if self.running_var is None:
            self.running_var = xp.ones(self.input_shape)

        if training:
            if Z.ndim == 4:
                # Spatial BN for CNN: (N, C, H, W) -> mean over (N, H, W)
                self.mu = Z.mean(axis=(0, 2, 3), keepdims=True)
                self.sigma = xp.var(Z, axis=(0, 2, 3), keepdims=True)
            else:
                # Standard BN for FC: (N, D) -> mean over (N)
                self.mu = Z.mean(axis=0, keepdims=True)
                self.sigma = xp.var(Z, axis=0, keepdims=True)
          
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sigma
          
            self.y_hat = (Z - self.mu) / xp.sqrt(self.sigma + self.epsilon)
           
        else:
            # 推理模式：使用训练过程中累积的 running statistics 不需要传给backward_prop
            mu = self.running_mean
            sigma = self.running_var
          
            self.y_hat = (Z - mu) / xp.sqrt(sigma + self.epsilon)# 预测时候虽然更新y_hat但是训练时会覆盖更新，不会出错
        
        y_tilde = self.gamma * self.y_hat + self.beta    
    #--------tmp array, transfer to cpu--------------#
        if training:
            self.mu = to_cpu(self.mu)
            self.sigma = to_cpu(self.sigma)
        self.running_mean = to_cpu(self.running_mean)
        self.running_var = to_cpu(self.running_var)
        self.y_hat = to_cpu(self.y_hat)
        self.gamma = to_cpu(self.gamma)
        self.beta = to_cpu(self.beta)
    #------------------------------------------------#
        return y_tilde
    
    def backward_prop(self, d_y_tilde:xp.ndarray)->xp.ndarray:
    #--------tmp array, transfer to gpu--------------#
        self.sigma = to_gpu(self.sigma)
        self.y_hat = to_gpu(self.y_hat)

        self.gamma = to_gpu(self.gamma)
        self.beta = to_gpu(self.beta)
        self.S_gamma = to_gpu(self.S_gamma)
        self.V_gamma = to_gpu(self.V_gamma)
        self.S_beta = to_gpu(self.S_beta)
        self.V_beta = to_gpu(self.V_beta)
    #------------------------------------------------#

        # Batch dimension is always axis 0 now
        if d_y_tilde.ndim == 4:
            axis = (0, 2, 3)
        else:
            axis = 0
        d_gamma = (d_y_tilde * self.y_hat).sum(axis=axis,keepdims=True) 
        d_beta = d_y_tilde.sum(axis=axis,keepdims=True)

        B = d_y_tilde.mean(axis=axis,keepdims=True)
        C = (d_y_tilde * self.y_hat).mean(axis=axis,keepdims=True)
        D = self.y_hat * C

        if self._Adam:
            self.V_gamma = self.Adam_beta1 * self.V_gamma + (1-self.Adam_beta1) * d_gamma
            self.S_gamma = self.Adam_beta2 * self.S_gamma + (1-self.Adam_beta2) * xp.power(d_gamma,2)
            self.gamma = self.gamma - self.learning_rate * self.V_gamma / (xp.sqrt(self.S_gamma) + self.epsilon)
            self.V_beta = self.Adam_beta1 * self.V_beta + (1-self.Adam_beta1) * d_beta
            self.S_beta = self.Adam_beta2 * self.S_beta + (1-self.Adam_beta2) * xp.power(d_beta,2)
            self.beta = self.beta - self.learning_rate * self.V_beta / (xp.sqrt(self.S_beta) + self.epsilon)
        else:
            # Gradient Clipping for simple SGD
            d_gamma = xp.clip(d_gamma, -5.0, 5.0)
            d_beta = xp.clip(d_beta, -5.0, 5.0)
            self.gamma = self.gamma - self.learning_rate * d_gamma
            self.beta = self.beta - self.learning_rate * d_beta
        d_Z = (self.gamma / xp.sqrt(self.sigma + self.epsilon)) * (d_y_tilde - B - D)

    #--------tmp array, transfer to cpu--------------#
                # self.sigma = to_cpu(self.sigma)
                # self.y_hat = to_cpu(self.y_hat)
        # 清空y_hat和sigma, mu以节省内存，因为它们在下一次forward_prop时会被覆盖更新
        # self.mu = None
        # self.sigma = None
        # self.y_hat = None
        del self.mu
        del self.sigma
        del self.y_hat
        cp.get_default_memory_pool().free_all_blocks()

        self.gamma = to_cpu(self.gamma)
        self.beta = to_cpu(self.beta)
        self.S_gamma = to_cpu(self.S_gamma)
        self.V_gamma = to_cpu(self.V_gamma)
        self.S_beta = to_cpu(self.S_beta)
        self.V_beta = to_cpu(self.V_beta)
    #------------------------------------------------#
        return d_Z





class Activation(layer):
    def __init__(self, activation:str='relu'):
        self.activation = activation
        self.Z = None
        self.Map = None
        self.Indice = None

    def get_config(self):
        return {
            'type': 'Activation',
            'activation': self.activation
        }

    def set_config(self, config:dict):
        self.activation = config['activation']

    
    @staticmethod
    def sigmoid(X:xp.ndarray)->xp.ndarray:
        # Numerically stable sigmoid
        X = X.clip(-100, 100)
        return xp.where(X >= 0,
                       1 / (1 + xp.exp(-X)),
                       xp.exp(X) / (1 + xp.exp(X)))
        # return 1 / (1 + xp.exp(-X))
    def forward_prop(self, Z:xp.ndarray)->xp.ndarray:
        """
        Z: (N, ...)
        return: (N, ...) same as Z
        """
    #--------tmp array, transfer to gpu--------------#
        if hasattr(self, 'Map') and self.Map is not None:
            self.Map = to_gpu(self.Map)
        if hasattr(self, 'Indice') and self.Indice is not None:
            self.Indice = to_gpu(self.Indice)
    #------------------------------------------------#

        # 临时类型检查----------------------------------------------------------------------------------------------------------------------
        if isinstance(Z, np.ndarray):
# ---------------------------------------不知道为什么会传入numpyarray
            # print("Warning: Activation.forward_prop received numpy array, converting to cupy array for GPU computation")
            Z = to_gpu(Z)

        self.Z = Z
        # to ensure activation function's emcapsulation, do not do reshape operation
        if self.activation == 'relu':
            self.Map = Z > 0
            out = Z * self.Map
            self.Z = to_cpu(self.Z) # 只保存输入Z到CPU，节省内存
            self.Map = to_cpu(self.Map) # 只保存bool类型的Map到CPU，节省内存
            return out
            # return Z * self.Map
        elif self.activation == 'leaky_relu':
            out = xp.where(Z > 0, Z, 0.01 * Z)
            # self.Z = to_cpu(self.Z) # 只保存输入Z到CPU，节省内存
            return out
            # return xp.where(Z > 0, Z, 0.01 * Z)
        elif self.activation == 'sigmoid':
            out = self.sigmoid(Z)
            # self.Z = to_cpu(self.Z) # 只保存输入Z到CPU，节省内存
            return out
            # return self.sigmoid(Z)  
        elif self.activation == 'softmax':
            if len(Z.shape) != 2: # (N, D_out)
                raise ValueError('FC layer softmax only support 2D input')
            # 对于语义分割，通常在通道维度进行softmax
            # 对于空间注意力，通常在H*W维度进行softmax
            exp_X = xp.exp(Z - xp.max(Z, axis=1, keepdims=True)) # 数值稳定版本
            self.Indice = exp_X / xp.sum(exp_X, axis=1, keepdims=True)
            self.Z = to_cpu(self.Z) # 只保存输入Z到CPU，节省内存
            self.Indice = to_cpu(self.Indice) # 只保存softmax输出的概率分布到CPU，节省内存
            return to_gpu(self.Indice)
            # return self.Indice
        elif self.activation == 'tanh':
            out = xp.tanh(Z)
            # self.Z = to_cpu(self.Z) # 只保存输入Z到CPU，节省内存
            return out
            # return xp.tanh(Z)
        else:
            raise ValueError('activation must be relu or sigmoid or softmax')
    def backward_prop(self, d_A:xp.ndarray)->xp.ndarray:
        """
        d_A: (N, ...) same as Z
        return: (N, ...) same as Z
        """
    #--------tmp array, transfer to gpu--------------#
        if hasattr(self, 'Z') and self.Z is not None:
            self.Z = to_gpu(self.Z)
        if hasattr(self, 'Map') and self.Map is not None:
            self.Map = to_gpu(self.Map)
        if hasattr(self, 'Indice') and self.Indice is not None:
            self.Indice = to_gpu(self.Indice)
    #------------------------------------------------#

        if self.activation == 'relu':
            d_Z = d_A * self.Map
            del self.Map # 反向传播用完就丢弃，节省内存
        elif self.activation == 'leaky_relu':
            d_Z = xp.where(self.Z > 0, d_A, 0.01 * d_A)
        elif self.activation == 'sigmoid':
            sigmoid_z = self.sigmoid(self.Z)
            d_Z = sigmoid_z * (1 - sigmoid_z) * d_A
        elif self.activation == 'softmax':
            d_Z = (self.Indice - d_A) / self.Z.shape[0] # y_hat-y_onehot / N
            # d_Z = (self.Indice - d_A) # y_hat-y_onehot
            del self.Indice # 反向传播用完就丢弃，节省内存
        elif self.activation == 'tanh':
            d_Z = (1 - xp.tanh(self.Z)**2) * d_A
        else:
            raise ValueError('activation must be relu or sigmoid')
    #--------tmp array, transfer to cpu--------------#  
        del self.Z
        cp.get_default_memory_pool().free_all_blocks()
    #------------------------------------------------#
        return d_Z


class Pooling(layer):# check
    def __init__(self,stride=1,
             pool_size=2,pool_type='max', same_padding:bool=False):
        self.padding = None
        self.stride = stride
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.same_padding = same_padding

        self.onehot = None
        # self.A_col = None

    def get_config(self):
        return {
            'type': 'Pooling',
            
            'stride': self.stride,
            'pool_size': self.pool_size,
            'pool_type': self.pool_type,
            'same_padding': self.same_padding
        }
   

    def set_config(self, config:dict):
        self.stride = config['stride']
        self.pool_size = config['pool_size']
        self.pool_type = config['pool_type']
        self.same_padding = config['same_padding']
    


    def forward_prop(self,A:xp.ndarray)->xp.ndarray:
        """
        A: (N, f_n^{l}, Z_H^{l}, Z_W^{l})(统一维度设计)
        Returns: (N, f_n^{l}, h_out, w_out)(统一维度设计)
        """
      
        N, f_n, H, W = A.shape
        # Save input shape for backward pass
        self.input_shape = (N, f_n, H, W)
        if self.padding is None:
            # 使用改进的 O(1) 算法计算 padding
            p_H1, p_H2 = calculate_dynamic_padding(H, self.pool_size, self.stride, self.same_padding)
            p_W1, p_W2 = calculate_dynamic_padding(W, self.pool_size, self.stride, self.same_padding)
            self.padding = (p_H1, p_H2, p_W1, p_W2)
        
   
        A = A.reshape(N*f_n, 1, H, W)  # (N*f_n, 1, H, W) # C = 1
    
        A_col = img2col(A, filter_size=self.pool_size, stride=self.stride, padding=self.padding) 
    
        self.A_col_shape = A_col.shape
        
       
        A_col_flat = A_col.reshape(self.A_col_shape[0]*self.A_col_shape[1]*self.A_col_shape[2], -1) # (N*f_n*h_out*w_out, pool_size*pool_size)  
       
        
        # Apply pooling along the pooling window dimension (axis=1)
        if self.pool_type == 'max':
      
            X_flat = A_col_flat.max(axis=1)  # (N*f_n*h_out*w_out,)
          
            Indices = A_col_flat.argmax(axis=1)  # (N*f_n*h_out*w_out,)
         
            self.onehot = xp.zeros(A_col_flat.shape)  # (N*f_n*h_out*w_out, pool_size*pool_size)
            self.onehot[xp.arange(Indices.shape[0]), Indices] = 1 # (N*f_n*h_out*w_out, pool_size*pool_size)
       
            
        elif self.pool_type == 'avg':
         
            X_flat = A_col_flat.mean(axis=1)# (N*f_n*h_out*w_out,)
         
            self.onehot = None
        else:
            raise ValueError('pool_type must be max or avg')
        
        # Reshape back: (N, f_n, h_out, w_out)
        h_out, w_out = self.A_col_shape[1], self.A_col_shape[2]
     
        X = X_flat.reshape(N, f_n, h_out, w_out)
    

        return X  # (N, f_n, h_out, w_out)
    def backward_prop(self,d_X:xp.ndarray)->xp.ndarray:
        """
        d_X: (N, f_n^{l}, h_out, w_out)(统一维度设计)
        Returns: (N, f_n^{l}, H, W)
        """

        N, f_n, h_out, w_out = d_X.shape
        
        # Flatten d_X to match the pooled output shape
        d_X_flat = d_X.reshape(-1,1)  # (N*f_n*h_out*w_out,)
        
        
        # Create gradient array for A_col_flat: (N*f_n*h_out*w_out, pool_size*pool_size)
        pool_window_size = self.pool_size * self.pool_size
        
        weight = xp.ones((d_X_flat.shape[0],pool_window_size)) / 4.0

        d_A = None
        if self.pool_type == 'max' and self.onehot is not None:
            d_A_col_flat = self.onehot * d_X_flat # (N*f_n*h_out*w_out, pool_size*pool_size)
        else:
            d_A_col_flat = weight * d_X_flat # (N*f_n*h_out*w_out, pool_size*pool_size)
        d_A_col = d_A_col_flat.reshape(N*f_n, h_out, w_out, 1, self.pool_size, self.pool_size)
        d_A = col2img(d_A_col,stride=self.stride, padding=self.padding) # (N*f_n, 1, H, W)   
        
        return d_A.reshape(N, f_n, d_A.shape[2], d_A.shape[3]) # (N, f_n, H, W)  
        
        

# 不知道为什么FC无法tocpu优化内存
class FC(layer): # fully connected layer
    def __init__(self,output_size,
                _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
                learning_rate=0.001):
        # Adam hyperparam        
        TrainableLayer.__init__(self, learning_rate=learning_rate, _Adam=_Adam, Adam_beta1=Adam_beta1, Adam_beta2=Adam_beta2, epsilon=epsilon)
        self.input_size = None # default None, will be set in forward_prop
        self.output_size = output_size

    #--------tmp array, transfer to cpu--------------#  
        self.A = None
        self.shape = None

        self.W = None
        # bias shape (1, output_size) to broadcast with (output_size, N)
        self.b = xp.zeros((1, output_size))


        self.S_W = None  # Will be initialized when W is created
        self.V_W = None
        self.S_b = xp.zeros_like(self.b)
        self.V_b = xp.zeros_like(self.b)
    #------------------------------------------------#

    def get_config(self):
        return {
            'type': 'FC',
            'output_size': self.output_size,
            '_Adam': self._Adam,
            'Adam_beta1': self.Adam_beta1,
            'Adam_beta2': self.Adam_beta2,
            'epsilon': self.epsilon,
        }
    def get_weights(self):
        return {
            'W': to_cpu(self.W),
            'b': to_cpu(self.b),

            'S_W': to_cpu(self.S_W),
            'V_W': to_cpu(self.V_W),
            'S_b': to_cpu(self.S_b),
            'V_b': to_cpu(self.V_b),
        }

    def set_config(self, config:dict):
        self.output_size = config['output_size']
        self.learning_rate = config['learning_rate']
        self._Adam = config['_Adam']
        self.Adam_beta1 = config['Adam_beta1']
        self.Adam_beta2 = config['Adam_beta2']
        self.epsilon = config['epsilon']        
    def set_weights(self, weights:dict):
        self.W = to_gpu(weights['W'])
        self.b = to_gpu(weights['b'])
        self.S_W = to_gpu(weights['S_W'])
        self.V_W = to_gpu(weights['V_W'])   
        self.S_b = to_gpu(weights['S_b'])
        self.V_b = to_gpu(weights['V_b'])




    def modified_hyperparam(self, learning_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        self.learning_rate = learning_rate
        if _Adam is not None:
            self._Adam = _Adam
        if beta1 is not None:
            self.Adam_beta1 = beta1
        if beta2 is not None:
            self.Adam_beta2 = beta2
        if epsilon is not None:
            self.epsilon = epsilon
    def forward_prop(self,A:xp.ndarray)->xp.ndarray:
        """
        A: Can be (N, f_n^{l}, X_H^{l}, X_W^{l}) from conv/pool layers or (N, D_out_prev) from FC layers
        W: (D_in, D_out)
        return Z: (N, D_out)
        """
    #--------tmp array, transfer to gpu--------------#  
        # if hasattr(self, 'W') and self.W is not None:
        #     self.W = to_gpu(self.W)
        # if hasattr(self, 'b') and self.b is not None:
        #     self.b = to_gpu(self.b)
    #------------------------------------------------#

        self.shape = A.shape # (N, f_n^{l}, X_H^{l}, X_W^{l}) or (N, D_out_prev)
        if len(A.shape) == 4: # (N, C, H, W)
            # Flatten the input to (N, D_in)
            self.A = A.reshape(A.shape[0], -1)  # (N, D_in)
        elif len(A.shape) == 2: # (N, D_in)
            self.A = A
        else:
            raise ValueError(f"Unsupported input shape: {A.shape}. Expected 2D or 4D input.")

        if self.W is None:
            self.input_size = self.A.shape[1]
            self.W = xp.random.randn(self.input_size,self.output_size) * xp.sqrt(2/self.input_size)
            self.S_W = xp.zeros_like(self.W)
            self.V_W = xp.zeros_like(self.W)

        Z = self.A @ self.W + self.b # (N, D_out)
    #--------tmp array, transfer to cpu--------------#  
        # self.A = to_cpu(self.A)
        # self.W = to_cpu(self.W)
        # self.b = to_cpu(self.b)
    #------------------------------------------------#
        return Z
    

    def backward_prop(self,d_Z:xp.ndarray)->xp.ndarray:
        """
        d_Z: (N, D_out)
        return: (N, f_n^{l}, X_H^{l}, X_W^{l}) or (N, D_out_prev) same as A
        """
    #--------tmp array, transfer to gpu--------------#  
        # self.A = to_gpu(self.A)
        # self.W = to_gpu(self.W)
        # self.b = to_gpu(self.b)

        # self.S_W = to_gpu(self.S_W)
        # self.V_W = to_gpu(self.V_W)
        # self.S_b = to_gpu(self.S_b)
        # self.V_b = to_gpu(self.V_b)
    #------------------------------------------------#

        d_W = self.A.T @ d_Z # (D_in, D_out)
        d_b = xp.sum(d_Z,axis=0,keepdims=True) # (1, D_out)
        
        # Gradient Clipping
        d_W = xp.clip(d_W, -5.0, 5.0)
        d_b = xp.clip(d_b, -5.0, 5.0)
        
        d_A = d_Z @ self.W.T  # (N, D_in)
        d_A = d_A.reshape(self.shape) # (N, C, H, W) or (N, D_out_prev)
        if self._Adam:
            self.V_W = self.Adam_beta1 * self.V_W + (1-self.Adam_beta1) * d_W # (D_in, D_out)
            self.S_W = self.Adam_beta2 * self.S_W + (1-self.Adam_beta2) * xp.power(d_W,2) # (D_in, D_out)
            self.W = self.W - self.learning_rate * self.V_W / (xp.sqrt(self.S_W) + self.epsilon) # (D_in, D_out)
            self.V_b = self.Adam_beta1 * self.V_b + (1-self.Adam_beta1) * d_b # (1, D_out)
            self.S_b = self.Adam_beta2 * self.S_b + (1-self.Adam_beta2) * xp.power(d_b,2) # (1, D_out)
            self.b = self.b - self.learning_rate * self.V_b / (xp.sqrt(self.S_b) + self.epsilon) # (1, D_out)
        else:
            self.W = self.W - self.learning_rate * d_W # (D_in, D_out)
            self.b = self.b - self.learning_rate * d_b # (1, D_out)

    #--------tmp array, transfer to cpu--------------#  
        # self.W = to_cpu(self.W)
        # self.b = to_cpu(self.b)
        # del self.A
        # cp.get_default_memory_pool().free_all_blocks()

        # self.S_W = to_cpu(self.S_W)
        # self.V_W = to_cpu(self.V_W)
        # self.S_b = to_cpu(self.S_b)
        # self.V_b = to_cpu(self.V_b)
    #------------------------------------------------#
        return d_A # (N, f_n^{l}, X_H^{l}, X_W^{l})


class Dropout(layer):
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward_prop(self, A:xp.ndarray, training:bool=True)->xp.ndarray:
        if training:
            # Inverted dropout: Scale by 1/(1-p) so expected sum remains same
            keep_prob = 1 - self.drop_rate
            self.mask = (xp.random.rand(*A.shape) < keep_prob) / keep_prob # *是解包运算符，把元组（2，2）解为2，2
            # / keep_prob缩放是为了保持数学期望值不变

            # self.mask = (xp.random.rand(*A.shape) < keep_prob) # *是解包运算符，把元组（2，2）解为2，2
            return A * self.mask
        else:
            return A
            
    def backward_prop(self, dZ:xp.ndarray)->xp.ndarray:
        return dZ * self.mask

    def get_config(self):
        return {
            'type': 'Dropout',
            'drop_rate': self.drop_rate
        }
    
    def set_config(self, config:dict):
        self.drop_rate = config['drop_rate']
    


class Sampling(layer):# 放弃Upsampling, 改用宽高维度分别进行采样
    def __init__(self, target_shape:tuple, mode:str='bilinear'):
        self.src_shape = None
        self.target_shape = target_shape
        self.mode = mode

        self.h_in_grid = None
        self.w_in_grid = None

        self.w_tl = None
        self.w_tr = None
        self.w_bl = None
        self.w_br = None

        self.top = None
        self.left = None
        self.bottom = None
        self.right = None


    def get_interp_positions(self,H_src, W_src, H_targ, W_targ, align_corners=True):
        """
        计算所有输出位置对应的输入插值位置
        :param H_src/W_src: 输入高/宽
        :param H_targ/W_targ: 输出高/宽
        :param align_corners: 映射规则
        :return: h_in_grid, w_in_grid（形状和输出一致，每个值是输入插值位置）
        """
        # 生成输出的所有坐标（0到H_targ-1，0到W_targ-1）
        h_out = xp.arange(H_targ)
        w_out = xp.arange(W_targ)
        h_out_grid, w_out_grid = xp.meshgrid(h_out, w_out, indexing='ij')  # (H_targ, W_targ)
        
        # 计算映射比例
        if align_corners:# 意思是只在四个元素之内插值，边界位置元素之外不插值 
            scale_h = (H_src - 1) / (H_targ - 1) if H_targ > 1 else 0 # 处理targ为0
            scale_w = (W_src - 1) / (W_targ - 1) if W_targ > 1 else 0
        else:
            scale_h = H_src / H_targ
            scale_w = W_src / W_targ
        
        # 计算每个输出位置对应的输入插值位置
        h_in_grid = h_out_grid * scale_h
        w_in_grid = w_out_grid * scale_w
        
        return h_in_grid, w_in_grid # (H_targ, W_targ)

    def forward_prop(self, A:xp.ndarray)->xp.ndarray:
        """
        A: (N, C, H, W)
        return: (N, C, H_targ, W_targ)
        """
        if len(A.shape)==4:
        
            self.src_shape = A.shape
            N,C,H_src,W_src=self.src_shape
            _,_,H_targ,W_targ=self.target_shape
            if self.mode == 'bilinear': # 双线性插值
                if self.h_in_grid is None or self.w_in_grid is None:
                    self.h_in_grid, self.w_in_grid = self.get_interp_positions(H_src, W_src, H_targ, W_targ)
                self.top = self.h_in_grid.astype(int) # (H_targ, W_targ)
                self.left = self.w_in_grid.astype(int) # (H_targ, W_targ)
                self.bottom = xp.minimum(self.h_in_grid+1, H_src-1).astype(int) # (H_targ, W_targ)
                self.right = xp.minimum(self.w_in_grid+1, W_src-1).astype(int) # (H_targ, W_targ)
                h_in_grid_flt = self.h_in_grid - self.top
                w_in_grid_flt = self.w_in_grid - self.left
                h_in_grid_counter_flt = 1 - h_in_grid_flt
                w_in_grid_counter_flt = 1 - w_in_grid_flt

                self.w_tl = h_in_grid_counter_flt * w_in_grid_counter_flt # (H_targ, W_targ)
                self.w_tr = h_in_grid_counter_flt * w_in_grid_flt # (H_targ, W_targ)
                self.w_bl = h_in_grid_flt * w_in_grid_counter_flt # (H_targ, W_targ)
                self.w_br = h_in_grid_flt * w_in_grid_flt # (H_targ, W_targ)

                out = xp.zeros((N, C, H_targ, W_targ))
                # 离某个点越近权重越大 # 这是一种高级索引写法
                out += A[:,:,self.top,self.left] * self.w_tl # topleft
                out += A[:,:,self.top,self.right] * self.w_tr # topright                
                out += A[:,:,self.bottom,self.left] * self.w_bl # bottomleft
                out += A[:,:,self.bottom,self.right] * self.w_br # bottomright
                return out
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        else:
            raise ValueError(f"Unsupported input shape: {A.shape}. Expected 4D input.")

    def backward_prop(self, d_Z: xp.ndarray) -> xp.ndarray:
        """
        更简洁的版本4实现
        d_Z: (N, C, H_targ, W_targ)
        return: (N, C, H_src, W_src)
        """
        N, C, H_targ, W_targ = d_Z.shape
        H_src, W_src = self.src_shape[2:]
        
        # 初始化梯度
        d_A = xp.zeros((N, C, H_src, W_src), dtype=d_Z.dtype)
        
        # 创建 batch 和 channel 的索引网格
        n_idx = xp.arange(N).reshape(N, 1, 1, 1)
        c_idx = xp.arange(C).reshape(1, C, 1, 1)
        
        # 四个角的配置
        corners = [
            (self.top, self.left, self.w_tl),
            (self.top, self.right, self.w_tr),
            (self.bottom, self.left, self.w_bl),
            (self.bottom, self.right, self.w_br)
        ]
        
        # 对每个角进行向量化累加
        for h_idx, w_idx, weight in corners:
            # 扩展坐标到 4D
            h_4d = h_idx[None, None, :, :]
            w_4d = w_idx[None, None, :, :]
            w_full = weight[None, None, :, :]
            
            # 使用广播创建完整的索引
            h_full = xp.broadcast_to(h_4d, (N, C, H_targ, W_targ))
            w_full = xp.broadcast_to(w_4d, (N, C, H_targ, W_targ))
            
            # 🔴 直接使用多维索引的 add.at
            xp.add.at(d_A, 
                    (xp.broadcast_to(n_idx, (N, C, H_targ, W_targ)),
                    xp.broadcast_to(c_idx, (N, C, H_targ, W_targ)),
                    h_full,
                    w_full),
                    d_Z * w_full)
        
        return d_A

    # def backward_prop(self, d_Z:xp.ndarray)->xp.ndarray:
    #     """
    #     版本2
    #     d_Z: (N, C, H_targ, W_targ)
    #     return: (N, C, H_src, W_src)
    #     """
    #     N,C,H_targ,W_targ = d_Z.shape
    #     H_src, W_src = self.src_shape[2:] # (H_targ, W_targ)

    #     # 展平向量以向量化加速add.at的计算

    #     print(d_Z.shape)
    #     print(self.w_tl.shape)
    #     # (N, C, H_targ, W_targ)
    #     val_tl = d_Z[:,:,self.top,self.left] * self.w_tl # topleft
    #     val_tr = d_Z[:,:,self.top,self.right] * self.w_tr # topright
    #     val_bl = d_Z[:,:,self.bottom,self.left] * self.w_bl # bottomleft
    #     val_br = d_Z[:,:,self.bottom,self.right] * self.w_br # bottomright
    #     val_tl = val_tl.reshape(N*C,-1)
    #     val_tr = val_tr.reshape(N*C,-1)
    #     val_bl = val_bl.reshape(N*C,-1)
    #     val_br = val_br.reshape(N*C,-1)

    #     # (1,H_targ*W_targ)
    #     idx_tl = (self.top * W_src + self.left).reshape(1,-1)
    #     idx_tr = (self.top * W_src + self.right).reshape(1,-1)
    #     idx_bl = (self.bottom * W_src + self.left).reshape(1,-1)
    #     idx_br = (self.bottom * W_src + self.right).reshape(1,-1)

    #     # (N*C,1)
    #     channel_offsets = xp.arange(N*C)[:, None] * H_src * W_src # (N*C, 1)

    #     # broadcast add
    #     idx_tl = channel_offsets + idx_tl
    #     idx_tr = channel_offsets + idx_tr
    #     idx_bl = channel_offsets + idx_bl
    #     idx_br = channel_offsets + idx_br

    #     d_A = xp.zeros((N*C,H_src*W_src))
    #     print(d_A.shape,idx_tl.shape,val_tl.shape)
    #     # print(xp.add.at(d_A, idx_tl, val_tl).shape)
    #     xp.add.at(d_A, idx_tl, val_tl)
    #     xp.add.at(d_A, idx_tr, val_tr)
    #     xp.add.at(d_A, idx_bl, val_bl)
    #     xp.add.at(d_A, idx_br, val_br)

    #     return d_A.reshape(self.src_shape)


    def get_config(self): # 可以不保存索引, 压缩大小
        return {
            'type': 'Sampling',
            'target_shape': self.target_shape,
            'mode': self.mode,
        }


    def set_config(self, config:dict):
        self.target_shape = config['target_shape']
        self.mode = config['mode']



# Residual Block
# 因为使用矩阵运算处理维度容易导致矩阵过大，内存无法处理这么大的矩阵，必须使用1*1卷积
# 1*1卷积要实现宽高维度放大的，可以使用转置卷积（但容易产生棋盘格伪影），现代的方法是插值法
class ResBlock(TrainableLayer):
    def __init__(self, Layers:list[layer], connected_layer:int = None,
                 src_shape:tuple=None, target_shape:tuple=None,
                learning_rate:float=0.001,
                _Adam:bool=False,Adam_beta1:float=0.9,Adam_beta2:float=0.999,epsilon:float=1e-8):
        TrainableLayer.__init__(self, learning_rate=learning_rate, _Adam=_Adam, Adam_beta1=Adam_beta1, Adam_beta2=Adam_beta2, epsilon=epsilon)
        self.Layers = Layers
        self.connected_layer = connected_layer if connected_layer is not None else len(Layers)-1

        self.src_shape = src_shape
        self.target_shape = target_shape # 在setweight中使用，防止参数不对应

        self.FC_proj = None # 用于FC层间残差的高效实现
        self.Sampling_proj = None
        self.Conv_proj = None
        
    # def safe_mul(self,A:xp.ndarray,B:xp.ndarray)->xp.ndarray:
    #     max_size = 100000
    #     if A.size > max_size or B.size > max_size:
    #         A_shape,B_shape=A.shape,B.shape
    #         middle=(A_shape[1]+2-1)//2
    #         A_1,A_2=A[:,0:middle],A[:,middle:]
    #         B_1,B_2=B[:,0:middle],B[:,middle:]
    #         C_11=self.safe_mul(A_1,B_1)
    #         C_12=self.safe_mul(A_1,B_2)
    #         C_21=self.safe_mul(A_2,B_1)
    #         C_22=self.safe_mul(A_2,B_2)
    #         C=xp.concatenate([xp.concatenate([C_11,C_21],axis=1),xp.concatenate([C_12,C_22],axis=1)],axis=0)
    #         return C
    #     else:
    #         return A @ B
    def forward_prop(self,X:xp.ndarray, training:bool=True)->xp.ndarray:
        out = X
        self.src_shape = X.shape
        src_size = 1

        # 跳过N维度
        for _ in self.src_shape[1:]:
            src_size *= _
        for i, layer in enumerate(self.Layers):
            try:
                out = layer.forward_prop(out, training=training)
            except TypeError:
                out = layer.forward_prop(out)

            if (i == self.connected_layer):
                self.target_shape = out.shape
                targ_size = 1
                for _ in self.target_shape[1:]:
                    targ_size *= _

                if len(self.src_shape) == 4 and len(self.target_shape) == 4:
                    if (self.src_shape == self.target_shape):
                        out += X
                    else:
                        tmp = X
                        if self.Sampling_proj is None:
                        # and (self.src_shape[2] < self.target_shape[2] or self.src_shape[3] < self.target_shape[3]):
                            self.Sampling_proj = Sampling(self.target_shape)
                        tmp = self.Sampling_proj.forward_prop(tmp) # (N, C, H_targ, W_targ)
                        if self.Conv_proj is None:
                            self.Conv_proj = Conv(filter_num = self.target_shape[1], filter_size = 1, filter_channel = self.src_shape[1], stride = 1, same_padding = True, _Adam = self._Adam, Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, epsilon = self.epsilon, learning_rate = self.learning_rate)
                        out += self.Conv_proj.forward_prop(tmp)
                elif len(self.src_shape) == 2 and len(self.target_shape) == 2: # FC -> FC
                    if (self.src_shape == self.target_shape):
                        out += X
                    else:
                        if self.FC_proj is None:
                            self.FC_proj = FC(output_size = self.target_shape[1],_Adam = self._Adam, Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, epsilon = self.epsilon, learning_rate = self.learning_rate)
                        out += self.FC_proj.forward_prop(X)
                elif len(self.src_shape) == 4 and len(self.target_shape) == 2: # Conv -> FC
                    if self.FC_proj is None:
                        self.FC_proj = FC(output_size = targ_size,_Adam = self._Adam, Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, epsilon = self.epsilon, learning_rate = self.learning_rate)
                    # out2 = self.FC_proj.forward_prop(X)
                    # print(out.shape,out2.shape,flush=True)
                    out += self.FC_proj.forward_prop(X)
                else:
                    raise ValueError(f"Unsupported target shape: {self.target_shape}")
        return out
    def backward_prop(self,d_Z:xp.ndarray)->xp.ndarray:
        connected_dZ = None
        # targ_size = 1
        # for _ in self.target_shape[1:]:
        #     targ_size *= _
        for i in range(len(self.Layers)-1,-1,-1):
            
            if i == self.connected_layer:
                if len(self.src_shape) == 4 and len(self.target_shape) == 4:
                    if self.src_shape == self.target_shape:
                        connected_dZ = d_Z # 1 * d_z
                    else:
                        connected_dZ = self.Conv_proj.backward_prop(d_Z)
                        # if (self.src_shape[2] < self.target_shape[2] or self.src_shape[3] < self.target_shape[3]):
                        connected_dZ = self.Sampling_proj.backward_prop(connected_dZ)
                        
                elif len(self.src_shape) == 2 and len(self.target_shape) == 2: # FC -> FC
                    if self.src_shape == self.target_shape:
                        connected_dZ = d_Z # 1 * d_z
                    else:   
                        connected_dZ = self.FC_proj.backward_prop(d_Z)
                elif len(self.src_shape) == 4 and len(self.target_shape) == 2: # Conv -> FC
                    connected_dZ = self.FC_proj.backward_prop(d_Z)
                else:
                    raise ValueError(f"Unsupported target shape: {self.target_shape}")           
            d_Z = self.Layers[i].backward_prop(d_Z)
        d_X = connected_dZ + d_Z
        return d_X


    def get_config(self):
        return {
            'type': 'ResBlock',
            'Layers': [layer.get_config() for layer in self.Layers],
            'connected_layer': self.connected_layer,
            # 'src_shape': self.src_shape,
            # 'target_shape': self.target_shape,   
            'learning_rate': self.learning_rate,
            '_Adam': self._Adam,
            'Adam_beta1': self.Adam_beta1,
            'Adam_beta2': self.Adam_beta2,
            'epsilon': self.epsilon,
            
            'src_shape': self.src_shape,
            'target_shape': self.target_shape,

        }
    def get_weights(self):
        return {
            'sampling_proj': self.Sampling_proj.get_weights() if self.Sampling_proj is not None else None,
            'conv_proj': self.Conv_proj.get_weights() if self.Conv_proj is not None else None,
            'fc_proj': self.FC_proj.get_weights() if self.FC_proj is not None else None,
            'Layers': [layer.get_weights() for layer in self.Layers]
        }
   
    def set_config(self, config:dict):
        self.Layers = [layer.set_config(layer_config) for layer_config in config['Layers']]
        self.connected_layer = config['connected_layer']
        # self.src_shape = config['src_shape']
        # self.target_shape = config['target_shape']
       
        self.learning_rate = config['learning_rate']
        self._Adam = config['_Adam']
        self.Adam_beta1 = config['Adam_beta1']
        self.Adam_beta2 = config['Adam_beta2']
        self.epsilon = config['epsilon']

        self.src_shape = config['src_shape']
        self.target_shape = config['target_shape']
        
    def set_weights(self, weights:dict):
        if weights.get('sampling_proj') is not None:
            self.Sampling_proj = Sampling(self.target_shape) # 需要先创建实例才能调用set_weights
            self.Sampling_proj.set_weights(weights['sampling_proj'])
        if weights.get('conv_proj') is not None:
            self.Conv_proj = Conv(filter_num = self.target_shape[1],
                                   filter_size = 1, filter_channel = self.src_shape[1], 
                                   stride = 1, same_padding = True, _Adam = self._Adam, 
                                   Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, 
                                   epsilon = self.epsilon, learning_rate = self.learning_rate)
            self.Conv_proj.set_weights(weights['conv_proj'])
        if weights.get('fc_proj') is not None:
            self.FC_proj = FC(output_size = self.target_shape[1],
                              _Adam = self._Adam, Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, 
                              epsilon = self.epsilon, learning_rate = self.learning_rate)
            self.FC_proj.set_weights(weights['fc_proj'])
        
        # 兼容旧版本可能直接传list的情况，或者字典里没有Layers的情况
        # layers_weights = weights.get('Layers')
        # if layers_weights is None and isinstance(weights, list):
        #     layers_weights = weights # 极其罕见的旧兼容
        layers_weights = weights['Layers']
        if layers_weights is not None:
            for layer, weight in zip(self.Layers, layers_weights):
                layer.set_weights(weight)
        else:
            print("Warning: No layer weights found in ResBlock weights. Skipping layer weights loading.")


    
class CNN:
    def __init__(self,layers:list[layer],
                learning_rate:float=0.001,
                _Adam:bool=False,beta1:float=0.9,beta2:float=0.999,epsilon:float=1e-8):
        self.layers = layers
        self.out = None

        self.len = len(layers)

        self.learning_rate = learning_rate
        self._Adam = _Adam
        self.Adam_beta1 = beta1
        self.Adam_beta2 = beta2
        self.epsilon = epsilon

        self.cost_history = []
        self.epoch_start = 0


    def save_model(self, filepath):
        """
        Save model architecture and weights to .npz file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        layer_configs = [layer.get_config() for layer in self.layers]

        params = {}
        # Save configs as object array (Pickle) instead of JSON string
        
        params['layer_configs'] = np.array(layer_configs, dtype=object)

        
        for i, layer in enumerate(self.layers):
            weights = layer.get_weights()
            if weights:
                for key, val in weights.items():
                    params[f'layer_{i}_weights_{key}'] = val
            

        # Save training state as object
        params['training_state'] = np.array({
            'learning_rate': self.learning_rate,
            'epoch_start': self.epoch_start,
            'cost_history': np.array(self.cost_history, dtype=object)
        }, dtype=object)

        np.savez_compressed(filepath, **params)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Load model architecture and weights from .npz file
        """
        
        if not os.path.exists(filepath):
            print(f"File {filepath} not found.")
            return None

        print(f"Loading model from {filepath}...")
        try:
            data = np.load(filepath, allow_pickle=True)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
        
        # self.layers = [] # remove self usage
        layers = []

        if 'layer_configs' not in data:
            print("Error: No layer configurations found in .npz file.")
            return None
            
        # Load layer configs (object array)
        try:
            layer_configs = data['layer_configs']
            if layer_configs.ndim == 0:
                layer_configs = layer_configs.item()
            elif layer_configs.dtype == object:
                layer_configs = layer_configs.tolist()
        except Exception:
            # Fallback for old models (if any)
            try:
                import json
                config_str = str(data['layer_configs'][0])
                layer_configs = json.loads(config_str)
            except ImportError:
                print("Error: Could not load layer_configs and json module not available.")
                return None
        
        # Pre-process data keys to avoid O(N*M) complexity
        # Group data by layer index and type (weights)
        layer_data = {}
        for key in data.files:
            # key format: layer_{i}_weights_{name} 
            if not key.startswith('layer_'):
                continue
                
            parts = key.split('_')
            if len(parts) < 4: continue # Not a weight key
            
            try:
                layer_idx = int(parts[1])
                data_type = parts[2] # 'weights'
                param_name = "_".join(parts[3:]) # handle names with underscores if any
                
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = {'weights': {}}
                
                if data_type in ['weights']:
                    val = data[key]
                    # Handle 0-D object arrays (common when saving lists/dicts with numpy)
                    if val.ndim == 0 and val.dtype == 'O':
                        val = val.item()
                    layer_data[layer_idx][data_type][param_name] = val
            except ValueError:
                continue

        def _create_layer(config):
            layer_type = config.pop('type')
            if layer_type == 'Conv':
                return Conv(**config)
            elif layer_type == 'BatchNorm':
                return BatchNorm(**config)
            elif layer_type == 'Activation':
                return Activation(**config)
            elif layer_type == 'Pooling':
                return Pooling(**config)
            elif layer_type == 'FC':
                return FC(**config)
            elif layer_type == 'Dropout':
                try:
                    return Dropout(**config)
                except NameError:
                    print("Warning: Dropout layer found in config but class not defined in CNN_util.py")
                    return None
            elif layer_type == 'ResBlock':
                sub_layers_configs = config.pop('Layers', [])
                sub_layers = []
                for sub_conf in sub_layers_configs:
                    l = _create_layer(sub_conf)
                    if l: sub_layers.append(l)
                return ResBlock(Layers=sub_layers, **config)
            else:
                print(f"Unknown layer type: {layer_type}")
                return None

        for i, config in enumerate(layer_configs):
            layer = _create_layer(config)
            if layer is None: continue
            
            # Efficiently restore weights  state
            if i in layer_data:
                if layer_data[i]['weights']:
                    layer.set_weights(layer_data[i]['weights'])
                
            
            layers.append(layer)
        
        # Initialize CNN
        cnn = CNN(layers=layers)

        # Restore global training state if available
        if 'training_state' in data:
            try:
                state_val = data['training_state'].item()
                # if state_val.ndim == 0:
                #     state = state_val.item()
                # else:
                #     # Handle case where it might be wrapped
                #     state = state_val.item() if state_val.size == 1 else state_val
                
                # # Check if it's old JSON format (string)
                # if isinstance(state, (str, xp.str_)):
                #      import json
                #      state = json.loads(str(state))
                # elif hasattr(state, 'item') and isinstance(state.item(), (str, xp.str_)):
                #      # Sometimes wrapped in array scalar of string
                #      import json
                #      state = json.loads(str(state.item()))

                # cnn.learning_rate = state_val.get('learning_rate', 0.001)
                # cnn.epoch_start = state_val.get('epoch_start', 0)
                # cnn.cost_history = state_val.get('cost_history', [])
                # print(f"Resuming training from epoch {cnn.epoch_start} with LR={cnn.learning_rate}")
 
                cnn.learning_rate = state_val['learning_rate']
                cnn.epoch_start = state_val['epoch_start']
                cnn.cost_history = state_val['cost_history']
                print(f"Resuming training from epoch {cnn.epoch_start} with LR={cnn.learning_rate}")
            except Exception as e:
                print(f"Warning: Could not load training state: {e}")
        
        # 4. Synchronize hyperparameters (learning rate, etc.) to all layers
        cnn.unified_hyperparam(learning_rate=cnn.learning_rate)

        print("Model loaded successfully.")
        return cnn

    def unified_hyperparam(self, learning_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        for layer in self.layers:
            layer.modified_hyperparam(learning_rate, _Adam, beta1, beta2, epsilon)
        
    def forward(self,X:xp.ndarray, training:bool=True)->xp.ndarray:
        self.out = X
        # import sys
        for i, layer in enumerate(self.layers):
            # layer_type = type(layer).__name__ 好像没有用
           
            try:
                self.out = layer.forward_prop(self.out, training=training)
            except TypeError:
                # 不支持 training 参数的层（Conv / FC / Pooling / Activation）
                self.out = layer.forward_prop(self.out)
          
        if training == False:
            return to_cpu(self.out)
        else:
            return self.out

    def calculate_cost(self, Y:xp.ndarray, transfer_onehot:bool=False)->float:
        """
        Calculate cross-entropy cost
        这里的输出是单类loss输出，实际上不能处理多类label，但是loss不影响训练，暂时做出让步
        """
        A_out = self.out
        # Handle different output shapes
        if len(A_out.shape) == 4:  # (N, C, H, W)
            A_out = A_out.reshape(A_out.shape[0], -1)
        elif len(A_out.shape) == 2 and A_out.shape[0] != Y.shape[0]:  # (N, D_out)
            if A_out.shape[0] != Y.shape[0] and A_out.shape[1] == Y.shape[0]:
                A_out = A_out.T
        A_out = to_cpu(A_out)
        # Clip to prevent log(0)
        A_out = np.clip(A_out, 1e-15, 1.0 - 1e-15)
        if transfer_onehot:
            # Calculate cost
            Y_flat = Y.flatten()
            Y_flat = to_cpu(Y_flat)
            cost = -np.mean(np.log(A_out[np.arange(Y_flat.shape[0]), Y_flat.astype(int)]))
        else:
            Y_flat = to_cpu(Y)
            cost = -np.mean(np.log(A_out[Y_flat.astype(int)]))
        # print(f"A_out shape: {A_out.shape}, Y_flat shape: {Y_flat.shape}")
        cost = -np.mean(np.log(A_out[np.arange(Y_flat.shape[0]), Y_flat.astype(int)]))
        return cost
    
    def calculate_cost_multilabel(self, Y:xp.ndarray)->float:
        # A_out, Y_flat: shape (batch_size, num_classes)
        # Avoid log(0) by clipping
        # A_out = to_cpu(self.out)
        # Y_tmp = to_cpu(Y)
        A_out = xp.clip(self.out, 1e-7, 1.0 - 1e-7)
        # Binary cross-entropy for each class
        # print("calculate cost",flush=True)
        cost = -xp.mean(Y * xp.log(A_out) + (1 - Y) * xp.log(1 - A_out))
        
        if xp.isnan(cost):
            print("Warning: Cost is NaN! Gradients might be exploding.")
            return float('nan')
            
        # print("get out",flush=True)
        return to_cpu(cost)
    
    def backward(self,dY:xp.ndarray):
        for i in range(self.len-1,-1,-1):
            # !!!!!!!
            # if dY is None: # Add check for None gradient
            #      continue
            dY = self.layers[i].backward_prop(dY)
    
    def train(self,X:xp.ndarray,Y:xp.ndarray,
              epochs:int=1000,batch_size:int=32,
              tolerance:float=1e-6,print_cost:bool=True, 
              loss:str='single', transfer_onehot:bool=False,save_path:str=None):
        
        # Keep data on CPU initially to save GPU memory and avoid "AlreadyMapped" errors
        # If input is already on GPU, it stays on GPU. If on CPU, it stays on CPU.
        # X = to_gpu(X) 
        # Y = to_gpu(Y)
        
        N = X.shape[0]
        # num_batches = N // batch_size + 1  # 这个计算有误，当batch_size整除N时，会多算一个batch
        num_batches = (N + batch_size - 1) // batch_size  # 修正计算方式
        
        print(f"Training with {N} samples, batch_size={batch_size}, num_batches={num_batches}")
        epoch_accumulated = 0

        # self.cost_history = self.cost_history.tolist()
        try:
            for i in range(epochs):

                # Shuffle data at the beginning of each epoch
                # Use appropriate random generator based on array type
                if isinstance(X, np.ndarray):
                    indices = np.random.permutation(N)
                else:
                    indices = xp.random.permutation(N)
                    
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]
                
                epoch_cost = 0
                
                # Process in batches
                for batch_idx in range(num_batches):
    #----------------文件外终止，软操作-----------------------------------------------------------------
                    if interrupt():
                        print("Training interrupted.")
                        self.epoch_start += epoch_accumulated
                        if save_path:
                            print(f"Saving model to {save_path}...")
                            self.save_model(save_path)
                        self.cost_history = np.array(self.cost_history, dtype=object)
                        return to_cpu(self.cost_history)

                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, N)
                    
                    # Slice first, then move to GPU
                    X_batch = to_gpu(X_shuffled[start_idx:end_idx])
                    Y_batch = to_gpu(Y_shuffled[start_idx:end_idx])
                    batch_size_actual = end_idx - start_idx
                    
                    # Clear the params
                    self.out = None

                    # Forward pass
                    self.forward(X_batch)
                    
                    batch_cost = 0
                    # Calculate cost for this batch
                    if loss == 'binary':
                        batch_cost = self.calculate_cost_multilabel(Y_batch)
                    else:
                        batch_cost = self.calculate_cost(Y_batch, transfer_onehot)
                    nnn = num_batches // 7
                    if print_cost and batch_idx % max(1, nnn) == 0:
                        print(f"Epoch {i + self.epoch_start}/{self.epoch_start + epochs} Batch {batch_idx}/{num_batches}  cost: {float(batch_cost):.6f}",flush=True)
                    epoch_cost += batch_cost
                    
                    # Calculate gradient for backward pass
                    # For softmax cross-entropy: dA = (A - Y_onehot) / batch_size
                    y_hat = self.out
                    
                    # Handle different output shapes - convert to (N, D_out) format, set num_classes
                    if len(y_hat.shape) == 4:  # (N, C, H, W)
                        y_hat = y_hat.reshape(y_hat.shape[0], -1)
                        num_classes = y_hat.shape[1]
                    elif len(y_hat.shape) == 2:
                        # FC layer outputs (D_out, N) format
                        # The last FC layer should output (num_classes, batch_size)
                        if y_hat.shape[1] == batch_size_actual:
                            # It's (D_out, N) = (num_classes, batch_size), transpose to (N, D_out)
                            num_classes = y_hat.shape[0]  # Get num_classes before transpose
                            y_hat = y_hat.T  # Now (N, D_out) = (batch_size, num_classes)
                        elif y_hat.shape[0] == batch_size_actual:
                            # Already (N, D_out) = (batch_size, num_classes)
                            num_classes = y_hat.shape[1]
                        else:
                            raise ValueError(f"Unexpected output shape: {y_hat.shape}")
                    
                    # Create one-hot encoding for Y_batch
                    if transfer_onehot:
                        Y_onehot = xp.zeros((batch_size_actual, num_classes)) # (N, D_out)
                        Y_onehot[xp.arange(batch_size_actual), Y_batch.flatten().astype(int)] = 1
    
                        # Backward pass
                        self.backward(Y_onehot)
                    else:
                        self.backward(Y_batch)
                    
                    cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory after each batch
                
                # Average cost for the epoch
                if num_batches > 0:
                    epoch_cost /= num_batches
                    # 将 CuPy 标量安全地转到 CPU，并存成 Python float，避免 np.array 直接接收 CuPy 对象
                    epoch_cost_cpu = float(to_cpu(epoch_cost))
                    # 始终把 cost_history 当作 Python list 使用，避免 numpy.ndarray 没有 append 的问题
                    if isinstance(self.cost_history, np.ndarray):
                        self.cost_history = self.cost_history.tolist()
                    self.cost_history.append(epoch_cost_cpu)
                    # 可选：如果后续需要 numpy 格式再单独转换，这里保持为 list 更安全
                    # 不适用学习率震荡更新， 依赖Adam优化器， 或者用learning rate decay
                    self.learning_rate *= 0.999
                    self.unified_hyperparam(learning_rate=self.learning_rate)

                    if print_cost:
                        print(f'Cost after epoch {i}: {epoch_cost:.6f}')


                # Save model at end of epoch
                if save_path:
                    self.save_model(save_path)

                if i > 0 and len(self.cost_history) >= 2 and abs(self.cost_history[-1] - self.cost_history[-2]) < tolerance:
                    print(f'Converged after {i} epochs')
                    break
                epoch_accumulated += 1

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            self.epoch_start += epoch_accumulated
        # finally:
            if save_path:
                print(f"Saving model to {save_path}...")
                self.save_model(save_path)
            
            self.cost_history = np.array(self.cost_history, dtype=object)
            return to_cpu(self.cost_history)
        
    def predict(self, X:np.ndarray, batch_size:int=32)->np.ndarray:
        """Make predictions on input data with batch processing to avoid memory issues"""
        N = X.shape[0]
        # X = to_gpu(X) # Keep on CPU
        # # 如果样本数小于等于 batch_size，直接处理
        
        # 分批处理：使用列表收集所有预测结果，最后一次性拼接
        all_predictions = []  # 列表收集每个 batch 的预测结果
        num_batches = (N + batch_size - 1) // batch_size # 向上取整
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, N)
            
            # Move only batch to GPU
            X_batch = to_gpu(X[start_idx:end_idx])
            batch_size_actual = end_idx - start_idx
            
            # Forward pass for this batch
            output = self.forward(X_batch, training=False)
            
            # Handle different output shapes - 与训练时的逻辑保持一致
            if len(output.shape) == 4:  # (N, C, H, W)
                output = output.reshape(output.shape[0], -1)
            elif len(output.shape) == 2:
                # 使用与训练时相同的逻辑判断是否需要转置
                if output.shape[1] == batch_size_actual:
                    # It's (D_out, N) = (num_classes, batch_size), transpose to (N, D_out)
                    output = output.T
                elif output.shape[0] == batch_size_actual:
                    # Already (N, D_out) = (batch_size, num_classes)
                    pass  # 不需要转置
                else:
                    # 如果都不匹配，尝试转置（兼容性处理）
                    if output.shape[0] != batch_size_actual and output.shape[1] == batch_size_actual:
                        output = output.T
            
            output = to_cpu(output)
            # Get predictions for this batch and add to list
            batch_predictions = np.argmax(output, axis=1).reshape(-1, 1)
            all_predictions.append(batch_predictions)
        
        # 一次性拼接所有 batch 的预测结果
        return np.concatenate(all_predictions, axis=0)

    def evaluate(self, X:np.ndarray, Y:np.ndarray, batch_size:int=32)->float:
        """Evaluate accuracy on test data with batch processing to avoid memory issues"""
        predictions = self.predict(X, batch_size=batch_size)
        accuracy = np.mean(predictions == Y)
        return accuracy


# # 批量show
# def show_images(images, labels, num_images=10, bias=0):
#     plt.figure(figsize=(10, 1))
#     for i in range(num_images):
#         plt.subplot(1, num_images, i + 1)
#         plt.imshow(images[i + bias].reshape(28, 28), cmap='gray')
#         plt.title(labels[i + bias][0])
#         plt.axis('off')
#     plt.show()





# debug用:检查层赋值和参数是否正确
def compare_layer_weights(layer1:dict, layer2:dict) -> bool:
    """
    Compare two layers' weights.
    """
    if (layer1 is None and layer2 is None):
        # print(f"Both layers have no weights.")
        return True
    for key1,key2 in zip(layer1.keys(), layer2.keys()):
        if key1 != key2:
            print(f"Weight keys differ: {key1} vs {key2}")
            return False
        if type(layer1[key1]) != type(layer2[key2]):
            print(f"Weight types differ for key '{key1}': {type(layer1[key1])} vs {type(layer2[key2])}")
            return False
        if layer1[key1] is None and layer2[key2] is None:
            continue
        if isinstance(layer1[key1], list) and isinstance(layer2[key2], list):
            if len(layer1[key1]) != len(layer2[key2]):
                print(f"Weight list lengths differ for key '{key1}': {len(layer1[key1])} vs {len(layer2[key2])}")
                return False
            for i, (w1, w2) in enumerate(zip(layer1[key1], layer2[key2])):
                if not compare_layer_weights(w1, w2):
                    print(f"Weight list items differ at index {i} for key '{key1}'")
                    return False
        elif isinstance(layer1[key1], np.ndarray) and isinstance(layer2[key2], dict):
            if not np.array_equal(layer1[key1], layer2[key2].get('weights')):
                print(f"Weight values differ for key '{key1}': {layer1[key1]} vs {layer2[key2].get('weights')}")
                return False
        elif isinstance(layer1[key1], dict) and isinstance(layer2[key2], dict):
            sublayer1=layer1[key1]
            sublayer2=layer2[key2]
            if not compare_layer_weights(sublayer1, sublayer2):
                print(f"Weight sub-layers differ for key '{key1}'")
                return False
        elif layer1[key1].all() != layer2[key2].all():
            print(f"Weight values differ for key '{key1}': {layer1[key1]} vs {layer2[key2]}")
            return False
    
    return True

def compare_layer_configs(layer1:dict, layer2:dict) -> bool:
    """
    Compare two layer configs.
    """
    if (layer1['type'] != layer2['type']):
        print(f"Layer types differ: {layer1['type']} vs {layer2['type']}")
        return False
    for key1,key2 in zip(layer1.keys(), layer2.keys()):

        if isinstance(layer1[key1], list) and isinstance(layer2[key2], list):
            if len(layer1[key1]) != len(layer2[key2]):
                print(f"Config list lengths differ for key '{key1}': {len(layer1[key1])} vs {len(layer2[key2])}")
                return False
            for i, (c1, c2) in enumerate(zip(layer1[key1], layer2[key2])):
                if not compare_layer_configs(c1, c2):
                    print(f"Config list items differ at index {i} for key '{key1}'")
                    return False
        elif layer1[key1] != layer2[key2]:
            print(f"Config values differ for key '{key1}': {layer1[key1]} vs {layer2[key2]}")
            return False

    return True