import numpy as np
import matplotlib.pyplot as plt
import os
import json
# import inspect
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

def to_cpu(x):
    """Move array to CPU (NumPy)"""
    if x is None: return None
    if xp == np: return x
    return cp.asnumpy(x)

def to_gpu(x):
    """Move array to GPU (CuPy)"""
    if x is None: return None
    if xp == np: return x
    return cp.asarray(x)
# ===========================================================

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
        X_paded = xp.pad(X, [(0,0), (0,0), (p_H1, p_H2), (p_W1, p_W2)], mode='constant')
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
    
    try:
        X_out = as_strided(X_paded,shape=new_shape,strides=new_strides) # (N,h_out,w_out,filter_c,filter_size,filter_size)
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
            # Note: For CuPy, basic slicing + addition is efficient
            X_padded[:, :, h_start:h_end:stride, w_start:w_end:stride] += val.transpose(0, 3, 1, 2)

    # Remove padding
    if p_H1 > 0 or p_H2 > 0 or p_W1 > 0 or p_W2 > 0:
        return X_padded[:, :, p_H1:p_H1+H, p_W1:p_W1+W]
    else:
        return X_padded


def calculate_dynamic_padding(input_size: int, filter_size: int, stride: int, 
                              same_padding: bool = False) -> tuple[int, int]:
    # 这里的计算是标量计算，不需要用cupy，保持原样即可
    if same_padding:
        total_p = (input_size - 1) * stride + filter_size - input_size
        
        if total_p < 0:
            return calculate_dynamic_padding(input_size, filter_size, stride, same_padding=False)
        
        if total_p % 2 == 0:
            return (total_p // 2, total_p // 2)
        else:
            return ((total_p + 1) // 2, total_p // 2)
    
    remainder = (input_size - filter_size) % stride
    if remainder == 0:
        total_p_min = 0
    else:
        total_p_min = stride - remainder
    
    total_p_required = max(0, filter_size - input_size)
    total_p = max(total_p_min, total_p_required)
    
    if (input_size + total_p - filter_size) % stride != 0:
        current_remainder = (input_size + total_p - filter_size) % stride
        total_p += stride - current_remainder
    
    h_out = (input_size + total_p - filter_size) // stride + 1
    if h_out <= 0:
        total_p += stride
        h_out = (input_size + total_p - filter_size) // stride + 1
    
    if total_p % 2 == 0:
        return (total_p // 2, total_p // 2)
    else:
        return ((total_p + 1) // 2, total_p // 2)


class layer:
    def forward_prop(self, X: xp.ndarray) -> xp.ndarray:
        pass
    def backward_prop(self, dY: xp.ndarray) -> xp.ndarray:
        pass
    def modified_hyperparam(self, learn_rate=None, _Adam=None, beta1=None, beta2=None, epsilon=None):
        pass 
    def get_config(self): return {}
    def get_weights(self): return {}
    def set_weights(self, weights): pass
    def get_optimizer_state(self): return {}
    def set_optimizer_state(self, state): pass

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

        self.padding = None
        self.same_padding = same_padding

        # initialize filters and bias on GPU/CPU
        self.F = xp.random.randn(filter_channel * filter_size * filter_size, filter_num) / xp.sqrt(filter_size * filter_size * filter_channel)
        self.bias = xp.zeros((1, filter_num))  # (1, f_n^{l})
        
        self.X_col = None
        self.col_shape = None

        self.S_F = xp.zeros_like(self.F)
        self.V_F = xp.zeros_like(self.F)
        self.S_bias = xp.zeros_like(self.bias)
        self.V_bias = xp.zeros_like(self.bias)
    
    def get_config(self):
        return {
            'type': 'Conv',
            'filter_num': int(self.filter_num),
            'filter_size': int(self.filter_size),
            'filter_channel': int(self.filter_channel),
            'stride': int(self.stride),
            '_Adam': bool(self._Adam),
            'Adam_beta1': float(self.Adam_beta1),
            'Adam_beta2': float(self.Adam_beta2),
            'epsilon': float(self.epsilon),
            'same_padding': bool(self.same_padding)
        }
    def get_weights(self):
        # Convert to CPU for saving
        return {
            'F': to_cpu(self.F),
            'bias': to_cpu(self.bias),
        }
    
    def get_optimizer_state(self):
        return {
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
        self._Adam = config['_Adam']
        self.Adam_beta1 = config['Adam_beta1']
        self.Adam_beta2 = config['Adam_beta2']
        self.epsilon = config['epsilon']
        self.same_padding = config['same_padding']

    def set_weights(self, weights:dict):
        # Move loaded weights to GPU
        self.F = to_gpu(weights['F'])
        self.bias = to_gpu(weights['bias'])

    def set_optimizer_state(self, optimizer_state:dict):
        self.S_F = to_gpu(optimizer_state['S_F'])
        self.V_F = to_gpu(optimizer_state['V_F'])
        self.S_bias = to_gpu(optimizer_state['S_bias'])
        self.V_bias = to_gpu(optimizer_state['V_bias'])
        
    def forward_prop(self, X:xp.ndarray)->xp.ndarray:
        ''' 
        X: (N, X_C^{l-1}, X_H^{l-1}, X_W^{l-1})
        F: (f_c^{l}*f_size^{l}*f_size^{l})
        Returns: (N, f_n^{l}, Z_H^{l}, Z_W^{l})
        '''
        N,C,H,W = X.shape
        if self.padding is None:
            p_H1, p_H2 = calculate_dynamic_padding(H, self.filter_size, self.stride, self.same_padding)
            p_W1, p_W2 = calculate_dynamic_padding(W, self.filter_size, self.stride, self.same_padding)
            self.padding = (p_H1, p_H2, p_W1, p_W2)
        
        self.X_col = img2col(X,filter_size=self.filter_size,stride=self.stride,padding=self.padding)
        self.col_shape = self.X_col.shape 
        self.X_col = self.X_col.reshape(self.col_shape[0]*self.col_shape[1]*self.col_shape[2],-1) 
        
        Z = self.X_col @ self.F + self.bias
        Z = Z.reshape(self.col_shape[0],self.col_shape[1],self.col_shape[2],self.filter_num)
        Z = Z.transpose(0,3,1,2)
        return Z

    def backward_prop(self, d_Z:xp.ndarray)->xp.ndarray:
        '''
        d_Z: (N, f_n^{l}, Z_H^{l}, Z_W^{l}) (统一维度设计)
        F: (X_C^{l-1} * filter_size^{l} * filter_size^{l}, f_n^{l})
        Returns: (N, X_C^{l-1}, X_H^{l-1}, X_W^{l-1})
        '''
        d_Z = d_Z.transpose(0,2,3,1).reshape(-1,self.filter_num) 
        
        d_F = self.X_col.T @ d_Z 
        d_bias = d_Z.sum(axis=0,keepdims=True).reshape(1,-1) 
        d_X_col = d_Z @ self.F.T 
        
        if self._Adam:
            self.V_F = self.Adam_beta1 * self.V_F + (1-self.Adam_beta1) * d_F
            self.S_F = self.Adam_beta2 * self.S_F + (1-self.Adam_beta2) * xp.power(d_F,2)
            self.F = self.F - self.learning_rate * self.V_F / (xp.sqrt(self.S_F) + self.epsilon)
            
            self.V_bias = self.Adam_beta1 * self.V_bias + (1-self.Adam_beta1) * d_bias
            self.S_bias = self.Adam_beta2 * self.S_bias + (1-self.Adam_beta2) * xp.power(d_bias,2)
            self.bias = self.bias - self.learning_rate * self.V_bias / (xp.sqrt(self.S_bias) + self.epsilon)
        else:
            self.F = self.F - self.learning_rate * d_F 
            self.bias = self.bias - self.learning_rate * d_bias 
            
        d_X_col = d_X_col.reshape(self.col_shape)
        d_X = col2img(d_X_col,stride=self.stride, padding=self.padding)
        return d_X


class BatchNorm(TrainableLayer):
    def __init__(self,
            _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
            momentum=0.8,
            learning_rate=0.001):
        TrainableLayer.__init__(self, learning_rate=learning_rate, _Adam=_Adam, Adam_beta1=Adam_beta1, Adam_beta2=Adam_beta2, epsilon=epsilon)

        self.input_shape = None
        self.mu = None
        self.sigma = None
        self.y_hat = None
        self.y_tilde = None

        self.gamma = None
        self.beta = None

        self.running_mean = None
        self.running_var = None
        self.momentum = momentum

        self.S_gamma = None
        self.V_gamma = None
        self.S_beta = None
        self.V_beta = None
        
    def get_config(self):
        return {
            'type': 'BatchNorm',
            '_Adam': bool(self._Adam),
            'Adam_beta1': float(self.Adam_beta1),
            'Adam_beta2': float(self.Adam_beta2),
            'epsilon': float(self.epsilon),
            'momentum': float(self.momentum),
            'learning_rate': float(self.learning_rate),
        }
    def get_weights(self):
        return {
            'gamma': to_cpu(self.gamma),
            'beta': to_cpu(self.beta),
            'running_mean': to_cpu(self.running_mean),
            'running_var': to_cpu(self.running_var),
        }
    def get_optimizer_state(self):
        return {
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
        self.gamma = to_gpu(weights.get('gamma'))
        self.beta = to_gpu(weights.get('beta'))
        self.running_mean = to_gpu(weights.get('running_mean'))
        self.running_var = to_gpu(weights.get('running_var'))
        
        # Init optimizer states if weights are loaded but state isn't yet
        if self.gamma is not None and self.S_gamma is None:
             self.S_gamma = xp.zeros_like(self.gamma)
             self.V_gamma = xp.zeros_like(self.gamma)
             self.S_beta = xp.zeros_like(self.beta)
             self.V_beta = xp.zeros_like(self.beta)

    def set_optimizer_state(self, optimizer_state:dict):
        self.S_gamma = to_gpu(optimizer_state['S_gamma'])
        self.V_gamma = to_gpu(optimizer_state['V_gamma'])
        self.S_beta = to_gpu(optimizer_state['S_beta'])
        self.V_beta = to_gpu(optimizer_state['V_beta'])

    def forward_prop(self, Z:xp.ndarray, training:bool=True)->xp.ndarray:
        if self.input_shape is None:
            self.input_shape = tuple([1]+list(Z.shape[1:]))
            
        if self.gamma is None:
            self.gamma = xp.ones(self.input_shape)
            self.S_gamma = xp.zeros_like(self.gamma)
            self.V_gamma = xp.zeros_like(self.gamma)
        if self.beta is None:
            self.beta = xp.zeros(self.input_shape)
            self.S_beta = xp.zeros_like(self.beta)
            self.V_beta = xp.zeros_like(self.beta)

        if self.running_mean is None:
            self.running_mean = xp.zeros(self.input_shape)
        if self.running_var is None:
            self.running_var = xp.ones(self.input_shape)

        if training:
            self.mu = Z.mean(axis=0, keepdims=True)
            self.sigma = xp.var(Z, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sigma

            self.y_hat = (Z - self.mu) / xp.sqrt(self.sigma + self.epsilon)
        else:
            mu = self.running_mean
            sigma = self.running_var
            self.y_hat = (Z - mu) / xp.sqrt(sigma + self.epsilon)

        y_tilde = self.gamma * self.y_hat + self.beta
        return y_tilde
    
    def backward_prop(self, d_y_tilde:xp.ndarray)->xp.ndarray:
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
            self.gamma = self.gamma - self.learning_rate * d_gamma
            self.beta = self.beta - self.learning_rate * d_beta
        d_Z = (self.gamma / xp.sqrt(self.sigma + self.epsilon)) * (d_y_tilde - B - D)
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
        X = xp.clip(X, -100, 100)
        return xp.where(X >= 0,
                       1 / (1 + xp.exp(-X)),
                       xp.exp(X) / (1 + xp.exp(X)))

    def forward_prop(self, Z:xp.ndarray)->xp.ndarray:
        """
        Z: (N, ...)
        return: (N, ...) same as Z
        """
        self.Z = Z
        if self.activation == 'relu':
            self.Map = Z > 0
            return Z * self.Map
        elif self.activation == 'sigmoid':
            return self.sigmoid(Z)
        elif self.activation == 'softmax':
            if len(Z.shape) != 2: 
                raise ValueError('FC layer softmax only support 2D input')
            exp_X = xp.exp(Z - xp.max(Z, axis=1, keepdims=True)) 
            self.Indice = exp_X / xp.sum(exp_X, axis=1, keepdims=True)
            return self.Indice
        else:
            raise ValueError('activation must be relu or sigmoid or softmax')
    def backward_prop(self, d_A:xp.ndarray)->xp.ndarray:
        """
        d_A: (N, ...) same as Z
        return: (N, ...) same as Z
        """
        if self.activation == 'relu':
            d_Z = d_A * self.Map
        elif self.activation == 'sigmoid':
            sigmoid_z = self.sigmoid(self.Z)
            d_Z = sigmoid_z * (1 - sigmoid_z) * d_A
        elif self.activation == 'softmax':
            d_Z = (self.Indice - d_A) / self.Z.shape[0] 
        else:
            raise ValueError('activation must be relu or sigmoid')
        return d_Z


class Pooling(layer):
    def __init__(self,stride=1,
             pool_size=2,pool_type='max', same_padding:bool=False):
        self.padding = None
        self.stride = stride
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.same_padding = same_padding

        self.onehot = None
        self.A_col = None

    def get_config(self):
        return {
            'type': 'Pooling',
            'stride': int(self.stride),
            'pool_size': int(self.pool_size),
            'pool_type': self.pool_type,
            'same_padding': bool(self.same_padding)
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
        self.input_shape = (N, f_n, H, W)
        if self.padding is None:
            p_H1, p_H2 = calculate_dynamic_padding(H, self.pool_size, self.stride, self.same_padding)
            p_W1, p_W2 = calculate_dynamic_padding(W, self.pool_size, self.stride, self.same_padding)
            self.padding = (p_H1, p_H2, p_W1, p_W2)
        
        A = A.reshape(N*f_n, 1, H, W)  
        A_col = img2col(A, filter_size=self.pool_size, stride=self.stride, padding=self.padding) 
        self.A_col_shape = A_col.shape
        
        A_col_flat = A_col.reshape(self.A_col_shape[0]*self.A_col_shape[1]*self.A_col_shape[2], -1) 
        
        if self.pool_type == 'max':
            X_flat = A_col_flat.max(axis=1) 
            Indices = A_col_flat.argmax(axis=1) 
            self.onehot = xp.zeros(A_col_flat.shape) 
            self.onehot[xp.arange(Indices.shape[0]), Indices] = 1 
            
        elif self.pool_type == 'avg':
            X_flat = A_col_flat.mean(axis=1)
            self.onehot = None
        else:
            raise ValueError('pool_type must be max or avg')
        
        h_out, w_out = self.A_col_shape[1], self.A_col_shape[2]
        X = X_flat.reshape(N, f_n, h_out, w_out)
        return X 

    def backward_prop(self,d_X:xp.ndarray)->xp.ndarray:
        """
        d_X: (N, f_n^{l}, h_out, w_out)(统一维度设计)
        Returns: (N, f_n^{l}, H, W)
        """
        N, f_n, h_out, w_out = d_X.shape
        d_X_flat = d_X.reshape(-1,1)
        
        pool_window_size = self.pool_size * self.pool_size
        weight = xp.ones((d_X_flat.shape[0],pool_window_size)) / 4.0

        d_A = None
        if self.pool_type == 'max' and self.onehot is not None:
            d_A_col_flat = self.onehot * d_X_flat 
        else:
            d_A_col_flat = weight * d_X_flat 
        d_A_col = d_A_col_flat.reshape(N*f_n, h_out, w_out, 1, self.pool_size, self.pool_size)
        d_A = col2img(d_A_col,stride=self.stride, padding=self.padding) 
        
        return d_A.reshape(N, f_n, d_A.shape[2], d_A.shape[3]) 


class FC(TrainableLayer): 
    def __init__(self,output_size,
                _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
                learning_rate=0.001):
        TrainableLayer.__init__(self, learning_rate=learning_rate, _Adam=_Adam, Adam_beta1=Adam_beta1, Adam_beta2=Adam_beta2, epsilon=epsilon)
        self.input_size = None 
        self.output_size = output_size

        self.A = None
        self.shape = None

        self.W = None
        self.b = xp.zeros((1, output_size))
        
        self.S_W = None 
        self.V_W = None
        self.S_b = xp.zeros_like(self.b)
        self.V_b = xp.zeros_like(self.b)

    def get_config(self):
        return {
            'type': 'FC',
            'output_size': int(self.output_size),
        }
    def get_weights(self):
        return {
            'W': to_cpu(self.W),
            'b': to_cpu(self.b),
        }
    def get_optimizer_state(self):
        return {
            'S_W': to_cpu(self.S_W),
            'V_W': to_cpu(self.V_W),
            'S_b': to_cpu(self.S_b),
            'V_b': to_cpu(self.V_b),
        }
    def set_config(self, config:dict):
        self.output_size = config['output_size']
    def set_weights(self, weights:dict):
        self.W = to_gpu(weights['W'])
        self.b = to_gpu(weights['b'])
    def set_optimizer_state(self, optimizer_state:dict):
        self.S_W = to_gpu(optimizer_state['S_W'])
        self.V_W = to_gpu(optimizer_state['V_W'])
        self.S_b = to_gpu(optimizer_state['S_b'])
        self.V_b = to_gpu(optimizer_state['V_b'])

    def forward_prop(self,A:xp.ndarray)->xp.ndarray:
        """
        A: Can be (N, f_n^{l}, X_H^{l}, X_W^{l}) from conv/pool layers or (N, D_out_prev) from FC layers
        W: (D_in, D_out)
        return Z: (N, D_out)
        """
        self.shape = A.shape 
        if len(A.shape) == 4: 
            self.A = A.reshape(A.shape[0], -1) 
        elif len(A.shape) == 2: 
            self.A = A
        else:
            raise ValueError(f"Unsupported input shape: {A.shape}. Expected 2D or 4D input.")

        if self.W is None:
            self.input_size = self.A.shape[1]
            self.W = xp.random.randn(self.input_size,self.output_size) * xp.sqrt(2/self.input_size)
            self.S_W = xp.zeros_like(self.W)
            self.V_W = xp.zeros_like(self.W)    

        Z = self.A @ self.W + self.b 
        return Z
    def backward_prop(self,d_Z:xp.ndarray)->xp.ndarray:
        """
        d_Z: (N, D_out)
        return: (N, f_n^{l}, X_H^{l}, X_W^{l}) or (N, D_out_prev) same as A
        """
        d_W = self.A.T @ d_Z 
        d_b = xp.sum(d_Z,axis=0,keepdims=True) 
        d_A = d_Z @ self.W.T  
        d_A = d_A.reshape(self.shape) 
        if self._Adam:
            self.V_W = self.Adam_beta1 * self.V_W + (1-self.Adam_beta1) * d_W 
            self.S_W = self.Adam_beta2 * self.S_W + (1-self.Adam_beta2) * xp.power(d_W,2) 
            self.W = self.W - self.learning_rate * self.V_W / (xp.sqrt(self.S_W) + self.epsilon) 
            self.V_b = self.Adam_beta1 * self.V_b + (1-self.Adam_beta1) * d_b 
            self.S_b = self.Adam_beta2 * self.S_b + (1-self.Adam_beta2) * xp.power(d_b,2) 
            self.b = self.b - self.learning_rate * self.V_b / (xp.sqrt(self.S_b) + self.epsilon) 
        else:
            self.W = self.W - self.learning_rate * d_W 
            self.b = self.b - self.learning_rate * d_b 
        return d_A 


class Dropout(layer):
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward_prop(self, A:xp.ndarray, training:bool=True)->xp.ndarray:
        if training:
            keep_prob = 1 - self.drop_rate
            self.mask = (xp.random.rand(*A.shape) < keep_prob) / keep_prob 
            return A * self.mask
        else:
            return A
            
    def backward_prop(self, dZ:xp.ndarray)->xp.ndarray:
        return dZ * self.mask

    def get_config(self):
        return {
            'type': 'Dropout',
            'drop_rate': float(self.drop_rate)
        }
    def set_config(self, config:dict):
        self.drop_rate = config['drop_rate']


class UpSampling(layer):
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

        self.idx_tl = None
        self.idx_tr = None
        self.idx_bl = None
        self.idx_br = None

    def get_interp_positions(self,H_src, W_src, H_targ, W_targ, align_corners=True):
        h_out = xp.arange(H_targ)
        w_out = xp.arange(W_targ)
        h_out_grid, w_out_grid = xp.meshgrid(h_out, w_out, indexing='ij') 
        
        if align_corners:
            scale_h = (H_src - 1) / (H_targ - 1) if H_targ > 1 else 0
            scale_w = (W_src - 1) / (W_targ - 1) if W_targ > 1 else 0
        else:
            scale_h = H_src / H_targ
            scale_w = W_src / W_targ
        
        h_in_grid = h_out_grid * scale_h
        w_in_grid = w_out_grid * scale_w
        return h_in_grid, w_in_grid 

    def forward_prop(self, A:xp.ndarray)->xp.ndarray:
        if len(A.shape)==4:
            self.src_shape = A.shape
            N,C,H_src,W_src=self.src_shape
            H_targ,W_targ=self.target_shape
            if self.mode == 'bilinear':
                if self.h_in_grid is None or self.w_in_grid is None:
                    self.h_in_grid, self.w_in_grid = self.get_interp_positions(H_src, W_src, H_targ, W_targ)
                
                h_in_grid_int = self.h_in_grid.astype(int)
                w_in_grid_int = self.w_in_grid.astype(int)
                h_in_grid_flt = self.h_in_grid - h_in_grid_int
                w_in_grid_flt = self.w_in_grid - w_in_grid_int
                h_in_grid_counter_flt = 1 - h_in_grid_flt
                w_in_grid_counter_flt = 1 - w_in_grid_flt
                
                out = xp.zeros((N, C, H_targ, W_targ))
                out += A[:,:,h_in_grid_int,w_in_grid_int] * h_in_grid_counter_flt * w_in_grid_counter_flt
                out += A[:,:,h_in_grid_int+1,w_in_grid_int] * h_in_grid_flt * w_in_grid_counter_flt
                out += A[:,:,h_in_grid_int,w_in_grid_int+1] * h_in_grid_counter_flt * w_in_grid_flt
                out += A[:,:,h_in_grid_int+1,w_in_grid_int+1] * h_in_grid_flt * w_in_grid_flt
                return out
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        else:
            raise ValueError(f"Unsupported input shape: {A.shape}. Expected 4D input.")
            
    def backward_prop(self, d_Z:xp.ndarray)->xp.ndarray:
        N, C, H_targ, W_targ = d_Z.shape
        H_src, W_src = self.src_shape[2], self.src_shape[3]
        
        num_pixels_out = H_targ * W_targ
        num_pixels_in = H_src * W_src
        num_channels = N * C        
        if (self.w_tl==None):
            h_floor_flat = xp.floor(self.h_in_grid).astype(int).reshape(-1,1) 
            w_floor_flat = xp.floor(self.w_in_grid).astype(int).reshape(-1,1) 
            
            h_ceil_flat = xp.minimum(h_floor_flat + 1, H_src - 1)
            w_ceil_flat = xp.minimum(w_floor_flat + 1, W_src - 1)
            
            weight_h_flat = self.h_in_grid.reshape(-1,1) - h_floor_flat
            weight_w_flat = self.w_in_grid.reshape(-1,1) - w_floor_flat

            self.w_tl = (1 - weight_h_flat) * (1 - weight_w_flat) 
            self.w_tr = weight_h_flat * (1 - weight_w_flat) 
            self.w_bl = (1 - weight_h_flat) * weight_w_flat 
            self.w_br = weight_h_flat * weight_w_flat 

            offset_tl = h_floor_flat * W_src + w_floor_flat 
            offset_tr = h_floor_flat * W_src + w_ceil_flat 
            offset_bl = h_ceil_flat * W_src + w_floor_flat 
            offset_br = h_ceil_flat * W_src + w_ceil_flat 

            channel_offsets = xp.arange(num_channels)[:, None] * num_pixels_in 

            self.idx_tl = (channel_offsets + offset_tl).flatten() 
            self.idx_tr = (channel_offsets + offset_tr).flatten() 
            self.idx_bl = (channel_offsets + offset_bl).flatten() 
            self.idx_br = (channel_offsets + offset_br).flatten() 
 
        d_A_flat = xp.zeros(num_channels * num_pixels_in, dtype=d_Z.dtype) 
        d_Z_flat = d_Z.reshape(num_channels, num_pixels_out) 
        
        val_tl = (d_Z_flat * self.w_tl).flatten() 
        val_tr = (d_Z_flat * self.w_tr).flatten() 
        val_bl = (d_Z_flat * self.w_bl).flatten() 
        val_br = (d_Z_flat * self.w_br).flatten() 
        
        # 使用 cupy.add.at 或 cupyx.scatter_add
        xp.add.at(d_A_flat, self.idx_tl, val_tl)
        xp.add.at(d_A_flat, self.idx_tr, val_tr)
        xp.add.at(d_A_flat, self.idx_bl, val_bl)
        xp.add.at(d_A_flat, self.idx_br, val_br)
        
        d_A = d_A_flat.reshape(N, C, H_src, W_src)
        return d_A

    def get_config(self): 
        return {
            'type': 'UpSampling',
            'target_shape': self.target_shape,
            'mode': self.mode,
        }
    def set_config(self, config:dict):
        self.target_shape = config['target_shape']
        self.mode = config['mode']


class ResBlock(TrainableLayer):
    def __init__(self, Layers:list[layer], connected_layer:int = None,
                learning_rate:float=0.001,
                _Adam:bool=False,Adam_beta1:float=0.9,Adam_beta2:float=0.999,epsilon:float=1e-8):
        TrainableLayer.__init__(self, learning_rate=learning_rate, _Adam=_Adam, Adam_beta1=Adam_beta1, Adam_beta2=Adam_beta2, epsilon=epsilon)
        self.Layers = Layers
        self.connected_layer = connected_layer if connected_layer is not None else len(Layers)-1
        
        # self.src_shape = tuple(src_shape) if src_shape is not None else None
        # self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.src_shape = None
        self.target_shape = None

        self.FC_proj = None 
        self.UpSampling_proj = None
        self.Conv_proj = None
        
    def forward_prop(self,X:xp.ndarray, training:bool=True)->xp.ndarray:
        out = X
        self.src_shape = X.shape
        src_size = 1

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
                        if self.UpSampling_proj is None and \
                            (self.src_shape[2] < self.target_shape[2] or self.src_shape[3] < self.target_shape[3]):
                            self.UpSampling_proj = UpSampling(self.target_shape)
                            tmp = self.UpSampling_proj.forward_prop(tmp) 
                        if self.Conv_proj is None:
                            self.Conv_proj = Conv(filter_num = self.target_shape[1], filter_size = 1, filter_channel = self.src_shape[1], stride = 1, same_padding = True, _Adam = self._Adam, Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, epsilon = self.epsilon, learning_rate = self.learning_rate)
                        out += self.Conv_proj.forward_prop(tmp)
                elif len(self.src_shape) == 2 and len(self.target_shape) == 2: 
                    if (self.src_shape == self.target_shape):
                        out += X
                    else:
                        if self.FC_proj is None:
                            self.FC_proj = FC(output_size = self.target_shape[1],_Adam = self._Adam, Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, epsilon = self.epsilon, learning_rate = self.learning_rate)
                        out += self.FC_proj.forward_prop(out)
                elif len(self.src_shape) == 4 and len(self.target_shape) == 2: 
                    if self.FC_proj is None:
                        self.FC_proj = FC(output_size = self.target_shape[1],_Adam = self._Adam, Adam_beta1 = self.Adam_beta1, Adam_beta2 = self.Adam_beta2, epsilon = self.epsilon, learning_rate = self.learning_rate)
                    out += self.FC_proj.forward_prop(out)
                else:
                    raise ValueError(f"Unsupported target shape: {self.target_shape}")
        return out
    def backward_prop(self,d_Z:xp.ndarray)->xp.ndarray:
        connected_dZ = None
        targ_size = 1
        for _ in self.target_shape[1:]:
            targ_size *= _
        for i in range(len(self.Layers)-1,-1,-1):
            d_Z = self.Layers[i].backward_prop(d_Z)
            if i == self.connected_layer:
                if len(self.src_shape) == 4 and len(self.target_shape) == 4:
                    if self.src_shape == self.target_shape:
                        connected_dZ = d_Z 
                    else:
                        tmp = d_Z
                        if (self.src_shape[2] < self.target_shape[2] or self.src_shape[3] < self.target_shape[3]):
                            tmp = self.UpSampling_proj.backward_prop(tmp)
                        connected_dZ = self.Conv_proj.backward_prop(tmp)
                elif len(self.src_shape) == 2 and len(self.target_shape) == 2:
                    if self.src_shape == self.target_shape:
                        connected_dZ = d_Z
                    else:   
                        connected_dZ = self.FC_proj.backward_prop(d_Z)
                elif len(self.src_shape) == 4 and len(self.target_shape) == 2:
                    connected_dZ = self.FC_proj.backward_prop(d_Z)
                else:
                    raise ValueError(f"Unsupported target shape: {self.target_shape}")           
            
        d_X = connected_dZ + d_Z
        return d_X


    def get_config(self):
        return {
            'type': 'ResBlock',
            'Layers': [layer.get_config() for layer in self.Layers],
            'connected_layer': int(self.connected_layer),
            # 'src_shape': self.src_shape,
            # 'target_shape': self.target_shape,   
        }
    def get_weights(self):
        return {
            'up_sampling_proj': self.UpSampling_proj.get_weights() if self.UpSampling_proj is not None else None,
            'conv_proj': self.Conv_proj.get_weights() if self.Conv_proj is not None else None,
            'fc_proj': self.FC_proj.get_weights() if self.FC_proj is not None else None,
            'Layers': [layer.get_weights() for layer in self.Layers]
        }
    def get_optimizer_state(self):
        return {
            'up_sampling_proj': self.UpSampling_proj.get_optimizer_state() if self.UpSampling_proj is not None else None,
            'conv_proj': self.Conv_proj.get_optimizer_state() if self.Conv_proj is not None else None,
            'fc_proj': self.FC_proj.get_optimizer_state() if self.FC_proj is not None else None,
            'Layers': [layer.get_optimizer_state() for layer in self.Layers],

            'learning_rate': float(self.learning_rate),
            '_Adam': bool(self._Adam),
            'Adam_beta1': float(self.Adam_beta1),
            'Adam_beta2': float(self.Adam_beta2),
            'epsilon': float(self.epsilon)
        }
    def set_config(self, config:dict):
        # Recursively recreate layers is complex here because we need layer instances.
        # This part assumes config loading is handled by CNN.load_model properly
        self.Layers = [layer.set_config(layer_config) for layer_config in config['Layers']]
        self.connected_layer = config['connected_layer']
        # self.src_shape = tuple(config['src_shape']) if config['src_shape'] else None
        # self.target_shape = tuple(config['target_shape']) if config['target_shape'] else None

        
    def set_weights(self, weights:dict):
        if weights.get('up_sampling_proj') is not None:
            if self.UpSampling_proj is None: 
                # Should be initialized by forward pass or config, if None try to infer?
                # For loading, layers should be init by config.
                pass 
            else:
                self.UpSampling_proj.set_weights(weights['up_sampling_proj'])
        if weights.get('conv_proj') is not None and self.Conv_proj is not None:
            self.Conv_proj.set_weights(weights['conv_proj'])
        if weights.get('fc_proj') is not None and self.FC_proj is not None:
            self.FC_proj.set_weights(weights['fc_proj'])
            
        layers_weights = weights.get('Layers')
        if layers_weights is None and isinstance(weights, list):
             layers_weights = weights

        if layers_weights is not None:
            for layer, weight in zip(self.Layers, layers_weights):
                layer.set_weights(weight)

    def set_optimizer_state(self, optimizer_states:dict):
        self.learning_rate = optimizer_states.get('learning_rate', self.learning_rate)
        self._Adam = optimizer_states.get('_Adam', self._Adam)
        self.Adam_beta1 = optimizer_states.get('Adam_beta1', self.Adam_beta1)
        self.Adam_beta2 = optimizer_states.get('Adam_beta2', self.Adam_beta2)
        self.epsilon = optimizer_states.get('epsilon', self.epsilon)

        if optimizer_states.get('up_sampling_proj') is not None and self.UpSampling_proj is not None:
            self.UpSampling_proj.set_optimizer_state(optimizer_states['up_sampling_proj'])
        if optimizer_states.get('conv_proj') is not None and self.Conv_proj is not None:
            self.Conv_proj.set_optimizer_state(optimizer_states['conv_proj'])
        if optimizer_states.get('fc_proj') is not None and self.FC_proj is not None:
            self.FC_proj.set_optimizer_state(optimizer_states['fc_proj'])
            
        if 'Layers' in optimizer_states:
            for layer, state in zip(self.Layers, optimizer_states['Layers']):
                layer.set_optimizer_state(state)

    
class CNN:
    def __init__(self,layers:list[layer],
                learning_rate:float=0.001,
                _Adam:bool=False,beta1:float=0.9,beta2:float=0.999,epsilon:float=1e-8):
        self.layers = layers
        self.forward_params = []

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
        Save model architecture and weights to .npz file (CPU compatible)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        layer_configs = [layer.get_config() for layer in self.layers]

        params = {}
        # Convert configs to JSON string
        params['layer_configs'] = np.array([json.dumps(layer_configs)]) 

        for i, layer in enumerate(self.layers):
            weights = layer.get_weights()
            if weights:
                for key, val in weights.items():
                    # val is already converted to CPU in get_weights()
                    params[f'layer_{i}_weights_{key}'] = val
            
            opt_state = layer.get_optimizer_state()
            if opt_state:
                for key, val in opt_state.items():
                    params[f'layer_{i}_optimizer_{key}'] = val

        params['training_state'] = np.array([json.dumps({
            'learning_rate': float(self.learning_rate),
            'epoch_start': int(self.epoch_start),
            'cost_history': [float(x) for x in self.cost_history] # Ensure pure python floats
        })])

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
        
        layers = []

        if 'layer_configs' not in data:
            print("Error: No layer configurations found in .npz file.")
            return None
            
        config_str = str(data['layer_configs'][0])
        layer_configs = json.loads(config_str)
        
        layer_data = {}
        for key in data.files:
            if not key.startswith('layer_'):
                continue
                
            parts = key.split('_')
            if len(parts) < 4: continue 
            
            try:
                layer_idx = int(parts[1])
                data_type = parts[2] 
                param_name = "_".join(parts[3:]) 
                
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = {'weights': {}, 'optimizer': {}}
                
                if data_type in ['weights', 'optimizer']:
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
            
            if i in layer_data:
                if layer_data[i]['weights']:
                    layer.set_weights(layer_data[i]['weights'])
                
                if layer_data[i]['optimizer']:
                    layer.set_optimizer_state(layer_data[i]['optimizer'])
            
            layers.append(layer)
        
        cnn = CNN(layers=layers)

        if 'training_state' in data:
            try:
                state_str = str(data['training_state'][0])
                state = json.loads(state_str)
                cnn.learning_rate = state.get('learning_rate', 0.001)
                cnn.epoch_start = state.get('epoch_start', 0)
                cnn.cost_history = state.get('cost_history', [])
                print(f"Resuming training from epoch {cnn.epoch_start} with LR={cnn.learning_rate}")
            except Exception as e:
                print(f"Warning: Could not load training state: {e}")
        
        cnn.unified_hyperparam(learning_rate=cnn.learning_rate)

        print("Model loaded successfully.")
        return cnn

    def unified_hyperparam(self, learning_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        for layer in self.layers:
            layer.modified_hyperparam(learning_rate, _Adam, beta1, beta2, epsilon)
        
    def forward(self,X:xp.ndarray,Y:xp.ndarray=None, training:bool=True)->xp.ndarray:
        self.forward_params = [X] 
        for i, layer in enumerate(self.layers):
            try:
                out = layer.forward_prop(self.forward_params[-1], training=training)
            except TypeError:
                out = layer.forward_prop(self.forward_params[-1])
            self.forward_params.append(out)
        return self.forward_params[-1]
    
    def calculate_cost(self, Y:xp.ndarray)->float:
        A_out = self.forward_params[-1]
        if len(A_out.shape) == 4: 
            A_out = A_out.reshape(A_out.shape[0], -1)
        elif len(A_out.shape) == 2 and A_out.shape[0] != Y.shape[0]: 
            if A_out.shape[0] != Y.shape[0] and A_out.shape[1] == Y.shape[0]:
                A_out = A_out.T
        
        Y_flat = Y.flatten()
        A_out = xp.clip(A_out, 1e-15, 1.0 - 1e-15)
        # Use simple indexing for cross entropy
        cost = -xp.mean(xp.log(A_out[xp.arange(Y_flat.shape[0]), Y_flat]))
        return float(cost) # Return python float
    
    def backward(self,dY:xp.ndarray):
        for i in range(self.len-1,-1,-1):
            dY = self.layers[i].backward_prop(dY)
    
    def train(self,X:xp.ndarray,Y:xp.ndarray,epochs:int=1000,batch_size:int=32,tolerance:float=1e-6,print_cost:bool=True, save_path:str=None):
        # Move data to GPU if using CuPy
        X = to_gpu(X)
        Y = to_gpu(Y)

        N = X.shape[0]
        num_batches = (N + batch_size - 1) // batch_size
        
        print(f"Training with {N} samples, batch_size={batch_size}, num_batches={num_batches}")
        epoch_accumulated = 0
        try:
            for i in range(epochs):
                epoch_accumulated += 1
                indices = xp.random.permutation(N)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]
                
                epoch_cost = 0
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, N)
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    Y_batch = Y_shuffled[start_idx:end_idx]
                    batch_size_actual = end_idx - start_idx
                    
                    self.forward_params = []
                    self.forward(X_batch)
                    
                    batch_cost = self.calculate_cost(Y_batch)
                    nnn = max(1, num_batches // 5)
                    if print_cost and batch_cost and batch_idx %nnn == 0:
                        print(f"Epoch {i + self.epoch_start}/{self.epoch_start + epochs} Batch {batch_idx}/{num_batches}  cost: {batch_cost:.6f}")
                    epoch_cost += batch_cost
                    
                    y_hat = self.forward_params[-1]
                    
                    if len(y_hat.shape) == 4:
                        y_hat = y_hat.reshape(y_hat.shape[0], -1)
                        num_classes = y_hat.shape[1]
                    elif len(y_hat.shape) == 2:
                        if y_hat.shape[1] == batch_size_actual:
                            num_classes = y_hat.shape[0] 
                            y_hat = y_hat.T 
                        elif y_hat.shape[0] == batch_size_actual:
                            num_classes = y_hat.shape[1]
                        else:
                            raise ValueError(f"Unexpected output shape: {y_hat.shape}")
                    else:
                        raise ValueError(f"Unexpected output shape: {y_hat.shape}")
                    
                    Y_onehot = xp.zeros((batch_size_actual, num_classes)) 
                    Y_onehot[xp.arange(batch_size_actual), Y_batch.flatten()] = 1
    
                    self.backward(Y_onehot)
                
                if num_batches > 0:
                    epoch_cost /= num_batches
                    self.cost_history.append(epoch_cost)
                    self.learning_rate *= 0.99
                    self.unified_hyperparam(learning_rate=self.learning_rate)

                if print_cost:
                    print(f'Cost after epoch {i}: {epoch_cost:.6f}')
                
                if save_path:
                    self.save_model(save_path)

                if i > 0 and len(self.cost_history) >= 2 and abs(self.cost_history[-1] - self.cost_history[-2]) < tolerance:
                    print(f'Converged after {i} epochs')
                    break
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            self.epoch_start += epoch_accumulated
            if save_path:
                print(f"Saving model to {save_path}...")
                self.save_model(save_path)
            return self.cost_history
        
        self.epoch_start += epoch_accumulated
        if save_path:
            print(f"Saving model to {save_path}...")
            self.save_model(save_path)
        return self.cost_history
        
    def predict(self, X:xp.ndarray, batch_size:int=32)->xp.ndarray:
        """Make predictions on input data with batch processing to avoid memory issues"""
        X = to_gpu(X)
        N = X.shape[0]
        
        if N <= batch_size:
            output = self.forward(X, training=False)
            if len(output.shape) == 4:
                output = output.reshape(output.shape[0], -1)
            elif len(output.shape) == 2:
                if output.shape[1] == N: output = output.T
                elif output.shape[0] == N: pass
                else:
                    if output.shape[0] != N and output.shape[1] == N: output = output.T
            
            return xp.argmax(output, axis=1).reshape(-1, 1)
        
        all_predictions = [] 
        num_batches = (N + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, N)
            X_batch = X[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx
            
            output = self.forward(X_batch, training=False)
            
            if len(output.shape) == 4: 
                output = output.reshape(output.shape[0], -1)
            elif len(output.shape) == 2:
                if output.shape[1] == batch_size_actual: output = output.T
                elif output.shape[0] == batch_size_actual: pass
                else:
                    if output.shape[0] != batch_size_actual and output.shape[1] == batch_size_actual: output = output.T
            
            batch_predictions = xp.argmax(output, axis=1).reshape(-1, 1)
            all_predictions.append(batch_predictions)
        
        return xp.concatenate(all_predictions, axis=0)

    def evaluate(self, X:xp.ndarray, Y:xp.ndarray, batch_size:int=32)->float:
        # X, Y need to be on GPU for predict
        X = to_gpu(X)
        Y = to_gpu(Y)
        predictions = self.predict(X, batch_size=batch_size)
        accuracy = xp.mean(predictions == Y)
        return float(accuracy)
