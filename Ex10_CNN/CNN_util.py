import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import os
import json

def img2col(X:np.ndarray, filter_size:int, stride:int = 1, padding:tuple[int,int]=(0,0), dilation:int = 1)->np.ndarray:
    '''
    X: (N,X_C,X_H,X_W)
    filter_size: (f_H,f_W) or f_size
    stride: f_s
    padding: (p_H,p_W)
    # return (N * h_out * w_out, X_C * filter_size * filter_size)
    return (N,h_out,w_out,C,filter_size,filter_size)
    '''
    N,C,H,W = X.shape # C == filter_c
    # if C!=filter_c:
    #     raise ValueError(f"img2col:channel of X({C}) is not match with channel of filter({filter_c})")
    # padding = (p_h, p_w), where p_h is height padding, p_w is width padding
    h_out = (H + 2*padding[0] - filter_size) // stride + 1  # height output uses H and padding[0] (p_h)
    w_out = (W + 2*padding[1] - filter_size) // stride + 1  # width output uses W and padding[1] (p_w)
    
    # Debug output (commented out)
    # print(f"DEBUG img2col: X.shape={X.shape}, filter_size={filter_size}, stride={stride}, padding={padding}")
    # print(f"DEBUG img2col: h_out={h_out}, w_out={w_out}")

    p_h, p_w = padding
    if p_h > 0 or p_w > 0:
        X_paded = np.pad(X, [(0,0), (0,0), (p_h, p_h), (p_w, p_w)], mode='constant')
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
    
    # Validate that the view will not access invalid memory
    # The last element accessed will be at position:
    # - h dimension: (h_out-1) * stride + (filter_size-1) in the padded image
    # - w dimension: (w_out-1) * stride + (filter_size-1) in the padded image
    last_h_idx = (h_out - 1) * stride + (filter_size - 1)
    last_w_idx = (w_out - 1) * stride + (filter_size - 1)
    
    # Debug output (commented out)
    # print(f"DEBUG img2col: X_paded.shape={X_paded.shape}, new_shape={new_shape}")
    # print(f"DEBUG img2col: X_paded.strides={X_paded.strides}, new_strides={new_strides}")
    # print(f"DEBUG img2col: last_h_idx={last_h_idx}, last_w_idx={last_w_idx}, H_padded={H_padded}, W_padded={W_padded}")
    
    if last_h_idx >= H_padded or last_w_idx >= W_padded:
        raise ValueError(f"img2col: as_strided would access invalid memory. "
                        f"last_h_idx={last_h_idx} >= H_padded={H_padded} or "
                        f"last_w_idx={last_w_idx} >= W_padded={W_padded}")
    
    # Debug output (commented out)
    # print(f"DEBUG img2col: About to call as_strided...")
    # import sys
    # sys.stdout.flush()  # Force flush to ensure output is printed
    
    try:
        X_out = as_strided(X_paded,shape=new_shape,strides=new_strides) # (N,h_out,w_out,filter_c,filter_size,filter_size)
        # print(f"DEBUG img2col: as_strided completed, X_out.shape={X_out.shape}")
        # sys.stdout.flush()
    except Exception as e:
        # print(f"DEBUG img2col: as_strided failed with error: {e}")
        # sys.stdout.flush()
        raise
    return X_out



def col2img(X_col:np.ndarray, stride:int=1, padding:tuple=(0,0))->np.ndarray:
    '''
    Vectorized col2img (optimized loop over filter dimensions only)
    X_col: (N, h_out, w_out, X_C, filter_size, filter_size)
    return (N, X_C, X_H, X_W)
    '''
    N, h_out, w_out, C, filter_size, _ = X_col.shape
    p_h, p_w = padding
    
    # Calculate output dimensions
    H = (h_out - 1) * stride - 2 * p_h + filter_size
    W = (w_out - 1) * stride - 2 * p_w + filter_size
    
    # Calculate padded dimensions
    H_padded = H + 2 * p_h
    W_padded = W + 2 * p_w
    
    # Initialize output array (padded)
    X_padded = np.zeros((N, C, H_padded, W_padded), dtype=X_col.dtype)
    
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
    if p_h > 0 or p_w > 0:
        return X_padded[:, :, p_h:p_h+H, p_w:p_w+W]
    else:
        return X_padded


class layer:
    def forward_prop(self,X:np.ndarray)->np.ndarray:
        pass
    def backward_prop(self,dY:np.ndarray)->np.ndarray:
        pass
    def modified_hyperparam(self, learn_rate=None, _Adam=None, beta1=None, beta2=None, epsilon=None):
        pass






class Conv(layer):
    def __init__(
            self, filter_num, filter_size, filter_channel, stride=1, 
            _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
            learn_rate=0.001, same_padding:bool=False):
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

        # initialize filters and bias
        self.F = np.random.randn(filter_channel * filter_size * filter_size, filter_num) / np.sqrt(filter_size * filter_size * filter_channel)
            # maybe more way to initialize filters
        self.bias = np.zeros((1, filter_num))  # (1, f_n^{l})
        # update everytime forward_prop is called
        self.X_col = None
        self.col_shape = None
        # update params
        self.epsilon = epsilon
        self.learn_rate = learn_rate
        # Adam optimizer
        self._Adam = _Adam
        self.Adam_beta1 = Adam_beta1
        self.Adam_beta2 = Adam_beta2
        
        self.S_F = np.zeros_like(self.F)
        self.V_F = np.zeros_like(self.F)
        self.S_bias = np.zeros_like(self.bias)
        self.V_bias = np.zeros_like(self.bias)
    

    def get_config(self):
        return {
            'type': 'Conv',
            'filter_num': self.filter_num,
            'filter_size': self.filter_size,
            'filter_channel': self.filter_channel,
            'stride': self.stride,
            '_Adam': self._Adam,
            'Adam_beta1': self.Adam_beta1,
            'Adam_beta2': self.Adam_beta2,
            'epsilon': self.epsilon,
            'same_padding': self.same_padding
        }
    def get_weights(self):
        return {
            'F': self.F,
            'bias': self.bias,
        }
    
    def get_optimizer_state(self):
        return {
            'S_F': self.S_F,
            'V_F': self.V_F,
            'S_bias': self.S_bias,
            'V_bias': self.V_bias,
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
        self.F = weights['F']
        self.bias = weights['bias']
    def set_optimizer_state(self, optimizer_state:dict):
        self.S_F = optimizer_state['S_F']
        self.V_F = optimizer_state['V_F']
        self.S_bias = optimizer_state['S_bias']
        self.V_bias = optimizer_state['V_bias']

    def modified_hyperparam(self, learn_rate=None, _Adam=None, beta1=None, beta2=None, epsilon=None):
        self.learn_rate = learn_rate
        if _Adam is not None:
            self._Adam = _Adam
        if beta1 is not None:
            self.Adam_beta1 = beta1
        if beta2 is not None:
            self.Adam_beta2 = beta2
        if epsilon is not None:
            self.epsilon = epsilon
        
    def forward_prop(self, X:np.ndarray)->np.ndarray:
        ''' 
        X: (N, X_C^{l-1}, X_H^{l-1}, X_W^{l-1})
        F: (f_c^{l}*f_size^{l}*f_size^{l})
        Returns: (N, f_n^{l}, Z_H^{l}, Z_W^{l})
        '''
        N,C,H,W = X.shape
        # adjust padding dynamically
        # stride seted to odd number is best, ensuring padding's validaity
        if self.padding is None:
            # helper = (H - self.filter_size)%self.stride - self.stride
            p_w,p_h = 0,0
            # while (helper % 2 != 0 or (H + 2*p_h - self.filter_size) // self.stride + 1 <= 0):
            #     if (self.stride%2==0):break
            #     helper -= self.stride # to ensure H+2*p-f == 0(mod s)
            #     p_h = -helper // 2
            # helper = (W - self.filter_size)%self.stride - self.stride
            # while (helper % 2 != 0 or (W + 2*p_w - self.filter_size) // self.stride + 1 <= 0):
            #     if (self.stride%2==0):break
            #     helper -= self.stride
            #     p_w = -helper // 2
            # Calculate normal padding with safety check to prevent infinite loop
            max_padding = max(H, W) + 10  # Safety limit
            while (H + 2*p_h - self.filter_size)%self.stride != 0 or \
                (H + 2*p_h - self.filter_size) // self.stride + 1 <= 0:
                p_h += 1
                if p_h > max_padding:
                    raise ValueError(f"Conv: Failed to find valid padding for H={H}, filter_size={self.filter_size}, stride={self.stride}")
            while (W + 2*p_w - self.filter_size)%self.stride != 0 or \
                (W + 2*p_w - self.filter_size) // self.stride + 1 <= 0:
                p_w += 1
                if p_w > max_padding:
                    raise ValueError(f"Conv: Failed to find valid padding for W={W}, filter_size={self.filter_size}, stride={self.stride}")
            if (self.same_padding):
                if ((H - 1) * self.stride + self.filter_size - H) % 2 == 0:
                    p_th = ((H - 1) * self.stride + self.filter_size - H) // 2
                    if ((H + 2*p_th - self.filter_size) // self.stride + 1) > 0:
                        p_h = p_th  
                else:
                    print(f"(H - 1) * self.stride + self.filter_size - H) % 2 != 0, H={H}, stride={self.stride}, filter_size={self.filter_size}, keep normal padding")
                if ((W - 1) * self.stride + self.filter_size - W) % 2 == 0:
                    p_tw = ((W - 1) * self.stride + self.filter_size - W) // 2
                    if ((W + 2*p_tw - self.filter_size) // self.stride + 1) > 0:
                        p_w = p_tw
                else:
                    print(f"(W - 1) * self.stride + self.filter_size - W) % 2 != 0, W={W}, stride={self.stride}, filter_size={self.filter_size}, keep normal padding") 
            self.padding = (p_h,p_w)
        
        # Debug output (commented out)
        # print(f"DEBUG Conv.forward_prop: About to call img2col with X.shape={X.shape}, filter_size={self.filter_size}, stride={self.stride}, padding={self.padding}")
        self.X_col = img2col(X,filter_size=self.filter_size,stride=self.stride,padding=self.padding)
        # print(f"DEBUG Conv.forward_prop: img2col completed, X_col.shape={self.X_col.shape}")
        # (N,h_out,w_out,filter_c,filter_size,filter_size)
        self.col_shape = self.X_col.shape # 每次forward_prop都要更新col_shape
        # print(f"DEBUG Conv.forward_prop: About to reshape, col_shape={self.col_shape}, F.shape={self.F.shape}")
        # import sys
        # sys.stdout.flush()
        self.X_col = self.X_col.reshape(self.col_shape[0]*self.col_shape[1]*self.col_shape[2],-1) # (N * h_out * w_out, X_C^{l-1} * filter_size^{l} * filter_size^{l})
        # print(f"DEBUG Conv.forward_prop: Reshape completed, X_col.shape={self.X_col.shape}")
        # sys.stdout.flush()
        
        # print(f"DEBUG Conv.forward_prop: About to do matrix multiplication: {self.X_col.shape} @ {self.F.shape}")
        # sys.stdout.flush()
        Z = self.X_col @ self.F + self.bias # (N * Z_H^{l} * Z_W^{l}, f_n^{l})
        # print(f"DEBUG Conv.forward_prop: Matrix multiplication completed, Z.shape={Z.shape}")
        # sys.stdout.flush()
        # Z = Z.reshape(shape[0],shape[1],shape[2],self.filter_num)# (N, Z_H^{l}, Z_W^{l}, f_n^{l})
        # print(f"DEBUG Conv.forward_prop: About to reshape Z, col_shape={self.col_shape}, filter_num={self.filter_num}")
        # sys.stdout.flush()
        Z = Z.reshape(self.col_shape[0],self.col_shape[1],self.col_shape[2],self.filter_num)# (N, Z_H^{l}, Z_W^{l}, f_n^{l})
        # print(f"DEBUG Conv.forward_prop: Reshape Z completed, Z.shape={Z.shape}")
        # sys.stdout.flush()
        # 下一层可能是任何一种层，所以要维护维度等于输入的维度
        # print(f"DEBUG Conv.forward_prop: About to transpose Z")
        # sys.stdout.flush()
        Z = Z.transpose(0,3,1,2)# (N, f_n^{l}, Z_H^{l}, Z_W^{l})
        # print(f"DEBUG Conv.forward_prop: Transpose completed, Z.shape={Z.shape}")
        # sys.stdout.flush()
        return Z
    def backward_prop(self, d_Z:np.ndarray)->np.ndarray:
        '''
        d_Z: (N, f_n^{l}, Z_H^{l}, Z_W^{l}) (统一维度设计)
        F: (X_C^{l-1} * filter_size^{l} * filter_size^{l}, f_n^{l})
        Returns: (N, X_C^{l-1}, X_H^{l-1}, X_W^{l-1})
        '''
        d_Z = d_Z.transpose(0,2,3,1).reshape(-1,self.filter_num) # (N * Z_H^{l} * Z_W^{l}, f_n^{l})
        
        d_F = self.X_col.T @ d_Z # (X_C * filter_size * filter_size, f_n^{l})
        d_bias = d_Z.sum(axis=0,keepdims=True).reshape(1,-1) # (1,f_n^{l})
        # d_F = np.clip(d_F, -10, 10)
        # d_bias = np.clip(d_bias, -10, 10)
        d_X_col = d_Z @ self.F.T # (N * Z_H^{l} * Z_W^{l}, X_C * filter_size * filter_size)
        # calculating gradient must be earlier than updating params
        if self._Adam:
            self.V_F = self.Adam_beta1 * self.V_F + (1-self.Adam_beta1) * d_F # (X_C * filter_size * filter_size, f_n^{l})
            self.S_F = self.Adam_beta2 * self.S_F + (1-self.Adam_beta2) * np.power(d_F,2) # (X_C * filter_size * filter_size, f_n^{l})
            self.F = self.F - self.learn_rate * self.V_F / (np.sqrt(self.S_F) + self.epsilon) # (X_C * filter_size * filter_size, f_n^{l})
            self.V_bias = self.Adam_beta1 * self.V_bias + (1-self.Adam_beta1) * d_bias # (1,f_n^{l})
            self.S_bias = self.Adam_beta2 * self.S_bias + (1-self.Adam_beta2) * np.power(d_bias,2) # (1,f_n^{l})
            self.bias = self.bias - self.learn_rate * self.V_bias / (np.sqrt(self.S_bias) + self.epsilon) # (1,f_n^{l})
        else:
            self.F = self.F - self.learn_rate * d_F # (X_C * filter_size * filter_size, f_n^{l})
            self.bias = self.bias - self.learn_rate * d_bias # (1,f_n^{l})  
        d_X_col = d_X_col.reshape(self.col_shape) # (N, h_out, w_out, filter_c, filter_size, filter_size)
        d_X = col2img(d_X_col,stride=self.stride, padding=self.padding) # (N,X_C^{l-1}, X_H^{l-1}, X_W^{l-1}) 
        # print(f"DEBUG: Conv.backward_prop: d_X shape={d_X.shape}")
        # print(f"DEBUG: Conv.backward_prop: filter_size={self.filter_size}, stride={self.stride}, padding={self.padding}")
        return d_X
        
        



class BatchNorm(layer):
    def __init__(self,
            _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
            momentum=0.8,
            learn_rate=0.001):
        self._Adam = _Adam
        self.Adam_beta1 = Adam_beta1
        self.Adam_beta2 = Adam_beta2
        self.epsilon = epsilon
        self.learn_rate = learn_rate

        
        self.input_shape = None

        # tmp params
        self.mu = None
        self.sigma = None
        self.y_hat = None
        self.y_tilde = None

        # learnable parameters
        self.gamma = None
        self.beta = None

        # Running statistics for inference
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum

        self.S_gamma = np.zeros_like(self.gamma)
        self.V_gamma = np.zeros_like(self.gamma)
        self.S_beta = np.zeros_like(self.beta)
        self.V_beta = np.zeros_like(self.beta)
        
    def get_config(self):
        return {
            'type': 'BatchNorm',
            '_Adam': self._Adam,
            'Adam_beta1': self.Adam_beta1,
            'Adam_beta2': self.Adam_beta2,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'learn_rate': self.learn_rate,
        }
    def get_weights(self):
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            # 保存推理用的统计量
            'running_mean': self.running_mean,
            'running_var': self.running_var,
        }
    def get_optimizer_state(self):
        return {
            'S_gamma': self.S_gamma,
            'V_gamma': self.V_gamma,
            'S_beta': self.S_beta,
            'V_beta': self.V_beta,
        }
    def set_config(self, config:dict):
        self._Adam = config['_Adam']
        self.Adam_beta1 = config['Adam_beta1']
        self.Adam_beta2 = config['Adam_beta2']
        self.epsilon = config['epsilon']
        self.momentum = config['momentum']
        self.learn_rate = config['learn_rate']
    def set_weights(self, weights:dict):
        # 兼容旧模型：老的 npz 里没有 running_mean / running_var
        self.gamma = weights.get('gamma', self.gamma)
        self.beta = weights.get('beta', self.beta)
        self.running_mean = weights.get('running_mean', self.running_mean)
        self.running_var = weights.get('running_var', self.running_var)
    def set_optimizer_state(self, optimizer_state:dict):
        self.S_gamma = optimizer_state['S_gamma']
        self.V_gamma = optimizer_state['V_gamma']
        self.S_beta = optimizer_state['S_beta']
        self.V_beta = optimizer_state['V_beta']

    def modified_hyperparam(self, learn_rate=None, _Adam=None, beta1=None, beta2=None, epsilon=None):
        self.learn_rate = learn_rate
        if _Adam is not None:
            self._Adam = _Adam
        if beta1 is not None:
            self.Adam_beta1 = beta1
        if beta2 is not None:
            self.Adam_beta2 = beta2
        if epsilon is not None:
            self.epsilon = epsilon
    def forward_prop(self, Z:np.ndarray, training:bool=True)->np.ndarray:
        # if not training:
        #     return Z
        # Z should be (N, ...)
        # Debug output (commented out)
        # print(f"DEBUG BatchNorm.forward_prop: Input Z.shape={Z.shape}, training={training}")
        # import sys
        # sys.stdout.flush()
        
        if self.input_shape is None:
            self.input_shape = tuple([1]+list(Z.shape[1:]))
            
            
        if self.gamma is None:
            self.gamma = np.ones(self.input_shape)
        if self.beta is None:
            self.beta = np.zeros(self.input_shape)

        # 初始化 running statistics
        if self.running_mean is None:
            self.running_mean = np.zeros(self.input_shape)
        if self.running_var is None:
            self.running_var = np.ones(self.input_shape)

        if training:
            # 训练模式：使用当前 batch 统计量，并更新 running_mean / running_var
            # print(f"DEBUG BatchNorm.forward_prop: About to compute mean, Z.shape={Z.shape}")
            # sys.stdout.flush()
            self.mu = Z.mean(axis=0, keepdims=True)
            # print(f"DEBUG BatchNorm.forward_prop: Mean completed, mu.shape={self.mu.shape}")
            # sys.stdout.flush()
            # print(f"DEBUG BatchNorm.forward_prop: About to compute variance")
            # sys.stdout.flush()
            self.sigma = np.var(Z, axis=0, keepdims=True)
            # print(f"DEBUG BatchNorm.forward_prop: Variance completed, sigma.shape={self.sigma.shape}")
            # sys.stdout.flush()

            # print(f"DEBUG BatchNorm.forward_prop: About to update running statistics")
            # sys.stdout.flush()
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sigma
            # print(f"DEBUG BatchNorm.forward_prop: Running statistics updated")
            # sys.stdout.flush()

            # print(f"DEBUG BatchNorm.forward_prop: About to compute y_hat")
            # sys.stdout.flush()
            self.y_hat = (Z - self.mu) / np.sqrt(self.sigma + self.epsilon)
            # print(f"DEBUG BatchNorm.forward_prop: y_hat computed, shape={self.y_hat.shape}")
            # sys.stdout.flush()
        else:
            # 推理模式：使用训练过程中累积的 running statistics 不需要传给backward_prop
            mu = self.running_mean
            sigma = self.running_var
            # DEBUG: 检查running statistics（仅在单样本时打印）
            # if Z.shape[0] == 1:  # 单样本预测时
            #     print(f"[DEBUG BatchNorm] Inference mode - Input shape: {Z.shape}, running_mean shape: {mu.shape if mu is not None else None}, running_var shape: {sigma.shape if sigma is not None else None}")
            #     if mu is not None:
            #         print(f"[DEBUG BatchNorm] running_mean stats: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
            #     if sigma is not None:
            #         print(f"[DEBUG BatchNorm] running_var stats: min={sigma.min():.4f}, max={sigma.max():.4f}, mean={sigma.mean():.4f}")
            # # 修复：确保sigma不会太小，避免数值不稳定
            # sigma = np.maximum(sigma, self.epsilon)  # 确保sigma至少为epsilon
            self.y_hat = (Z - mu) / np.sqrt(sigma + self.epsilon)# 预测时候虽然更新y_hat但是训练时会覆盖更新，不会出错

        # print(f"DEBUG BatchNorm.forward_prop: About to compute y_tilde")
        # sys.stdout.flush()
        y_tilde = self.gamma * self.y_hat + self.beta
        # print(f"DEBUG BatchNorm.forward_prop: y_tilde computed, shape={y_tilde.shape}")
        # sys.stdout.flush()
        return y_tilde
    
    def backward_prop(self, d_y_tilde:np.ndarray)->np.ndarray:
        # Batch dimension is always axis 0 now
        axis = 0
        d_gamma = (d_y_tilde * self.y_hat).sum(axis=axis,keepdims=True) 
        d_beta = d_y_tilde.sum(axis=axis,keepdims=True)

        B = d_y_tilde.mean(axis=axis,keepdims=True)
        C = (d_y_tilde * self.y_hat).mean(axis=axis,keepdims=True)
        D = self.y_hat * C

        if self._Adam:
            self.V_gamma = self.Adam_beta1 * self.V_gamma + (1-self.Adam_beta1) * d_gamma
            self.S_gamma = self.Adam_beta2 * self.S_gamma + (1-self.Adam_beta2) * np.power(d_gamma,2)
            self.gamma = self.gamma - self.learn_rate * self.V_gamma / (np.sqrt(self.S_gamma) + self.epsilon)
            self.V_beta = self.Adam_beta1 * self.V_beta + (1-self.Adam_beta1) * d_beta
            self.S_beta = self.Adam_beta2 * self.S_beta + (1-self.Adam_beta2) * np.power(d_beta,2)
            self.beta = self.beta - self.learn_rate * self.V_beta / (np.sqrt(self.S_beta) + self.epsilon)
        else:
            self.gamma = self.gamma - self.learn_rate * d_gamma
            self.beta = self.beta - self.learn_rate * d_beta
        d_Z = (self.gamma / np.sqrt(self.sigma + self.epsilon)) * (d_y_tilde - B - D)

        # print(f"DEBUG: BatchNorm.backward_prop: d_Z shape={d_Z.shape}")
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
    def get_weights(self):
        pass
    def get_optimizer_state(self):
        pass
    def set_config(self, config:dict):
        self.activation = config['activation']
    def set_weights(self, weights:dict):
        pass
    def set_optimizer_state(self, optimizer_state:dict):
        pass

    def modified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        pass
    @staticmethod
    def sigmoid(X:np.ndarray)->np.ndarray:
        # Numerically stable sigmoid
        X = X.clip(-100, 100)
        return np.where(X >= 0,
                       1 / (1 + np.exp(-X)),
                       np.exp(X) / (1 + np.exp(X)))
        # return 1 / (1 + np.exp(-X))
    def forward_prop(self, Z:np.ndarray)->np.ndarray:
        """
        Z: (N, ...)
        return: (N, ...) same as Z
        
        """
        self.Z = Z
        # to ensure activation function's emcapsulation, do not do reshape operation
        if self.activation == 'relu':
            self.Map = Z > 0
            return Z * self.Map
        elif self.activation == 'sigmoid':
            return self.sigmoid(Z)
        elif self.activation == 'softmax':
            if len(Z.shape) != 2: # (N, D_out)
                raise ValueError('FC layer softmax only support 2D input')
            # 对于语义分割，通常在通道维度进行softmax
            # 对于空间注意力，通常在H*W维度进行softmax
            exp_X = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # 数值稳定版本
            self.Indice = exp_X / np.sum(exp_X, axis=1, keepdims=True)
            return self.Indice
        else:
            raise ValueError('activation must be relu or sigmoid or softmax')
    def backward_prop(self, d_A:np.ndarray)->np.ndarray:
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
            d_Z = (self.Indice - d_A) / self.Z.shape[0] # y_hat-y_onehot
            # d_Z = (self.Indice - d_A) # y_hat-y_onehot
        else:
            raise ValueError('activation must be relu or sigmoid')

        # print(f"DEBUG: Activation.backward_prop: d_Z shape={d_Z.shape}")
        # print(f"DEBUG: Activation.backward_prop: Z shape={self.Z.shape}")
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
        self.A_col = None

    def get_config(self):
        return {
            'type': 'Pooling',
            
            'stride': self.stride,
            'pool_size': self.pool_size,
            'pool_type': self.pool_type,
            'same_padding': self.same_padding
        }
    def get_weights(self):
        pass
    def get_optimizer_state(self):
        pass
    def set_config(self, config:dict):
        self.stride = config['stride']
        self.pool_size = config['pool_size']
        self.pool_type = config['pool_type']
        self.same_padding = config['same_padding']
    def set_weights(self, weights:dict):
        pass
    def set_optimizer_state(self, optimizer_state:dict):
        pass

    def modified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        pass
    def forward_prop(self,A:np.ndarray)->np.ndarray:
        """
        A: (N, f_n^{l}, Z_H^{l}, Z_W^{l})(统一维度设计)
        Returns: (N, f_n^{l}, h_out, w_out)(统一维度设计)
        """
        # Debug output (commented out)
        # print(f"DEBUG Pooling.forward_prop: Input A.shape={A.shape}, pool_size={self.pool_size}, stride={self.stride}, pool_type={self.pool_type}")
        # import sys
        # sys.stdout.flush()
        N, f_n, H, W = A.shape
        # Save input shape for backward pass
        self.input_shape = (N, f_n, H, W)
        if self.padding is None:
            p_h,p_w = 0,0

            max_padding = max(H, W) + 10  # Safety limit
            while (H + 2*p_h - self.pool_size)%self.stride != 0 or \
                (H + 2*p_h - self.pool_size) // self.stride + 1 <= 0:
                p_h += 1
                if p_h > max_padding:
                    raise ValueError(f"Conv: Failed to find valid padding for H={H}, pool_size={self.pool_size}, stride={self.stride}")
            while (W + 2*p_w - self.pool_size)%self.stride != 0 or \
                (W + 2*p_w - self.pool_size) // self.stride + 1 <= 0:
                p_w += 1
                if p_w > max_padding:
                    raise ValueError(f"Conv: Failed to find valid padding for W={W}, pool_size={self.pool_size}, stride={self.stride}")


            if (self.same_padding):
                if ((H - 1) * self.stride + self.pool_size - H) % 2 == 0:
                    p_th = ((H - 1) * self.stride + self.pool_size - H) // 2
                    if ((H + 2*p_th - self.pool_size) // self.stride + 1) > 0:
                        p_h = p_th  
                else:
                    print(f"(H - 1) * self.stride + self.pool_size - H) % 2 != 0, H={H}, stride={self.stride}, pool_size={self.pool_size}, keep normal padding")
                if ((W - 1) * self.stride + self.pool_size - W) % 2 == 0:
                    p_tw = ((W - 1) * self.stride + self.pool_size - W) // 2
                    if ((W + 2*p_tw - self.pool_size) // self.stride + 1) > 0:
                        p_w = p_tw
                else:
                    print(f"(W - 1) * self.stride + self.pool_size - W) % 2 != 0, W={W}, stride={self.stride}, pool_size={self.pool_size}, keep normal padding")
            self.padding = (p_h,p_w)
        
        # Debug output (commented out)
        # print(f"DEBUG Pooling.forward_prop: padding={self.padding}, About to reshape A")
        # sys.stdout.flush()
        A = A.reshape(N*f_n, 1, H, W)  # (N*f_n, 1, H, W) # C = 1
        # print(f"DEBUG Pooling.forward_prop: A reshaped to {A.shape}, About to call img2col")
        # sys.stdout.flush()
        # Use img2col to extract pooling windows
        A_col = img2col(A, filter_size=self.pool_size, stride=self.stride, padding=self.padding) 
        # print(f"DEBUG Pooling.forward_prop: img2col completed, A_col.shape={A_col.shape}")
        # sys.stdout.flush()
        # A_col: (N*f_n, h_out, w_out, 1, pool_size, pool_size)
        self.A_col_shape = A_col.shape
        
        # print(f"DEBUG Pooling.forward_prop: About to flatten A_col, A_col_shape={self.A_col_shape}")
        # sys.stdout.flush()
        # Flatten spatial dimensions: (N*f_n*h_out*w_out, pool_size*pool_size)
        A_col_flat = A_col.reshape(self.A_col_shape[0]*self.A_col_shape[1]*self.A_col_shape[2], -1) # (N*f_n*h_out*w_out, pool_size*pool_size)  
        # print(f"DEBUG Pooling.forward_prop: A_col flattened to {A_col_flat.shape}")
        # sys.stdout.flush()
        
        # Apply pooling along the pooling window dimension (axis=1)
        if self.pool_type == 'max':
            # print(f"DEBUG Pooling.forward_prop: About to compute max pooling")
            # sys.stdout.flush()
            X_flat = A_col_flat.max(axis=1)  # (N*f_n*h_out*w_out,)
            # print(f"DEBUG Pooling.forward_prop: Max completed, About to compute argmax")
            # sys.stdout.flush()
            Indices = A_col_flat.argmax(axis=1)  # (N*f_n*h_out*w_out,)
            # print(f"DEBUG Pooling.forward_prop: Argmax completed, About to create onehot")
            # sys.stdout.flush()
            self.onehot = np.zeros(A_col_flat.shape)  # (N*f_n*h_out*w_out, pool_size*pool_size)
            # print(f"DEBUG Pooling.forward_prop: Onehot created, About to set indices")
            # sys.stdout.flush()
            self.onehot[np.arange(Indices.shape[0]), Indices] = 1 # (N*f_n*h_out*w_out, pool_size*pool_size)
            # print(f"DEBUG Pooling.forward_prop: Onehot indices set")
            # sys.stdout.flush()
            
        elif self.pool_type == 'avg':
            # print(f"DEBUG Pooling.forward_prop: About to compute mean (avg pooling)")
            # sys.stdout.flush()
            X_flat = A_col_flat.mean(axis=1)# (N*f_n*h_out*w_out,)
            # print(f"DEBUG Pooling.forward_prop: Mean completed")
            # sys.stdout.flush()
            self.onehot = None
        else:
            raise ValueError('pool_type must be max or avg')
        
        # Reshape back: (N, f_n, h_out, w_out)
        h_out, w_out = self.A_col_shape[1], self.A_col_shape[2]
        # print(f"DEBUG Pooling.forward_prop: About to reshape X_flat, h_out={h_out}, w_out={w_out}")
        # sys.stdout.flush()
        X = X_flat.reshape(N, f_n, h_out, w_out)
        # print(f"DEBUG Pooling.forward_prop: Reshape completed, X.shape={X.shape}")
        # sys.stdout.flush()

        return X  # (N, f_n, h_out, w_out)
    def backward_prop(self,d_X:np.ndarray)->np.ndarray:
        """
        d_X: (N, f_n^{l}, h_out, w_out)(统一维度设计)
        Returns: (N, f_n^{l}, H, W)
        """

        N, f_n, h_out, w_out = d_X.shape
        
        # Flatten d_X to match the pooled output shape
        d_X_flat = d_X.reshape(-1,1)  # (N*f_n*h_out*w_out,)
        
        
        # Create gradient array for A_col_flat: (N*f_n*h_out*w_out, pool_size*pool_size)
        pool_window_size = self.pool_size * self.pool_size
        
        weight = np.ones((d_X_flat.shape[0],pool_window_size)) / 4.0

        d_A = None
        if self.pool_type == 'max' and self.onehot is not None:
            d_A_col_flat = self.onehot * d_X_flat # (N*f_n*h_out*w_out, pool_size*pool_size)
        else:
            d_A_col_flat = weight * d_X_flat # (N*f_n*h_out*w_out, pool_size*pool_size)
        d_A_col = d_A_col_flat.reshape(N*f_n, h_out, w_out, 1, self.pool_size, self.pool_size)
        d_A = col2img(d_A_col,stride=self.stride, padding=self.padding) # (N*f_n, 1, H, W)   
        


        # # Reshape to match number of filters
        # d_A = d_A.reshape(N, f_n, d_A.shape[2], d_A.shape[3]) # (N, f_n, H_col2img, W_col2img)
        # # Ensure output matches input shape (crop or pad if necessary)
        # if hasattr(self, 'input_shape') and self.input_shape is not None:
        #     _, _, H_expected, W_expected = self.input_shape
        #     H_actual, W_actual = d_A.shape[2], d_A.shape[3]
            
        #     if H_actual != H_expected or W_actual != W_expected:
        #         # Crop or pad to match expected shape
        #         if H_actual > H_expected:
        #             d_A = d_A[:, :, :H_expected, :]
        #         elif H_actual < H_expected:
        #             pad_h = H_expected - H_actual
        #             d_A = np.pad(d_A, [(0,0), (0,0), (0, pad_h), (0,0)], mode='constant')
                
        #         if W_actual > W_expected:
        #             d_A = d_A[:, :, :, :W_expected]
        #         elif W_actual < W_expected:
        #             pad_w = W_expected - W_actual
        #             d_A = np.pad(d_A, [(0,0), (0,0), (0,0), (0, pad_w)], mode='constant')
        
        # return d_A  # (N, f_n, H, W)    

        # print(f"DEBUG: Pooling.backward_prop: d_A shape={d_A.shape}")
        # print(f"DEBUG: Pooling.backward_prop: pool_size={self.pool_size}, stride={self.stride}, padding={self.padding}")
        return d_A.reshape(N, f_n, d_A.shape[2], d_A.shape[3]) # (N, f_n, H, W)  
        
        

        
    



class FC(layer): # fully connected layer
    def __init__(self,output_size,
                _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
                learn_rate=0.001):
        self.input_size = None # default None, will be set in forward_prop
        self.output_size = output_size

        self.A = None
        self.shape = None

        self.W = None
        # bias shape (1, output_size) to broadcast with (output_size, N)
        self.b = np.zeros((1, output_size))

        # Adam optimizer
        self._Adam = _Adam
        self.Adam_beta1 = Adam_beta1
        self.Adam_beta2 = Adam_beta2
        self.epsilon = epsilon
        self.learn_rate = learn_rate
        
        self.S_W = None  # Will be initialized when W is created
        self.V_W = None
        self.S_b = np.zeros_like(self.b)
        self.V_b = np.zeros_like(self.b)

    def get_config(self):
        return {
            'type': 'FC',
            'output_size': self.output_size,
        }
    def get_weights(self):
        return {
            'W': self.W,
            'b': self.b,
        }
    def get_optimizer_state(self):
        return {
            'S_W': self.S_W,
            'V_W': self.V_W,
            'S_b': self.S_b,
            'V_b': self.V_b,
        }
    def set_config(self, config:dict):
        self.output_size = config['output_size']
    def set_weights(self, weights:dict):
        self.W = weights['W']
        self.b = weights['b']
    def set_optimizer_state(self, optimizer_state:dict):
        self.S_W = optimizer_state['S_W']
        self.V_W = optimizer_state['V_W']
        self.S_b = optimizer_state['S_b']
        self.V_b = optimizer_state['V_b']


    def modified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        self.learn_rate = learn_rate
        if _Adam is not None:
            self._Adam = _Adam
        if beta1 is not None:
            self.Adam_beta1 = beta1
        if beta2 is not None:
            self.Adam_beta2 = beta2
        if epsilon is not None:
            self.epsilon = epsilon
    def forward_prop(self,A:np.ndarray)->np.ndarray:
        """
        A: Can be (N, f_n^{l}, X_H^{l}, X_W^{l}) from conv/pool layers or (N, D_out_prev) from FC layers
        W: (D_in, D_out)
        return Z: (N, D_out)
        """
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
            self.W = np.random.randn(self.input_size,self.output_size) * np.sqrt(2/self.input_size)
            self.S_W = np.zeros_like(self.W)
            self.V_W = np.zeros_like(self.W)    

        Z = self.A @ self.W + self.b # (N, D_out)
        return Z
    def backward_prop(self,d_Z:np.ndarray)->np.ndarray:
        """
        d_Z: (N, D_out)
        return: (N, f_n^{l}, X_H^{l}, X_W^{l}) or (N, D_out_prev) same as A
        """
        d_W = self.A.T @ d_Z # (D_in, D_out)
        d_b = np.sum(d_Z,axis=0,keepdims=True) # (1, D_out)
        # d_W = np.clip(d_W, -10, 10)
        # d_b = np.clip(d_b, -10, 10)
        d_A = d_Z @ self.W.T  # (N, D_in)
        d_A = d_A.reshape(self.shape) # (N, H, W, C) or (N, D_out_prev)
        if self._Adam:
            self.V_W = self.Adam_beta1 * self.V_W + (1-self.Adam_beta1) * d_W # (D_in, D_out)
            self.S_W = self.Adam_beta2 * self.S_W + (1-self.Adam_beta2) * np.power(d_W,2) # (D_in, D_out)
            self.W = self.W - self.learn_rate * self.V_W / (np.sqrt(self.S_W) + self.epsilon) # (D_in, D_out)
            self.V_b = self.Adam_beta1 * self.V_b + (1-self.Adam_beta1) * d_b # (1, D_out)
            self.S_b = self.Adam_beta2 * self.S_b + (1-self.Adam_beta2) * np.power(d_b,2) # (1, D_out)
            self.b = self.b - self.learn_rate * self.V_b / (np.sqrt(self.S_b) + self.epsilon) # (1, D_out)
        else:
            self.W = self.W - self.learn_rate * d_W # (D_in, D_out)
            self.b = self.b - self.learn_rate * d_b # (1, D_out)
        # print(f"DEBUG: FC.backward_prop: d_A shape={d_A.shape}")
        return d_A # (N, f_n^{l}, X_H^{l}, X_W^{l})


class Dropout(layer):
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward_prop(self, A:np.ndarray, training:bool=True)->np.ndarray:
        if training:
            # Inverted dropout: Scale by 1/(1-p) so expected sum remains same
            keep_prob = 1 - self.drop_rate
            self.mask = (np.random.rand(*A.shape) < keep_prob) / keep_prob # *是解包运算符，把元组（2，2）解为2，2
            # / keep_prob缩放是为了保持数学期望值不变

            # self.mask = (np.random.rand(*A.shape) < keep_prob) # *是解包运算符，把元组（2，2）解为2，2
            return A * self.mask
        else:
            return A
            
    def backward_prop(self, dZ:np.ndarray)->np.ndarray:
        return dZ * self.mask

    def get_config(self):
        return {
            'type': 'Dropout',
            'drop_rate': self.drop_rate
        }
    def get_weights(self):
        return None
    def get_optimizer_state(self):
        return None
    def set_config(self, config:dict):
        self.drop_rate = config['drop_rate']
    def set_weights(self, weights:dict):
        pass
    def set_optimizer_state(self, optimizer_state:dict):
        pass


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
        Save model architecture and weights to .npz file
        """

        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        

        layer_configs = [layer.get_config() for layer in self.layers]

        params = {}
        params['layer_configs'] = np.array([json.dumps(layer_configs)]) 

        
        for i, layer in enumerate(self.layers):
            weights = layer.get_weights()
            if weights:
                for key, val in weights.items():
                    params[f'layer_{i}_weights_{key}'] = val
            
            opt_state = layer.get_optimizer_state()
            if opt_state:
                for key, val in opt_state.items():
                    params[f'layer_{i}_optimizer_{key}'] = val

        params['training_state'] = np.array([json.dumps({
            'learning_rate': self.learning_rate,
            'epoch_start': self.epoch_start,
            'cost_history': self.cost_history
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
        data = np.load(filepath)
        
        # self.layers = [] # remove self usage
        layers = []

        if 'layer_configs' not in data:
            print("Error: No layer configurations found in .npz file.")
            return None
            
        # Extract JSON string from numpy array
        config_str = str(data['layer_configs'][0])
        layer_configs = json.loads(config_str)
        
        # Pre-process data keys to avoid O(N*M) complexity
        # Group data by layer index and type (weights/optimizer)
        layer_data = {}
        for key in data.files:
            # key format: layer_{i}_weights_{name} or layer_{i}_optimizer_{name}
            if not key.startswith('layer_'):
                continue
                
            parts = key.split('_')
            if len(parts) < 4: continue # Not a weight/optimizer key
            
            try:
                layer_idx = int(parts[1])
                data_type = parts[2] # 'weights' or 'optimizer'
                param_name = "_".join(parts[3:]) # handle names with underscores if any
                
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = {'weights': {}, 'optimizer': {}}
                
                if data_type in ['weights', 'optimizer']:
                    layer_data[layer_idx][data_type][param_name] = data[key]
            except ValueError:
                continue

        for i, config in enumerate(layer_configs):
            layer_type = config.pop('type')
            if layer_type == 'Conv':
                layer = Conv(**config)
            elif layer_type == 'BatchNorm':
                layer = BatchNorm(**config)
            elif layer_type == 'Activation':
                layer = Activation(**config)
            elif layer_type == 'Pooling':
                layer = Pooling(**config)
            elif layer_type == 'FC':
                layer = FC(**config)
            elif layer_type == 'Dropout':
                try:
                    layer = Dropout(**config)
                except NameError:
                    print("Warning: Dropout layer found in config but class not defined in CNN_util.py")
                    continue
            else:
                print(f"Unknown layer type: {layer_type}")
                continue
            
            # Efficiently restore weights and optimizer state
            if i in layer_data:
                if layer_data[i]['weights']:
                    layer.set_weights(layer_data[i]['weights'])
                
                if layer_data[i]['optimizer']:
                    layer.set_optimizer_state(layer_data[i]['optimizer'])
            
            layers.append(layer)
        
        # Initialize CNN
        cnn = CNN(layers=layers)

        # Restore global training state if available
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
        
        # 4. Synchronize hyperparameters (learning rate, etc.) to all layers
        cnn.unified_hyperparam(learn_rate=cnn.learning_rate)

        print("Model loaded successfully.")
        return cnn

    def unified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        for layer in self.layers:
            layer.modified_hyperparam(learn_rate, _Adam, beta1, beta2, epsilon)
        
    def forward(self,X:np.ndarray,Y:np.ndarray=None, training:bool=True)->np.ndarray:
        self.forward_params = [X]  # Reset forward params for each forward pass
        # import sys
        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__
            # Debug output (commented out)
            # print(f"DEBUG CNN.forward: Processing layer {i}: {layer_type}, input shape={self.forward_params[-1].shape}")
            # sys.stdout.flush()
            # 尝试传递 training 参数（BatchNorm / Dropout 会用到）
            try:
                out = layer.forward_prop(self.forward_params[-1], training=training)
            except TypeError:
                # 不支持 training 参数的层（Conv / FC / Pooling / Activation）
                out = layer.forward_prop(self.forward_params[-1])
            # Debug output (commented out)
            # print(f"DEBUG CNN.forward: Layer {i} ({layer_type}) completed, output shape={out.shape}")
            # sys.stdout.flush()
            self.forward_params.append(out)
        return self.forward_params[-1]
    
    def calculate_cost(self, Y:np.ndarray)->float:
        """Calculate cross-entropy cost"""
        A_out = self.forward_params[-1]
        # Handle different output shapes
        if len(A_out.shape) == 4:  # (N, H, W, C)
            A_out = A_out.reshape(A_out.shape[0], -1)
        elif len(A_out.shape) == 2 and A_out.shape[0] != Y.shape[0]:  # (N, D_out)
            if A_out.shape[0] != Y.shape[0] and A_out.shape[1] == Y.shape[0]:
                A_out = A_out.T
        # Calculate cost
        Y_flat = Y.flatten()
        # Clip to prevent log(0)
        A_out = np.clip(A_out, 1e-15, 1.0 - 1e-15)
        cost = -np.mean(np.log(A_out[np.arange(Y_flat.shape[0]), Y_flat]))
        return cost
    
    def backward(self,dY:np.ndarray):
        for i in range(self.len-1,-1,-1):
            # !!!!!!!
            # if dY is None: # Add check for None gradient
            #      continue
            dY = self.layers[i].backward_prop(dY)
    
    def train(self,X:np.ndarray,Y:np.ndarray,epochs:int=1000,batch_size:int=32,tolerance:float=1e-6,print_cost:bool=True, save_path:str=None):
        N = X.shape[0]
        num_batches = N // batch_size + 1  # Ceiling division
        
        print(f"Training with {N} samples, batch_size={batch_size}, num_batches={num_batches}")
        epoch_accumulated = 0
        try:
            for i in range(epochs):
                epoch_accumulated += 1
                # Shuffle data at the beginning of each epoch
                indices = np.random.permutation(N)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]
                
                epoch_cost = 0
                
                # Process in batches
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, N)
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    Y_batch = Y_shuffled[start_idx:end_idx]
                    batch_size_actual = end_idx - start_idx
                    
                    # Clear the params
                    self.forward_params = []

                    # Forward pass
                    self.forward(X_batch)
                    
                    # Calculate cost for this batch
                    batch_cost = self.calculate_cost(Y_batch)
                    nnn = num_batches // 5
                    if print_cost and batch_cost and batch_idx %nnn == 0:
                        print(f"Epoch {i + self.epoch_start}/{self.epoch_start + epochs} Batch {batch_idx}/{num_batches}  cost: {batch_cost:.6f}")
                    epoch_cost += batch_cost
                    
                    # Calculate gradient for backward pass
                    # For softmax cross-entropy: dA = (A - Y_onehot) / batch_size
                    y_hat = self.forward_params[-1]
                    
                    # Handle different output shapes - convert to (N, D_out) format, set num_classes
                    if len(y_hat.shape) == 4:  # (N, H, W, C)
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
                    else:
                        raise ValueError(f"Unexpected output shape: {y_hat.shape}")
                    
                    # Create one-hot encoding for Y_batch
                    Y_onehot = np.zeros((batch_size_actual, num_classes)) # (N, D_out)
                    Y_onehot[np.arange(batch_size_actual), Y_batch.flatten()] = 1
    
                    # Backward pass
                    self.backward(Y_onehot)
                
                # Average cost for the epoch
                if num_batches > 0:
                    epoch_cost /= num_batches
                    self.cost_history.append(epoch_cost)
                    # 不适用学习率震荡更新， 依赖Adam优化器， 或者用learning rate decay
                    self.learning_rate *= 0.99
                    self.unified_hyperparam(learn_rate=self.learning_rate)

                if print_cost:
                    print(f'Cost after epoch {i}: {epoch_cost:.6f}')
                

                # Save model at end of epoch
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
    
    def predict(self, X:np.ndarray)->np.ndarray:
        """Make predictions on input data"""
        # DEBUG: 检查输入数据（仅在单样本时打印）- commented out
        # if X.shape[0] == 1:
        #     print(f"[DEBUG predict] Single sample - Input shape: {X.shape}, min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")
        
        # Forward pass
        # 推理阶段：关闭 Dropout，并让 BatchNorm 使用 running_mean / running_var
        output = None
        # if X.shape[0]==1:
        #     output = self.forward(X, training=False)
        # else:
        #     output = self.forward(X, training=True)
        output = self.forward(X, training=False)
        
        # DEBUG: 检查输出（仅在单样本时打印）- commented out
        # if X.shape[0] == 1:
        #     print(f"[DEBUG predict] Single sample - Output shape: {output.shape}, min: {output.min():.4f}, max: {output.max():.4f}")
        #     if len(output.shape) == 2 and output.shape[1] > 1:
        #         print(f"[DEBUG predict] Single sample - Output probabilities: {output[0]}")
        #         print(f"[DEBUG predict] Single sample - Predicted class: {np.argmax(output, axis=1)[0]}")

        # Handle different output shapes
        if len(output.shape) == 4:  # (N, H, W, C)
            output = output.reshape(output.shape[0], -1)
        elif len(output.shape) == 2:
             if output.shape[0] != X.shape[0] and output.shape[1] == X.shape[0]:
                 output = output.T

        # Return predicted classes (argmax)
        return np.argmax(output, axis=1).reshape(-1, 1)

    def evaluate(self, X:np.ndarray, Y:np.ndarray)->float:
        """Evaluate accuracy on test data"""
        predictions = self.predict(X)
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







