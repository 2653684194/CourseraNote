import numpy as np
import os
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt


def img2col(X:np.ndarray, filter_size:int, filter_c:int, stride:int = 1, padding:tuple[int,int]=(0,0), dilation:int = 1)->np.ndarray:
    '''
    X: (N,X_C,X_H,X_W)
    filter_size: (f_H,f_W) or f_size
    stride: f_s
    padding: (p_H,p_W)
    # return (N * h_out * w_out, X_C * filter_size * filter_size)
    return (N,h_out,w_out,C,filter_size,filter_size)
    '''
    N, C, H, W = X.shape
    p_h, p_w = padding
    
    # Calculate output dimensions
    h_out = (H + 2*p_h - filter_size) // stride + 1
    w_out = (W + 2*p_w - filter_size) // stride + 1
    
    # Pad input
    if p_h > 0 or p_w > 0:
        X_padded = np.pad(X, [(0,0), (0,0), (p_h, p_h), (p_w, p_w)], mode='constant')
    else:
        X_padded = X
    
    # Create view with as_strided
    new_shape = (N, h_out, w_out, C, filter_size, filter_size)
    s_n, s_c, s_h, s_w = X_padded.strides
    new_strides = (s_n, s_h * stride, s_w * stride, s_c, s_h, s_w)
    
    X_col = as_strided(X_padded, shape=new_shape, strides=new_strides)
    return X_col





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
            learn_rate=0.001):
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

        self.padding = None #
        # initialize filters and bias
        self.F = np.random.randn(filter_channel * filter_size * filter_size, filter_num) / np.sqrt(filter_size * filter_size * filter_channel)
            # maybe more way to initialize filters
        
        self.bias = np.zeros((1, filter_num))  # (1, f_n^{l})
        self.X_col = None
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
        '''
        # print("Conv forward start", flush=True) 
        self.X_col = None # Force reset
        self.input_shape = X.shape
        N,C,H,W = X.shape
        if self.padding is None:
            helper = (H - self.filter_size)%self.stride - self.stride
            cnt = 0
            while (helper % 2 != 0):
                helper -= self.stride
                cnt += 1
                if cnt > 10: break
            p_h = -helper // 2
            if p_h < 0: p_h = 0
            
            helper = (W - self.filter_size)%self.stride - self.stride
            cnt = 0
            while (helper % 2 != 0):
                helper -= self.stride
                cnt += 1
                if cnt > 10: break
            p_w = -helper // 2
            if p_w < 0: p_w = 0
            self.padding = (p_h,p_w)

        self.X_col = img2col(X,filter_c=self.filter_channel,filter_size=self.filter_size,stride=self.stride,padding=self.padding)
        # (N,h_out,w_out,filter_c,filter_size,filter_size)
        shape = self.X_col.shape
        
        Z = self.X_col.reshape(shape[0]*shape[1]*shape[2],-1) @ self.F + self.bias # (N * Z_H^{l} * Z_W^{l}, f_n^{l})
        
        # Reshape to (N, H, W, C) then transpose to (N, C, H, W)
        Z = Z.reshape(shape[0], shape[1], shape[2], self.filter_num) # (N, H, W, C)
        return Z.transpose(0, 3, 1, 2) # (N, C, H, W)

    def backward_prop(self, d_Z:np.ndarray)->np.ndarray:
        '''
        d_Z: (N, f_n, H, W) from next layer (e.g. Activation)
        '''
        # Transpose to (N, H, W, f_n) and flatten to match forward pass logic
        d_Z = d_Z.transpose(0, 2, 3, 1).reshape(-1, self.filter_num)
        
        X_col_reshaped = self.X_col.reshape(-1, self.F.shape[0])
        d_F = X_col_reshaped.T @ d_Z 
        d_bias = d_Z.sum(axis=0).reshape(1,-1) 
        
        # Clip gradients
        d_F = np.clip(d_F, -1.0, 1.0)
        d_bias = np.clip(d_bias, -1.0, 1.0)
        
        d_X_col = d_Z @ self.F.T # (N * Z_H^{l} * Z_W^{l}, X_C * filter_size * filter_size)
        
        # Update params
        if self._Adam:
            self.V_F = self.Adam_beta1 * self.V_F + (1-self.Adam_beta1) * d_F 
            self.S_F = self.Adam_beta2 * self.S_F + (1-self.Adam_beta2) * np.power(d_F,2) 
            self.F = self.F - self.learn_rate * self.V_F / (np.sqrt(self.S_F) + self.epsilon)
            self.V_bias = self.Adam_beta1 * self.V_bias + (1-self.Adam_beta1) * d_bias 
            self.S_bias = self.Adam_beta2 * self.S_bias + (1-self.Adam_beta2) * np.power(d_bias,2)
            self.bias = self.bias - self.learn_rate * self.V_bias / (np.sqrt(self.S_bias) + self.epsilon)
        else:
            self.F = self.F - self.learn_rate * d_F 
            self.bias = self.bias - self.learn_rate * d_bias   
        
        # Reshape d_X_col to 6D for col2img
        d_X_col = d_X_col.reshape(self.X_col.shape)
        d_X = col2img(d_X_col,stride=self.stride, padding=self.padding)       
        return d_X



class BatchNorm(layer):
    def __init__(self,
            _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
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

        self.S_gamma = np.zeros_like(self.gamma)
        self.V_gamma = np.zeros_like(self.gamma)
        self.S_beta = np.zeros_like(self.beta)
        self.V_beta = np.zeros_like(self.beta)
        
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
    def forward_prop(self, Z:np.ndarray)->np.ndarray:
        # Z should be (N, ...)
        
        if self.input_shape is None:
             # For FC layers, shape is (N, D_out)
            if len(Z.shape) == 2:
                 # Parameter shape needs to broadcast with (N, D_out)
                 # Gamma/Beta should be (1, D_out)
                 self.input_shape = (1, Z.shape[1])
            else:
                # For Conv layers (N, C, H, W)
                # Gamma/Beta should be (1, C, H, W) to broadcast
                self.input_shape = tuple([1] + list(Z.shape[1:]))
            
        if self.gamma is None:
            self.gamma = np.ones(self.input_shape)
        if self.beta is None:
            self.beta = np.zeros(self.input_shape)

        # Batch dimension is always axis 0 now
        mu = Z.mean(axis=0, keepdims=True)
        sigma = np.var(Z, axis=0, keepdims=True)
        y_hat = (Z - mu) / np.sqrt(sigma + self.epsilon)
        y_tilde = self.gamma * y_hat + self.beta

        self.mu = mu
        self.sigma = sigma
        self.y_hat = y_hat
        self.y_tilde = y_tilde
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
        return d_Z


class Activation(layer):
    def __init__(self, activation:str='relu',
            _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
            learn_rate=0.001):
        self.activation = activation
        self.Z = None
        self.Map = None
        self.Indice = None
    def modified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        self.learn_rate = learn_rate
        self._Adam = _Adam
        self.Adam_beta1 = beta1
        self.Adam_beta2 = beta2
        self.epsilon = epsilon  
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
        """
        self.Z = Z
        # to ensure activation function's emcapsulation, do not do reshape operation
        if self.activation == 'relu':
            self.Map = Z > 0
            return Z * self.Map
        elif self.activation == 'sigmoid':
            return self.sigmoid(Z)
        elif self.activation == 'softmax':
            # Handle both (N, num_classes)
            # Input is (N, num_classes)
            # We want to perform softmax along the LAST axis (classes)
            
            # For numerical stability subtract max
            exp_X = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
            self.Indice = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
            return self.Indice
        else:
            raise ValueError('activation must be relu or sigmoid or softmax')
    def backward_prop(self, d_A:np.ndarray)->np.ndarray:
        """
        
        """
        if self.activation == 'relu':
            d_Z = d_A * self.Map
        elif self.activation == 'sigmoid':
            if d_A is None:
                print("ERROR: sigmoid backward received None d_A")
                return None
            sigmoid_z = self.sigmoid(self.Z)
            if sigmoid_z is None:
                print("ERROR: sigmoid(self.Z) returned None")
                return None
            
            d_Z = d_A * sigmoid_z * (1 - sigmoid_z)
        elif self.activation == 'softmax':
            # Softmax + CrossEntropy combination typically simplifies to (A - Y)
            # Here we assume d_A is passed as (y_hat - y_onehot) if combined, 
            # BUT standard backprop for activation layer alone is different.
            # However, in this codebase, the training loop passes (A - Y_onehot) as dY to the last layer.
            # If the last layer IS Softmax, then d_Z = d_A if d_A was indeed passed as (A - Y).
            # But wait, self.backward(Y_onehot) passes Y_onehot directly? No.
            
            # In CNN.train:
            # y_hat = self.forward_params[-1] (Softmax output)
            # Y_onehot created.
            # self.backward(Y_onehot)
            
            # If CNN.backward takes Y_onehot as dY, let's check:
            # def backward(self,dY:np.ndarray):
            #     for i in range(self.len-1,-1,-1):
            #         dY = self.layers[i].backward_prop(dY)
            
            # So the last layer (Softmax) receives Y_onehot as dY? That seems wrong for standard notation (dY usually means dL/dY).
            # But let's look at the old implementation:
            # d_Z = self.Indice - d_A # y_hat-y_onehot
            
            # So d_A coming in is treated as Y_onehot (target).
            # And the output d_Z is (y_hat - y_target).
            # This is the gradient of CrossEntropy + Softmax combined w.r.t Z.
            
            # So if input d_A is actually Y_onehot (N, D_out)
            # And self.Indice is y_hat (N, D_out)
            # Then d_Z = (y_hat - Y_onehot) / N (if averaging) or just sum.
            # The previous implementation divided by batch size in backward_prop? No, it did:
            # d_Z = (self.Indice - d_A) / N
            
            N = self.Indice.shape[0] # N is axis 0 now
            d_Z = (self.Indice - d_A) / N # y_hat - y_onehot normalized
        else:
            raise ValueError('activation must be relu or sigmoid')
        return d_Z
        



class Pooling(layer):# check
    def __init__(self,stride=1,
             pool_size=2,pool_type='max',
             _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
            learn_rate=0.001):
        self.padding = None
        self.stride = stride
        self.pool_size = pool_size
        self.pool_type = pool_type
        
        self.onehot = None
        self.A_col = None
    def modified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        self.learn_rate = learn_rate
        self._Adam = _Adam
        self.Adam_beta1 = beta1
        self.Adam_beta2 = beta2
        self.epsilon = epsilon
    def forward_prop(self,A:np.ndarray)->np.ndarray:
        """
        A: (N, f_n, Z_H, Z_W) - Channels first
        Returns: (N, f_n, h_out, w_out) - Channels first
        """
        self.input_shape = A.shape
        N, f_n, H, W = A.shape
        if self.padding is None:
            helper = (H - self.pool_size)%self.stride - self.stride
            cnt = 0
            while (helper % 2 != 0):
                helper -= self.stride
                cnt += 1
                if cnt > 10: break
            p_h = -helper // 2
            if p_h < 0: p_h = 0
            
            helper = (W - self.pool_size)%self.stride - self.stride
            cnt = 0
            while (helper % 2 != 0):
                helper -= self.stride
                cnt += 1
                if cnt > 10: break
            p_w = -helper // 2
            if p_w < 0: p_w = 0
            self.padding = (p_h,p_w)
        
        # A is already (N, C, H, W)
        
        # Use img2col to extract pooling windows
        A_col = img2col(A, filter_size=self.pool_size, filter_c=1, stride=self.stride, padding=self.padding) 
        # A_col: (N, h_out, w_out, C, pool_size, pool_size)
        self.A_col_shape = A_col.shape
        
        # Flatten spatial dimensions: (N*h_out*w_out*C, pool_size*pool_size)
        A_col_flat = A_col.reshape(-1, self.pool_size*self.pool_size) 
        
        # Apply pooling along the pooling window dimension (axis=1)
        if self.pool_type == 'max':
            X_flat = A_col_flat.max(axis=1)
            self.Indices = A_col_flat.argmax(axis=1)
            self.onehot = np.zeros(A_col_flat.shape)
            self.onehot[np.arange(self.Indices.shape[0]), self.Indices] = 1
            
        else:
            X_flat = A_col_flat.mean(axis=1)
            self.onehot = None
        
        # Reshape back
        h_out, w_out = self.A_col_shape[1], self.A_col_shape[2]
        # First reshape to (N, h_out, w_out, C)
        X = X_flat.reshape(N, h_out, w_out, f_n)
        
        # Transpose to (N, C, h_out, w_out)
        X = X.transpose(0, 3, 1, 2)
        
        # Store A_col shape and Indices for backward pass
        self.A_col_flat = A_col_flat
        
        return X
    def backward_prop(self,d_X:np.ndarray)->np.ndarray:
        """
        d_X: (N, f_n, h_out, w_out)
        Returns: (N, f_n, H, W)
        """
        try:
            N, f_n, h_out, w_out = d_X.shape

            # Transpose d_X to match forward pass flattening order (N, h, w, C)
            d_X_trans = d_X.transpose(0, 2, 3, 1) # (N, h_out, w_out, f_n)
            d_X_flat = d_X_trans.reshape(-1) # (N*h*w*f_n)

            # Create gradient array for A_col_flat
            pool_window_size = self.pool_size * self.pool_size

            if self.pool_type == 'max' and self.onehot is not None:
                d_A_col_flat = np.zeros_like(self.onehot)
                d_A_col_flat[np.arange(d_X_flat.shape[0]), self.Indices] = d_X_flat
            else:
                d_A_col_flat = d_X_flat / pool_window_size

            # Reshape d_A_col to (N, h_out, w_out, C, pool_size, pool_size)
            d_A_col = d_A_col_flat.reshape(N, h_out, w_out, f_n, self.pool_size, self.pool_size)
            
            # col2img returns (N, C, H, W)
            d_A = col2img(d_A_col, stride=self.stride, padding=self.padding) 
            
            # Restore original shape if pixels were dropped
            if self.input_shape is not None and d_A.shape != self.input_shape:
                pad_h = self.input_shape[2] - d_A.shape[2]
                pad_w = self.input_shape[3] - d_A.shape[3]
                if pad_h > 0 or pad_w > 0:
                    d_A = np.pad(d_A, [(0,0), (0,0), (0, pad_h), (0, pad_w)], mode='constant')
            
            return d_A
        except Exception as e:
            print(f"Pooling back_prop error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
  

class FC(layer): # fully connected layer
    def __init__(self,output_size,
                _Adam = 0, Adam_beta1=0.9, Adam_beta2=0.999, epsilon=1e-8,
                learn_rate=0.001):
        self.input_size = None # default None, will be set in forward_prop
        self.output_size = output_size

        self.A = None
        self.shape = None

        self.W = None
        # bias shape (output_size, 1) to broadcast with (output_size, N)
        self.b = np.zeros((output_size, 1))

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

    def forward_prop(self,A:np.ndarray)->np.ndarray:
        """
        A: Can be (N, H, W, C) from conv/pool layers or (N, D_out_prev) from FC layers
        W: (D_in, D_out)
        """
        self.shape = A.shape
        
        # Handle different input formats
        if len(A.shape) == 4:  # From conv/pool layers: (N, H, W, C)
            # Flatten spatial dimensions
            N, H, W, C = A.shape
            self.A = A.reshape(N, -1)  # (N, D_in) where D_in = H*W*C
        elif len(A.shape) == 2:
            # From FC layers: should be (N, D_out_prev) format
            self.A = A
        else:
            raise ValueError(f"Unexpected input shape for FC layer: {A.shape}")

        if self.input_size is None:
            self.input_size = self.A.shape[1] # D_in is at index 1
            print(f"FC init: input_size={self.input_size}, output_size={self.output_size}")
            self.W = np.random.randn(self.input_size,self.output_size) * np.sqrt(2/self.input_size)
            self.S_W = np.zeros_like(self.W)
            self.V_W = np.zeros_like(self.W)

        # print(f"FC forward: A.shape={self.A.shape}, W.shape={self.W.shape}")
        Z = self.A @ self.W + self.b.T # (N, D_out) broadcast b (D_out, 1) to (1, D_out) effectively or just (D_out,)
        return Z

    def backward_prop(self, d_Z):
        # d_Z: (N, D_out)
        # self.A: (N, D_in)
        # self.W: (D_in, D_out)
        
        d_W = self.A.T @ d_Z # (D_in, N) @ (N, D_out) -> (D_in, D_out)
        d_b = np.sum(d_Z, axis=0, keepdims=True).T # Sum over N (axis 0) -> (1, D_out) -> T -> (D_out, 1)
        d_A = d_Z @ self.W.T  # (N, D_out) @ (D_out, D_in) -> (N, D_in)

        # Reshape d_A to match the expected input format for the previous layer
        if len(self.shape) == 4:  # Previous layer was conv/pool
            # Reshape back to (N, H, W, C) format
            # self.A was (N, D_in) flattened from (N, H, W, C)
            d_A = d_A.reshape(self.shape)  # (N, D_in) -> (N, H, W, C)
        elif len(self.shape) == 2:  # Previous layer was FC
            # Keep in (N, D_out_prev) format
            d_A = d_A  # Already (N, D_in)

        # Update weights with gradient clipping
        # Clip gradients to prevent explosion
        d_W = np.clip(d_W, -1.0, 1.0)
        d_b = np.clip(d_b, -1.0, 1.0)
        
        if self._Adam:
            self.V_W = self.Adam_beta1 * self.V_W + (1-self.Adam_beta1) * d_W
            self.S_W = self.Adam_beta2 * self.S_W + (1-self.Adam_beta2) * np.power(d_W,2)
            self.W = self.W - self.learn_rate * self.V_W / (np.sqrt(self.S_W) + self.epsilon)
            
            self.V_b = self.Adam_beta1 * self.V_b + (1-self.Adam_beta1) * d_b
            self.S_b = self.Adam_beta2 * self.S_b + (1-self.Adam_beta2) * np.power(d_b,2)
            self.b = self.b - self.learn_rate * self.V_b / (np.sqrt(self.S_b) + self.epsilon)
        else:
            self.W = self.W - self.learn_rate * d_W # (D_in, D_out)
            self.b = self.b - self.learn_rate * d_b # (D_out, 1)

        return d_A


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
    
    def unified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        for layer in self.layers:
            layer.modified_hyperparam(learn_rate, _Adam, beta1, beta2, epsilon)
        
    def forward(self,X:np.ndarray,Y:np.ndarray=None)->np.ndarray:
        self.forward_params = [X]  # Reset forward params for each forward pass
        for i, layer in enumerate(self.layers):
            out = layer.forward_prop(self.forward_params[-1])
            # if out.shape[0] != X.shape[0]: print(f"Shape mismatch at layer {i} {type(layer).__name__}: {out.shape} vs input {X.shape}", flush=True)
            self.forward_params.append(out)
        return self.forward_params[-1]
    
    def calculate_cost(self, Y:np.ndarray)->float:
        """Calculate cross-entropy cost"""
        A_out = self.forward_params[-1]

        # Handle different output shapes
        if len(A_out.shape) == 4:  # (N, H, W, C)
            A_out = A_out.reshape(A_out.shape[0], -1)
        elif len(A_out.shape) == 2:  # (N, D_out)
            # Ensure it's (N, D_out)
            if A_out.shape[0] != Y.shape[0]:
                 # Try transposing if user passed incorrect shape or previous layers did weird things
                 if A_out.shape[1] == Y.shape[0]:
                     A_out = A_out.T
        
        # Calculate cost
        Y_flat = Y.flatten()
        # Clip to prevent log(0)
        A_out = np.clip(A_out, 1e-15, 1.0 - 1e-15)
        cost = -np.mean(np.log(A_out[np.arange(Y_flat.shape[0]), Y_flat]))
        return cost
    
    def backward(self,dY:np.ndarray):
        for i in range(self.len-1,-1,-1):
            if dY is None: # Add check for None gradient
                 continue
            result = self.layers[i].backward_prop(dY)
            dY = result
    
    def train(self,X:np.ndarray,Y:np.ndarray,epochs:int=1000,batch_size:int=128,tolerance:float=1e-12,print_cost:bool=True):
        N = X.shape[0]
        num_batches = N // batch_size + 1  # Ceiling division

        print(f"Training with {N} samples, batch_size={batch_size}, num_batches={num_batches}")

        for i in range(epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            epoch_cost = 0


            # Process in batches
            for batch_idx in range(num_batches):
                # print(f"Starting Batch {batch_idx+1}/{num_batches}", flush=True) # DEBUG
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

                if batch_idx % 1 == 0:  # Print every batch
                    print(f"Epoch {i+1}/{epochs}, Batch {batch_idx+1}/{num_batches}: Cost = {batch_cost:.6f}", flush=True)
                epoch_cost += batch_cost

                # Calculate gradient for backward pass
                # For softmax cross-entropy: dA = (A - Y_onehot) / batch_size
                y_hat = self.forward_params[-1]

                # Handle different output shapes - convert to (N, D_out) format
                if len(y_hat.shape) == 4:  # (N, H, W, C)
                    y_hat = y_hat.reshape(y_hat.shape[0], -1)
                    num_classes = y_hat.shape[1]
                elif len(y_hat.shape) == 2:
                    # FC layer outputs (N, D_out) format
                    if y_hat.shape[0] == batch_size_actual:
                         # Correct (N, D_out)
                         num_classes = y_hat.shape[1]
                    elif y_hat.shape[1] == batch_size_actual:
                        # (D_out, N) -> Transpose to (N, D_out)
                        num_classes = y_hat.shape[0]
                        y_hat = y_hat.T
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
                # if len(self.cost_history) > 0 and epoch_cost > self.cost_history[-1]:
                #     self.learning_rate *= 0.5
                    
                # else:
                #     self.learning_rate *= 1.05
                self.learning_rate *= 0.95
                self.unified_hyperparam(learn_rate=self.learning_rate)
                self.cost_history.append(epoch_cost)
        

            # Print progress every epoch
            if print_cost:
                print(f'Epoch {i+1:3d}/{epochs}: Cost = {epoch_cost:.6f}')

            if i > 0 and len(self.cost_history) >= 2 and abs(self.cost_history[-1] - self.cost_history[-2]) < tolerance:
                print(f'Converged after {i+1} epochs')
                break

        print("Training completed!")
        return self.cost_history

    def predict(self, X:np.ndarray)->np.ndarray:
        """Make predictions on input data"""
        # Forward pass
        output = self.forward(X)

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


# Define base path
base_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_path, 'data', 'MNIST', 'raw')

# load t10k-images.idx3-ubyte
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    return data
images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))

# 16字节的文件头 + 10000张图片 * 28 * 28像素 = 7840016字节
images = np.frombuffer(images, np.uint8, offset=16).reshape(-1, 28*28)

# load t10k-labels.idx1-ubyte
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    return data
labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
# 8字节的文件头 + 10000个标签 = 10008字节
labels = np.frombuffer(labels, np.uint8, offset=8).reshape(-1, 1)


# 批量show
def show_images(images, labels, num_images=10, bias=0):
    plt.figure(figsize=(10, 1))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i + bias].reshape(28, 28), cmap='gray')
        plt.title(labels[i + bias][0])
        plt.axis('off')
    plt.show()


# load train-images.idx3-ubyte
images_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
images_train = np.frombuffer(images_train, np.uint8, offset=16).reshape(-1, 28*28) #(N, 28*28)

# load train-labels.idx1-ubyte
labels_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
# 8字节的文件头 + 60000个标签 = 10008字节
labels_train = np.frombuffer(labels_train, np.uint8, offset=8).reshape(-1, 1) #(N, 1)

# show_images(images_train, labels_train, num_images=10 ,bias = 0)


A = images_train.reshape(-1,28,28,1) # (N,28,28,1)
A = A.transpose(0,3,1,2) / 255.0 # (N,1,28,28)
y_onehot = labels_train




# Prepare test data
A_test = images.reshape(-1,28,28,1) # (N,28,28,1)
A_test = A_test.transpose(0,3,1,2) # (N,1,28,28)
A_test = A_test / 255.0
y_test = labels

print("Data shapes:")
print(f"Train: X={A.shape}, Y={y_onehot.shape}")
print(f"Test:  X={A_test.shape}, Y={y_test.shape}")

# Create CNN model - Moderate complexity with lower learning rate

cnn = CNN(layers=[
    Conv(filter_num=8, filter_size=3, filter_channel=1, stride=1, _Adam=1, learn_rate=0.01),
    Conv(filter_num=8, filter_size=3, filter_channel=8, stride=1, _Adam=1, learn_rate=0.01),
    BatchNorm(_Adam=1, learn_rate=0.01),
    Activation('relu'),

    Pooling(pool_size=3, stride=3),
    Conv(filter_num=16, filter_size=3, filter_channel=8, stride=1, _Adam=1, learn_rate=0.01),
    Conv(filter_num=16, filter_size=3, filter_channel=16, stride=1, _Adam=1, learn_rate=0.01),
    BatchNorm(_Adam=1, learn_rate=0.01),
    Activation('relu'),
    
    Pooling(pool_size=3, stride=3),
    FC(output_size=128, _Adam=1, learn_rate=0.01),
    BatchNorm(_Adam=1, learn_rate=0.01),
    Activation('relu'),
    FC(output_size=64, _Adam=1, learn_rate=0.01),
    BatchNorm(_Adam=1, learn_rate=0.01),
    Activation('relu'),
    FC(output_size=10, _Adam=1, learn_rate=0.01),
    Activation('softmax')
])

# Train the model
print("\nStarting training...")
cost_history = cnn.train(X=A, Y=y_onehot, epochs=10, batch_size=1024)

# Evaluate on test set
print("\nEvaluating on test set...")
test_accuracy = cnn.evaluate(A_test, y_test)
print(f"Test accuracy: {test_accuracy:.2%}")

# Show cost history
plt.figure(figsize=(10, 6))
plt.plot(cost_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Training Cost History')
plt.grid(True)
# plt.show()
plt.savefig('cost_history.png')

print(f"\nFinal training cost: {cost_history[-1]:.6f}")
print(f"Test accuracy: {test_accuracy:.2%}")


"""
1. 全连接层使用了 Sigmoid 激活函数（主要原因）
你现在的代码中，隐藏层的全连接层使用了 Sigmoid：
FC(output_size=512, ...),Activation('sigmoid'),  # <--- 这里是瓶颈
问题：Sigmoid 函数的导数最大只有 0.25（在 x=0 时），且在两端趋近于 0。当层数加深时，梯度在反向传播过程中会经过连续的链式法则相乘。例如经过 3 层 Sigmoid，梯度会至少衰减为 $0.25^3 \approx 0.015$。这会导致前面的卷积层几乎收不到梯度更新，也就无法学习到特征（即梯度消失）。
解决：请将隐藏层的激活函数改为 relu。ReLU 在正区间的导数为 1，可以很好地保持梯度强度，支持更深的网络训练。
2. 缺乏 Batch Normalization (BN层)
问题：现代深度网络（如 VGG, ResNet）之所以能堆叠几十层，核心归功于 Batch Normalization。没有 BN 层，深层网络的每一层输入分布都会剧烈变化（Internal Covariate Shift），导致训练非常困难，必须使用极小的学习率，收敛极慢。
现状：你的实现中没有 BN 层。在这种情况下，超过 3-4 层的网络（尤其是包含多个卷积堆叠时）非常难训练。
解决：在手写框架中实现 BN 比较复杂。如果没有 BN，建议保持网络“浅而宽”（例如 2 个卷积层 + 1-2 个全连接层），或者严格使用 ReLU 和较好的初始化（你的 He 初始化是 OK 的）。
3. 学习率策略震荡
问题：你代码中加入的 cost 上升就 *0.5，下降就 *1.05 的策略过于激进。对于深层网络，Loss 表面更复杂，这种大幅度的学习率跳变会导致模型在一个局部极小值附近反复横跳，无法收敛。
解决：使用平滑的衰减策略（如每 Epoch 衰减 0.95），或者完全信赖 Adam 优化器的自适应能力，固定一个较小的学习率（如 0.001）。

"""