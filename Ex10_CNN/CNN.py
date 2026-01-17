import numpy as np
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
    N,_,H,W = X.shape
    h_out = (W + 2*padding[0] - filter_size) // stride + 1
    w_out = (H + 2*padding[1] - filter_size) // stride + 1

    # s_n = C*H*W, s_c = H*W, s_h = W, s_w = 1
    s_n,s_c,s_h,s_w = X.strides


    p_h, p_w = padding
    if p_h > 0 or p_w > 0:
        X_paded = np.pad(X, [(0,0), (0,0), (p_h, p_h), (p_w, p_w)], mode='constant')
    else:
        X_paded = X

    new_shape = (N,h_out,w_out,filter_c,filter_size,filter_size)
    print(new_shape)
    new_strides = (s_n, # N
                   s_h*stride, # h_out
                   s_w*stride, # w_out
                   s_c, # C
                   s_h, # f_H
                   s_w) # f_W
    X_out = as_strided(X_paded,shape=new_shape,strides=new_strides) # (N,h_out,w_out,filter_c,filter_size,filter_size)
    return X_out




def col2img(X_col:np.ndarray, stride:int=1, padding:tuple=(0,0))->np.ndarray:
    '''
    X_col: (N, h_out, w_out, X_C, filter_size, filter_size)
    X_shape: (N,X_C,X_H,X_W)
    filter_size: (f_H,f_W) or f_size
    stride: f_s
    padding: (p_H,p_W)
    return (N,X_C,X_H,X_W)
    '''
    N,h_out,w_out,X_C,filter_size,_ = X_col.shape
    h = (h_out - 1) * stride - 2 * padding[0] + filter_size
    w = (w_out - 1) * stride - 2 * padding[1] + filter_size
    s_n = X_C * h * w
    s_c = h * w
    s_h = w
    s_w = 1
    X_out = np.zeros(N*X_C*h*w)
    for p in range(N):
        for n in range(h_out):
            for m in range(w_out):
                for k in range(X_C):
                    for j in range(filter_size):
                        for i in range(filter_size):
                            X_out[p*s_n + n*s_h*stride + m*s_w*stride + k*s_c + j*s_h + i*s_w] += X_col[p,n,m,k,j,i]
    X_out = X_out.reshape(N,X_C,h,w)
    return X_out
    


class layer:
    def forward_prop(self,X:np.ndarray)->np.ndarray:
        pass
    def backward_prop(self,dY:np.ndarray)->np.ndarray:
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
        
    def forward_prop(self, X:np.ndarray)->np.ndarray:
        ''' 
        X: (N, X_C^{l-1}, X_H^{l-1}, X_W^{l-1})
        F: (f_c^{l}*f_size^{l}*f_size^{l})
        '''
        N,C,H,W = X.shape
        if self.padding is None:
            helper = (H - self.filter_size)%self.stride - self.stride
            while (helper % 2 != 0):
                helper -= self.stride
            p_h = -helper // 2
            helper = (W - self.filter_size)%self.stride - self.stride
            while (helper % 2 != 0):
                helper -= self.stride
            p_w = -helper // 2
            self.padding = [p_h,p_w]

        self.X_col = img2col(X,filter_c=self.filter_channel,filter_size=self.filter_size,stride=self.stride,padding=self.padding)
        # (N,h_out,w_out,filter_c,filter_size,filter_size)
        shape = self.X_col.shape
        
        Z = self.X_col.reshape(shape[0]*shape[1]*shape[2],-1) @ self.F + self.bias # (N * Z_H^{l} * Z_W^{l}, f_n^{l})
        
        return Z.reshape(shape[0],shape[1],shape[2],self.filter_num)# (N, Z_H^{l}, Z_W^{l}, f_n^{l})

    def back_prop(self, d_Z:np.ndarray)->np.ndarray:
        '''
        d_Z: (N * Z_H^{l} * Z_W^{l}, f_n^{l})
        '''
        d_F = self.X_col.T @ d_Z # (X_C * filter_size * filter_size, f_n^{l})
        d_bias = d_Z.sum(axis=0).reshape(1,-1) # (1,f_n^{l})
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
        
        d_X = col2img(d_X_col,stride=self.stride, padding=self.padding) # (N,X_C^{l-1}, X_H^{l-1}, X_W^{l-1}) 
        # d_X = d_X.unfold(2,self.filter_size,self.stride,self.padding) # (N,X_C^{l-1}, Z_H^{l-1}, Z_W^{l-1}, filter_size, filter_size)
        return d_X
        


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
        return 1 / (1 + np.exp(-X))
    def forward_prop(self, Z:np.ndarray)->np.ndarray:
        """
        Z: (N, Z_H^{l}, Z_W^{l}, f_n^{l})
        """
        self.Z = Z
        # to ensure activation function's emcapsulation, do not do reshape operation
        if self.activation == 'relu':
            self.Map = Z > 0
            return Z * self.Map
        elif self.activation == 'sigmoid':
            return self.sigmoid(Z)
        elif self.activation == 'softmax':
            exp_X = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            self.Indice = exp_X / np.sum(exp_X, axis=1, keepdims=True)
            return self.Indice
        else:
            raise ValueError('activation must be relu or sigmoid or softmax')
    def backward_prop(self, d_A:np.ndarray)->np.ndarray:
        """
        
        """
        if self.activation == 'relu':
            d_Z = d_A * self.Map
        elif self.activation == 'sigmoid':
            d_Z = self.sigmoid(self.Z) * (1 - self.sigmoid(self.Z))
        elif self.activation == 'softmax':
            d_Z = self.Indice - d_A # y_hat-y_onehot
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
        A: (N, Z_H^{l}, Z_W^{l}, f_n^{l})
        Returns: (N, h_out, w_out, f_n^{l})
        """
        N, H, W, f_n = A.shape
        if self.padding is None:
            helper = (H - self.pool_size)%self.stride - self.stride
            while (helper % 2 != 0):
                helper -= self.stride
            p_h = -helper // 2
            helper = (W - self.pool_size)%self.stride - self.stride
            while (helper % 2 != 0):
                helper -= self.stride
            p_w = -helper // 2
            self.padding = [p_h,p_w]
        
        # Reshape to (N*f_n, 1, H, W) for img2col
        A = A.transpose(0, 3, 1, 2)  # (N, f_n, H, W)
        A = A.reshape(N*f_n, 1, H, W)  # (N*f_n, 1, H, W)
        
        # Use img2col to extract pooling windows
        A_col = img2col(A, filter_size=self.pool_size, filter_c=1, stride=self.stride, padding=self.padding) 
        # A_col: (N*f_n, h_out, w_out, 1, pool_size, pool_size)
        self.A_col_shape = A_col.shape
        
        # Flatten spatial dimensions: (N*f_n*h_out*w_out, pool_size*pool_size)
        A_col_flat = A_col.reshape(self.A_col_shape[0]*self.A_col_shape[1]*self.A_col_shape[2], -1) # (N*f_n*h_out*w_out, pool_size*pool_size)  
        
        # Apply pooling along the pooling window dimension (axis=1)
        if self.pool_type == 'max':
            X_flat = A_col_flat.max(axis=1)  # (N*f_n*h_out*w_out,)
            Indices = A_col_flat.argmax(axis=1)  # (N*f_n*h_out*w_out,)
            self.onehot = np.zeros(A_col_flat.shape)  # (N*f_n*h_out*w_out, pool_size*pool_size)
            self.onehot[np.arange(Indices.shape[0]), Indices] = 1 # (N*f_n*h_out*w_out, pool_size*pool_size)
            
        else:
            X_flat = A_col_flat.mean(axis=1)
            self.onehot = None
        
        # Reshape back: (N*f_n, h_out, w_out)
        h_out, w_out = self.A_col_shape[1], self.A_col_shape[2]
        X = X_flat.reshape(N, f_n, h_out, w_out)
        
        # Reshape to (N, f_n, h_out, w_out) and transpose to (N, h_out, w_out, f_n)
        X = X.transpose(0, 2, 3, 1)
        
        # Store A_col shape and Indices for backward pass
        self.A_col_flat = A_col_flat
        
        return X  # (N, h_out, w_out, f_n)
    def back_prop(self,d_X:np.ndarray)->np.ndarray:
        """
        d_X: (N, h_out, w_out, f_n^{l})
        Returns: (N, H, W, f_n^{l})
        """

        N, h_out, w_out, f_n = d_X.shape
        
        # Flatten d_X to match the pooled output shape
        d_X_flat = d_X.transpose(0, 3, 1, 2).reshape(-1)  # (N*f_n*h_out*w_out,)
        
        # Create gradient array for A_col_flat: (N*f_n*h_out*w_out, pool_size*pool_size)
        pool_window_size = self.pool_size * self.pool_size
        
        if self.pool_type == 'max' and self.onehot is not None:
            d_A_col_flat = self.onehot * d_X_flat # (N*f_n*h_out*w_out, pool_size*pool_size)
        else:
            d_A_col_flat = d_X_flat / pool_window_size
        
        d_A_col = d_A_col_flat.reshape(N*f_n, h_out, w_out, 1, self.pool_size, self.pool_size)
        d_A = col2img(d_A_col,stride=self.stride, padding=self.padding) # (N*f_n, 1, H, W)
        d_A = d_A.reshape(N, f_n, d_A.shape[2], d_A.shape[3]) # (N, f_n, H, W)
        d_A = d_A.transpose(0, 2, 3, 1) # (N, H, W, f_n)
        
        return d_A
        
  

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

    def modified_hyperparam(self, learn_rate:float=0.001,
            _Adam:bool=False, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        self.learn_rate = learn_rate
        self._Adam = _Adam
        self.Adam_beta1 = beta1
        self.Adam_beta2 = beta2
        self.epsilon = epsilon
    def forward_prop(self,A:np.ndarray)->np.ndarray:
        """
        A: (N, X_H^{l}, X_W^{l}, f_n^{l}) (N, D_in)
        W: (D_in, D_out)
        """
        self.shape = A.shape
        self.A = A.reshape(-1,self.shape[0]) # (D_in, N)
        if self.input_size is None:
            self.input_size = self.A.shape[0]
            self.W = np.random.randn(self.input_size,self.output_size) * np.sqrt(2/self.input_size)
        
        Z = self.W.T @ self.A + self.b # (D_out, N)
        return Z
    def back_prop(self,d_Z:np.ndarray)->np.ndarray:
        """
        d_Z: (D_out, N) 
        """
        d_W = self.A @ d_Z.T # (D_in, D_out)
        d_b = np.sum(d_Z,axis=0) # (D_out, 1)
        d_A = self.W @ d_Z  # (D_in,N)
        d_A = d_A.reshape(self.shape)
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

        return d_A # (N, X_H^{l}, X_W^{l}, f_n^{l}) 这个是传给conv层的，因为FC层间是sigmoid，不需要传d_A


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
        for layer in self.layers:
            self.forward_params.append(layer.forward_prop(self.forward_params[-1]))
        return self.forward_params[-1]
    
    def calculate_cost(self, Y:np.ndarray)->float:
        """Calculate cross-entropy cost"""
        A_out = self.forward_params[-1]
        # Handle different output shapes
        if len(A_out.shape) == 4:  # (N, H, W, C)
            A_out = A_out.reshape(A_out.shape[0], -1)
        elif len(A_out.shape) == 2 and A_out.shape[0] != Y.shape[0]:  # (D_out, N)
            A_out = A_out.T  # (N, D_out)
        # Calculate cost
        Y_flat = Y.flatten()
        cost = -np.mean(np.log(A_out[np.arange(Y_flat.shape[0]), Y_flat] + 1e-8))
        return cost
    
    def backward(self,dY:np.ndarray):
        for i in range(self.len-1,-1,-1):
            dY = self.layers[i].backward_prop(dY)
    
    def train(self,X:np.ndarray,Y:np.ndarray,epochs:int=1000,batch_size:int=32,tolerance:float=1e-6,print_cost:bool=False):
        N = X.shape[0]
        num_batches = N // batch_size + 1  # Ceiling division
        
        for i in range(epochs):
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
                A = self.forward(X_batch)
                
                # Calculate cost for this batch
                batch_cost = self.calculate_cost(Y_batch)
                epoch_cost += batch_cost
                
                # Calculate gradient for backward pass
                # For softmax cross-entropy: dA = (A - Y_onehot) / batch_size
                y_hat = self.forward_params[-1]
                y_hat.shape
                
                # Handle different output shapes - convert to (N, D_out) format
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
            
            if print_cost and i % 10 == 0:
                print(f'Cost after epoch {i}: {epoch_cost:.6f}')
            
            if i > 0 and len(self.cost_history) >= 2 and abs(self.cost_history[-1] - self.cost_history[-2]) < tolerance:
                print(f'Converged after {i} epochs')
                break


# load t10k-images.idx3-ubyte
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    return data
images = load_mnist_images('./data/MNIST/raw/t10k-images-idx3-ubyte')

# 16字节的文件头 + 10000张图片 * 28 * 28像素 = 7840016字节
images = np.frombuffer(images, np.uint8, offset=16).reshape(-1, 28*28)

# load t10k-labels.idx1-ubyte
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    return data
labels = load_mnist_labels('./data/MNIST/raw/t10k-labels-idx1-ubyte')
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
images_train = load_mnist_images('./data/MNIST/raw/train-images-idx3-ubyte')
images_train = np.frombuffer(images_train, np.uint8, offset=16).reshape(-1, 28*28) #(N, 28*28)

# load train-labels.idx1-ubyte
labels_train = load_mnist_labels('./data/MNIST/raw/train-labels-idx1-ubyte')
# 8字节的文件头 + 60000个标签 = 10008字节
labels_train = np.frombuffer(labels_train, np.uint8, offset=8).reshape(-1, 1) #(N, 1)

show_images(images_train, labels_train, num_images=10 ,bias = 0)


A = images_train.reshape(-1,28,28,1) # (N,28,28,1)
A = A.transpose(0,3,1,2) # (N,1,28,28)
y_onehot = labels_train




# reguilarization parameter
A = A / 255.0
cnn1 = Conv(filter_num=8, filter_size=3, filter_channel=1, stride=1, _Adam=1, learn_rate=0.01)
act1 = Activation('relu')
pool1 = Pooling(pool_size=2, stride=2)
cnn2 = Conv(filter_num=16, filter_size=3, filter_channel=8, stride=1, _Adam=1, learn_rate=0.01) # f_c^{l} = f_n^{l-1}
act2 = Activation('relu')
pool2 = Pooling(pool_size=2, stride=2)
fc1 = FC(output_size=64, _Adam=1, learn_rate=0.01)
act_fc1 = Activation('sigmoid')
fc2 = FC(output_size=10, _Adam=1, learn_rate=0.01)
act_fc2 = Activation('softmax')
cnn = CNN(layers=[cnn1, act1, pool1, cnn2, act2, pool2, fc1, act_fc1, fc2, act_fc2])
cnn.train(X=A, Y=y_onehot, epochs=10)