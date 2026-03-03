from CNN_v4_cupy import *

import numpy as np
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


class YOLOv1:
    def __init__(self, layers:list[layer],
        lambda_coord:float=5.0,
        lambda_noobj:float=0.5,
        learning_rate:float=0.001,
        _Adam:bool=False,beta1:float=0.9,beta2:float=0.999,epsilon:float=1e-8,):
        self.layers = layers
        self.out = None
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.len = len(layers)

        self.learning_rate = learning_rate
        self._Adam = _Adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.cost_history = []
        self.accuracy_history = []

        # YOLO grid/box/class configuration (set later in explicit_init)
        self.B = None  # box number
        self.C = None  # class number
        self.S = None  # grid size (S_w, S_h)

        self.Indices = None
        self.start_idx = None
        self.end_idx = None

        self.iou = None

        self.isInBox = None
        self.isInGrid = None

        self.epoch_start = 0

    def explicit_init(self, Y:xp.ndarray, S:tuple[int, int], B:int, C:int):
        """Initialize YOLO-specific masks and store S, B, C on the model."""
        # Save configuration
        self.S = S
        self.B = B
        self.C = C

        self.isInBox = xp.zeros((*Y.shape[:-1], B))
        self.isInGrid = xp.zeros(Y.shape[:-1])
        BoxIndices = xp.arange(B)
        self.isInBox[..., BoxIndices] = (Y[..., BoxIndices*5+4] == xp.asarray(1.0))
        self.isInGrid = xp.sum(self.isInBox[..., BoxIndices], axis=-1) > xp.asarray(0)


    def save_model(self, save_path:str):
        """Save the model parameters to a file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        layer_configs = [layer.getconfig() for layer in self.layers]

        params = {}
        params['layer_configs'] = np.array(layer_configs, dtype=object)

        for i,layer in enumerate(self.layers):
            weight = layer.get_weights()
            if weight:
                for key, val in weight.items():
                    params[f'layer_{i}_weight_{key}']=val


        params['training_state'] = np.array({
            'learning_rate': self.learning_rate,    
            'epoch_start': self.epoch_start,
            'cost_history': np.array(self.cost_history, dtype=object)
        },dtype=object)

        np.savez_compressed(save_path, **params)
        print(f"Model saved to {save_path}")
    
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
        # Group data by layer index and type (weights/hyperparam)
        layer_data = {}
        for key in data.files:
            # key format: layer_{i}_weights_{name} or layer_{i}_hyperparam_{name}
            if not key.startswith('layer_'):
                continue
                
            parts = key.split('_')
            if len(parts) < 4: continue # Not a weight/hyperparam key
            
            try:
                layer_idx = int(parts[1])
                data_type = parts[2] # 'weights' or 'hyperparam'
                param_name = "_".join(parts[3:]) # handle names with underscores if any
                
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = {'weights': {}, 'hyperparam': {}}
                
                if data_type in ['weights', 'hyperparam']:
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
            
            # Efficiently restore weights and hyperparam state
            if i in layer_data:
                if layer_data[i]['weights']:
                    layer.set_weights(layer_data[i]['weights'])
                
                if layer_data[i]['hyperparam']:
                    layer.set_hyperparams(layer_data[i]['hyperparam'])
            
            layers.append(layer)
        
        # Initialize CNN
        cnn = CNN(layers=layers)

        # Restore global training state if available
        if 'training_state' in data:
            try:
                state_val = data['training_state'].item()

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
       

    def forward(self, X:xp.ndarray, training:bool=True) -> xp.ndarray:
        self.out = X
        
        for i, layer in enumerate(self.layers):
            try:
                self.out = layer.forward_prop(self.out, training=training)
            except TypeError:
                # 不支持 training 参数的层（Conv / FC / Pooling / Activation）
                self.out = layer.forward_prop(self.out)
        
        if training == False:
            return to_cpu(self.out)
        else:
            return self.out

    def IoU(self,box1:xp.ndarray,box2:xp.ndarray)->xp.ndarray:
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2
        
        # 计算交集
        inter_x1 = xp.maximum(box1_x1, box2_x1)
        inter_y1 = xp.maximum(box1_y1, box2_y1)
        inter_x2 = xp.minimum(box1_x2, box2_x2)
        inter_y2 = xp.minimum(box1_y2, box2_y2)
        
        inter_width = xp.maximum(0, inter_x2 - inter_x1)
        inter_height = xp.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        box1_area = box1[..., 2] * box1[..., 3]
        box2_area = box2[..., 2] * box2[..., 3]
        union_area = box1_area + box2_area - inter_area
        
        # 计算 IoU
        iou = inter_area / (union_area + 1e-9)  # 加小值避免除零
        
        return iou 
    
    def Loss(self, Y:xp.ndarray)->float:
        out_tmp = self.out.reshape(Y.shape)
        # IoU per grid cell (broadcast to all B boxes)
        self.iou = self.IoU(out_tmp[..., :4], Y[..., :4])        # (N_batch, S_w, S_h)
        self.iou = self.iou[..., None]                            # (N_batch, S_w, S_h, 1)


        box_indices = xp.arange(self.B)
        
        isInBox = self.isInBox[self.indices[self.start_idx:self.end_idx]]
        isInGrid = self.isInGrid[self.indices[self.start_idx:self.end_idx]]
        # print(isInBox.shape,isInGrid.shape)

        # !!!!!!!!!!!!!
        epsilon = 1e-6

        CoordLoss = 0
        CoordLoss += xp.sum((
            xp.square(Y[..., box_indices*5]-out_tmp[..., box_indices*5]) + 
            xp.square(Y[..., box_indices*5+1]-out_tmp[..., box_indices*5+1])
            ) * isInBox
        )
        CoordLoss += xp.sum((
            xp.square(xp.sqrt(xp.clip(Y[..., box_indices*5+2], epsilon, None))-xp.sqrt(xp.clip(out_tmp[..., box_indices*5+2], epsilon, None))) + 
            xp.square(xp.sqrt(xp.clip(Y[..., box_indices*5+3], epsilon, None))-xp.sqrt(xp.clip(out_tmp[..., box_indices*5+3], epsilon, None)))
            ) * isInBox
        ) 
        CoordLoss *= self.lambda_coord  
        
        ConfLoss = 0
        ConfLoss += xp.sum(
            xp.square(out_tmp[..., box_indices*5+4] - self.iou) * isInBox
        )
        ConfLoss += self.lambda_noobj * xp.sum((
            xp.square(out_tmp[..., box_indices*5+4]) * (1-isInBox) # 可以约去0
            ) 
        )

        ClsLoss = xp.sum(
            xp.sum(xp.square(Y[..., self.B*5:] - out_tmp[..., self.B*5:]), axis=-1) * isInGrid
        )

        total = CoordLoss + ConfLoss + ClsLoss

        if xp.isnan(total) or xp.isinf(total):
            print("CoordLoss:", float(to_cpu(CoordLoss)),
                "ConfLoss:", float(to_cpu(ConfLoss)),
                "ClsLoss:", float(to_cpu(ClsLoss)))
        return total
        


    def backward(self, Y:xp.ndarray):
        out_tmp = self.out.reshape(Y.shape)
        # self.iou = self.IoU(out_tmp[..., :4], Y[..., :4])
        # self.iou = self.iou[..., None]

        isInBox = self.isInBox[self.indices[self.start_idx:self.end_idx]]
        isInGrid = self.isInGrid[self.indices[self.start_idx:self.end_idx]]
        # print(isInBox.shape,isInGrid.shape)

        # !!!!!!!!!!!!!
        epsilon = 1e-9

        # calculate dY
        BoxIndices = xp.arange(self.B)
        dY = xp.zeros_like(out_tmp)
        dY[..., BoxIndices*5] = self.lambda_coord * 2 * (
            out_tmp[..., BoxIndices*5] - Y[..., BoxIndices*5]
            ) * isInBox
        dY[..., BoxIndices*5+1] = self.lambda_coord * 2 * (
            out_tmp[..., BoxIndices*5+1] - Y[..., BoxIndices*5+1]
        ) * isInBox
        dY[..., BoxIndices*5+2] = self.lambda_coord * ( 
            1 - xp.sqrt(Y[...,BoxIndices*5+2] / (out_tmp[..., BoxIndices*5+2] + epsilon))
        ) * isInBox
        dY[..., BoxIndices*5+3] = self.lambda_coord * (
            1 - xp.sqrt(Y[...,BoxIndices*5+3] / (out_tmp[..., BoxIndices*5+3] + epsilon))
        ) * isInBox
        dY[..., BoxIndices*5+4] = 2 * (
            out_tmp[..., BoxIndices*5+4] - self.iou
        ) * isInBox
        dY[..., BoxIndices*5+4] = self.lambda_noobj * 2 * (
            out_tmp[..., BoxIndices*5+4]
        ) * (1-isInBox)
        dY[..., self.B*5:] = 2 * (
            out_tmp[..., self.B*5:] - Y[..., self.B*5:]
        ) * isInGrid[..., None]

        # Flatten to match final FC layer output (N, D_out) before backprop
        dY = dY.reshape(dY.shape[0], -1)

        for i in range(self.len-1,-1,-1):
            dY = self.layers[i].backward_prop(dY)

    def train(self, X:xp.ndarray, Y:xp.ndarray, 
        epochs:int=1000, batch_size:int=8, tolerance:float=1e-6,
        print_cost:bool=True, save_path:str=None):
        N = X.shape[0]

        num_batches = (N + batch_size - 1) // batch_size  # 修正计算方式
        
        print(f"Training with {N} samples, batch_size={batch_size}, num_batches={num_batches}")
        epoch_accumulated = 0

        # self.cost_history = self.cost_history.tolist()
        try:
            for i in range(epochs):
                epoch_accumulated += 1
                # Shuffle data at the beginning of each epoch
                self.indices = xp.random.permutation(N)
                X_shuffled = X[self.indices]
                Y_shuffled = Y[self.indices]
                
                epoch_cost = 0
                
                # Process in batches
                for batch_idx in range(num_batches):
                    self.start_idx = batch_idx * batch_size
                    self.end_idx = min(self.start_idx + batch_size, N)
                    
                    X_batch = X_shuffled[self.start_idx:self.end_idx]
                    Y_batch = Y_shuffled[self.start_idx:self.end_idx]
                    batch_size_actual = self.end_idx - self.start_idx
                    
                    # Clear the params
                    self.out = None

                    # Forward pass
                    self.forward(X_batch, training=True)
                    
                    # Calculate cost for this batch
                    batch_cost = self.Loss(Y_batch)
                    nnn = num_batches // 5
                    if print_cost and batch_cost and batch_idx % nnn == 0:
                        print(f"Epoch {i + self.epoch_start}/{self.epoch_start + epochs} Batch {batch_idx}/{num_batches}  cost: {batch_cost:.6f}")
                    epoch_cost += batch_cost
                    
                    self.backward(Y_batch)
                
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
                    self.learning_rate *= 0.99
                    self.unified_hyperparam(learning_rate=self.learning_rate)

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
            
            
            return to_cpu(self.cost_history)
        
        self.epoch_start += epoch_accumulated
        if save_path:
            print(f"Saving model to {save_path}...")
            self.save_model(save_path)
        self.cost_history = np.array(self.cost_history, dtype=object)
        return to_cpu(self.cost_history)


