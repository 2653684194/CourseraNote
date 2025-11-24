import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

#下载数据集
batch_size_train = 500 #每次投喂的样本数量
batch_size_test = 1000
#训练集数据
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True, #加载该数据集(download=True)
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), #Normalize()转换使用的值0.1307和0.3081是该数据集的全局平均值和标准偏差，这里将它们作为给定值
  batch_size=batch_size_train, shuffle=True)
#测试集数据
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True) #使用size=1000对这个数据集进行测试

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

show_images(images, labels, num_images=10 ,bias = 0)

# load train-images.idx3-ubyte
images_train = load_mnist_images('./data/MNIST/raw/train-images-idx3-ubyte')
images_train = np.frombuffer(images_train, np.uint8, offset=16).reshape(-1, 28*28)

# load train-labels.idx1-ubyte
labels_train = load_mnist_labels('./data/MNIST/raw/train-labels-idx1-ubyte')
# 8字节的文件头 + 60000个标签 = 10008字节
labels_train = np.frombuffer(labels_train, np.uint8, offset=8).reshape(-1, 1)

show_images(images_train, labels_train, num_images=10 ,bias = 0)

print(type(images_train))
print(type(labels_train))
print(images_train.shape)
print(labels_train.shape)

X = images_train
y_onehot = labels_train

m,k = X.shape # m 样本数, k 特征数

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # 指数可以无限小，但不能无限大，-maxsoft是为了防止溢出 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cost_func(X:np.ndarray,
              W1:np.ndarray,b1:np.ndarray,gamma1:np.ndarray,beta1:np.ndarray,
              W2:np.ndarray,b2:np.ndarray,gamma2:np.ndarray,beta2:np.ndarray,
              W3:np.ndarray,b3:np.ndarray,
              y:np.ndarray,epsilon:float=1e-6
              )->float:
    """
    X:(m,k), m is num of samples, k is features
    W1:(k,n1), n1 is num of neurons of layer1
    b1:(1,n1), 
    z1,z1_hat:(m,n1)
    gamma1,beta1:(1,n1)
    z1_tilde:(m,n1), elements_wise multiply
    a1:(m,n1)

    W2:(n1,n2), n2 is num of neurons of layer2
    b2:(1,n2), 
    z2,z2_hat:(m,n2)
    gamma2,beta2:(1,n2)
    z2_tilde:(m,n2), elements_wise multiply
    a2:(m,n2)

    W3:(n2,10)
    b3:(1,10)
    z3:(m,10)
    a3:(m,10)

    y:(m,10):one-hot coding
    ...
    """
    m=X.shape[0]
    # 可能调整矩阵形状

    z1 = X @ W1 + b1 # (m,n1)
    # sigma1 = np.sum(np.power((z1-z1.mean(axis=0,keepdims=True)),2),axis=0,keepdims=True) / m # sigma^2
    sigma1 = np.var(z1, axis=0, keepdims=True)  # 推荐这个！
    z1_hat = (z1 - z1.mean(axis=0,keepdims=True))/np.sqrt(sigma1+epsilon)
    z1_tilde = gamma1 * z1_hat + beta1 # 逐元素乘
    a1 = sigmoid(z1_tilde)
    z2 = a1 @ W2 + b2
    # sigma2 = np.sum(np.power((z2-z2.mean(axis=0,keepdims=True)),2),axis=0,keepdims=True) / m
    sigma2 = np.var(z2, axis=0, keepdims=True)  # 推荐这个！
    z2_hat = (z2 - z2.mean(axis=0,keepdims=True))/np.sqrt(sigma2+epsilon)
    z2_tilde = gamma2 * z2_hat + beta2 # 逐元素乘
    a2 = sigmoid(z2_tilde)
    z3 = a2 @ W3 + b3
    a3 = softmax(z3)

    #  # 数值稳定性处理
    a3 = np.clip(a3, 1e-12, 1.0 - 1e-12)

    # Muti_class classicatioin Entropy
    return -np.sum(y * np.log(a3))/m

def forward_prop(X:np.ndarray,
              W1:np.ndarray,b1:np.ndarray,gamma1:np.ndarray,beta1:np.ndarray,
              W2:np.ndarray,b2:np.ndarray,gamma2:np.ndarray,beta2:np.ndarray,
              W3:np.ndarray,b3:np.ndarray,
              y_onehot:np.ndarray,
              epsilon:float=1e-6
            ):
    m = y_onehot.shape[0]
    m=y_onehot.shape[0]
    z1 = X @ W1 + b1 # (m,n1)
    # sigma1 = np.sum(np.power((z1-z1.mean(axis=0,keepdims=True)),2),axis=0,keepdims=True) / m # sigma^2
    sigma1 = np.var(z1, axis=0, keepdims=True)  # 推荐这个！
    z1_hat = (z1 - z1.mean(axis=0,keepdims=True))/np.sqrt(sigma1+epsilon)
    z1_tilde = gamma1 * z1_hat + beta1 # 逐元素乘
    a1 = sigmoid(z1_tilde)
    z2 = a1 @ W2 + b2
    # sigma2 = np.sum(np.power((z2-z2.mean(axis=0,keepdims=True)),2),axis=0,keepdims=True) / m
    sigma2 = np.var(z2, axis=0, keepdims=True)  # 推荐这个！
    z2_hat = (z2 - z2.mean(axis=0,keepdims=True))/np.sqrt(sigma2+epsilon)
    z2_tilde = gamma2 * z2_hat + beta2 # 逐元素乘
    a2 = sigmoid(z2_tilde)
    z3 = a2 @ W3 + b3
    a3 = softmax(z3)

    return a3,a2,z2_hat,sigma2,a1,z1_hat,sigma1

def BN_backward(delta_hat:np.ndarray,z_hat:np.ndarray,sigma:np.ndarray,gamma:np.ndarray,epsilon:float=1e-6)->np.ndarray:
    m = delta_hat.shape[0]
    B = delta_hat.sum(axis=0,keepdims=True) / m
    C = (delta_hat * z_hat).sum(axis=0,keepdims=True) / m
    D = z_hat * C
    
    # print(gamma.shape,sigma.shape,delta_hat.shape)
    # return delta
    return ( gamma / np.sqrt(sigma + epsilon) )* (delta_hat - B -D)

# 随机初始化权重为小值， 以便找到权重对输入输出的影响
def randInitializeWeights(L_in: int, L_out: int) -> np.ndarray:
    epsilon_init = np.sqrt(6/(L_in + L_out))
    # 随机初始化权重矩阵W，范围在[-epsilon_init, epsilon_init]
    W = np.random.rand(L_in, L_out) * 2 * epsilon_init - epsilon_init
    return W

def EWA(V,Adam_beta1,d):
    return Adam_beta1 * V + (1-Adam_beta1) * d

def RMSprop(S,Adam_beta2,d):
    return Adam_beta2 * S + (1-Adam_beta2) * np.power(d,2)

def Adam(V,S,Adam_beta1,Adam_beta2,Para,d,iter,alpha,epsilon=1e-6):
    V_new = EWA(V,Adam_beta1,d)
    S_new = RMSprop(S,Adam_beta2,d)
    V_corrected = V_new / (1-np.power(Adam_beta1,iter + 1)) # iter 从0 开始
    S_corrected = S_new / (1-np.power(Adam_beta2,iter + 1))
    update=Para - alpha * V_corrected / np.sqrt(S_corrected + epsilon)
    return update,V_new,S_new


def Batch_Norm_Gradient_Descent(X:np.ndarray,y_onehot:np.ndarray,
                                epsilon:float=1e-6,
                                Adam_beta1:float=0.9,Adam_beta2:float=0.99,_Adam:bool=True,
                                alpha_decay_rate:float=0.5,
                                layer1_size:int = 8,layer2_size:int = 15,
                                batchsize:int=1000, alpha:float=0.01, iter:int=500
                                ):
    m,k=X.shape
    W1,b1 = randInitializeWeights(k,layer1_size),np.ones((1,layer1_size))
    W2,b2 = randInitializeWeights(layer1_size,layer2_size),np.ones((1,layer2_size))
    W3,b3 = randInitializeWeights(layer2_size,10),np.ones((1,10))
    gamma1,beta1= np.ones((1,layer1_size)), np.zeros((1,layer1_size))
    gamma2,beta2 = np.ones((1,layer2_size)), np.zeros((1,layer2_size))

    # Adam parameter
    VdW1, Vdb1, VdW2, Vdb2, VdW3, Vdb3, Vdgamma1, Vdbeta1, Vdgamma2, Vdbeta2 = \
        np.zeros((k,layer1_size)),np.zeros((1,layer1_size)),\
        np.zeros((layer1_size,layer2_size)),np.zeros((1,layer2_size)),\
        np.zeros((layer2_size,10)),np.zeros((1,10)),\
        np.zeros((1,layer1_size)),np.zeros((1,layer1_size)),\
        np.zeros((1,layer2_size)),np.zeros((1,layer2_size))
    SdW1, Sdb1, SdW2, Sdb2, SdW3, Sdb3, Sdgamma1, Sdbeta1, Sdgamma2, Sdbeta2 = \
        np.zeros((k,layer1_size)),np.zeros((1,layer1_size)),\
        np.zeros((layer1_size,layer2_size)),np.zeros((1,layer2_size)),\
        np.zeros((layer2_size,10)),np.zeros((1,10)),\
        np.zeros((1,layer1_size)),np.zeros((1,layer1_size)),\
        np.zeros((1,layer2_size)),np.zeros((1,layer2_size))

    
    cost_history = []
    
    for _ in range(iter):
        batch_num = int(m/batchsize) # better makesure batchsize devide m
        for i in range(batch_num):
            
            X_batch = X[i*batchsize:(i+1)*batchsize,:]
            y_batch = y_onehot[i*batchsize:(i+1)*batchsize,:]

            a3,a2,z2_hat,sigma2,a1,z1_hat,sigma1 = forward_prop(X_batch,W1,b1,gamma1,beta1,
                                                                W2,b2,gamma2,beta2,
                                                                W3,b3,y_batch,epsilon)
            delta3 = y_batch - a3 # (m,10)
            gradient_W3 = a2.T @ delta3 # (m,n2).T @ (m,10)
            gradient_b3 = delta3.sum(axis=0,keepdims=1) # (1,10) !!!
            
            delta2_hat = (delta3 @ W3.T) * (a2 * (1-a2)) # (m,n3) (n2,n3).T
            gradient_gamma2 = (delta2_hat * z2_hat).sum(axis=0,keepdims=1) # (m,n2) * (m,n2) Sum^(m) -> (1,n2)
            gradient_beta2 = delta2_hat.sum(axis=0,keepdims=1) # (1,n2)

            delta2 = BN_backward(delta2_hat,z2_hat,sigma2,gamma2,epsilon)
            gradient_W2 = a1.T @ delta2 # (m,n1).T @ (m,n2)
            gradient_b2 = delta2.sum(axis=0,keepdims=1) # (1,n2) !!!

            delta1_hat = (delta2 @ W2.T) * (a1 * (1-a1))
            gradient_gamma1 = (delta1_hat * z1_hat).sum(axis=0,keepdims=1)
            gradient_beta1 = delta1_hat.sum(axis=0,keepdims=1)

            delta1 = BN_backward(delta1_hat,z1_hat,sigma1,gamma1,epsilon)
            gradient_W1 = X_batch.T @ delta1 # (m,k).T @ (k,n1)
            gradient_b1 = delta1.sum(axis=0,keepdims=1)

            # Adam
            if _Adam:
                W1,VdW1,SdW1 = Adam(VdW1,SdW1,Adam_beta1,Adam_beta2,W1,gradient_W1,_,alpha,epsilon)
                b1,Vdb1,Sdb1 = Adam(Vdb1,Sdb1,Adam_beta1,Adam_beta2,b1,gradient_b1,_,alpha,epsilon)
                W2,VdW2,SdW2 = Adam(VdW2,SdW2,Adam_beta1,Adam_beta2,W2,gradient_W2,_,alpha,epsilon)
                b2,Vdb2,Sdb2 = Adam(Vdb2,Sdb2,Adam_beta1,Adam_beta2,b2,gradient_b2,_,alpha,epsilon)
                W3,VdW3,SdW3 = Adam(VdW3,SdW3,Adam_beta1,Adam_beta2,W3,gradient_W3,_,alpha,epsilon)
                b3,Vdb3,Sdb3 = Adam(Vdb3,Sdb3,Adam_beta1,Adam_beta2,b3,gradient_b3,_,alpha,epsilon)
                gamma1,Vdgamma1,Sdgamma1 = Adam(Vdgamma1,Sdgamma1,Adam_beta1,Adam_beta2,gamma1,gradient_gamma1,_,alpha,epsilon)
                beta1,Vdbeta1,Sdbeta1 = Adam(Vdbeta1,Sdbeta1,Adam_beta1,Adam_beta2,beta1,gradient_beta1,_,alpha,epsilon)
                gamma2,Vdgamma2,Sdgamma2 = Adam(Vdgamma2,Sdgamma2,Adam_beta1,Adam_beta2,gamma2,gradient_gamma2,_,alpha,epsilon)
                beta2,Vdbeta2,Sdbeta2 = Adam(Vdbeta2,Sdbeta2,Adam_beta1,Adam_beta2,beta2,gradient_beta2,_,alpha,epsilon)
            else:
                W1 = W1 - alpha * gradient_W1
                b1 = b1 - alpha * gradient_b1
                W2 = W2 - alpha * gradient_W2
                b2 = b2 - alpha * gradient_b2
                W3 = W3 - alpha * gradient_W3
                b3 = b3 - alpha * gradient_b3
                gamma1 = gamma1 - alpha * gradient_gamma1
                beta1 = beta1 - alpha * gradient_beta1
                gamma2 = gamma2 - alpha * gradient_gamma2
                beta2 = beta2 - alpha * gradient_beta2
            # if (i+1)*batchsize % 10000 ==0:
            #     print(f"{(i+1)*batchsize} / {m}\n")
                
        cost = cost_func(X,W1,b1,gamma1,beta1,W2,b2,gamma2,beta2,W3,b3,y_onehot)

        if ( _ > 0 and cost_history[-1] > cost):
            alpha *= 1.05
        else:
            alpha *=0.95
        # if ( _ > 0 and np.abs(cost_history[-1] - cost) < 1e-6 ):
        #     cost_history.append(cost)
        #     break
        cost_history.append(cost)
        # alpha = 1 / (1 + np.power(alpha_decay_rate,_)) * alpha0
        if (_+1) % 10==0:
            print(f"{_+1} / {iter} iterations\n")
            print(f"cost:{cost}")

    return W1,b1,gamma1,beta1,W2,b2,gamma2,beta2,W3,b3,cost_history
     

     # read data
X = images_train
y = labels_train
print(X)
print(y)

# transform onehot
m = y.shape[0]
y_onehot = np.zeros((m, 10))
y_onehot[np.arange(m), y.flatten()] = 1
# train the model

W1,b1,gamma1,beta1,W2,b2,gamma2,beta2,W3,b3,cost_history = Batch_Norm_Gradient_Descent(
    X,y_onehot
)

# plot the cost history
# plot the cost history
plt.figure(figsize=(8,4))
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Training Cost History')
plt.grid(True)
plt.show()

cost_history


# check credibility
a3,_,_,_,_,_,_ = forward_prop(X,W1,b1,gamma1,beta1,W2,b2,gamma2,beta2,W3,b3,y_onehot)
predicted = np.argmax(softmax(a3),axis=1,keepdims=1)
print(predicted)
print(y)
idx_correct = predicted==y
print(idx_correct.shape,predicted.shape)

cred = np.sum(idx_correct) / idx_correct.shape[0]
print(cred)