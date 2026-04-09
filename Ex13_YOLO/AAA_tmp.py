import numpy as np
import os
from LoadKaggleDataSet import load_classified_data

GENERAL_DATA_DIR = 'archive_2007/VOCtrainval_06-Nov-2007/'
LBLS_DIR = os.path.join(GENERAL_DATA_DIR, 'ImageSets/Main')
IMGS_DIR = os.path.join(GENERAL_DATA_DIR, 'JPEGImages')

X, Y, classes = load_classified_data(IMGS_DIR, LBLS_DIR, (448,448), N=3500, 
                                     classes = [
                                        'bicycle',  
                                        'bus', 'car', 
                                        # 'cat', 'cow','dog', 'horse', 
                                        'motorbike', 'person',   
                                     ])
print(X.shape,Y.shape)

X_train = X.transpose(0, 3, 1, 2)

import matplotlib.pyplot as plt
def check_data(X, Y, batch_size=10, offset=50):
    start = offset
    end = min(offset + batch_size, X.shape[0])
    for i in range(start, end):
        plt.imshow(X[i].astype(np.uint8))
        plt.title(f"label: {Y[i]}")
        plt.axis('off')
        plt.show()
check_data(X, Y)


from CNN_v5_cupy import *

ai_layers = [
    # 特征聚合层 - 保持不变
    Conv(filter_num=512, filter_size=3, filter_channel=1024, stride=1, same_padding=True),
    BatchNorm(),
    Activation('leaky_relu'),
    
    # 全局平均池化
    Pooling(pool_size=14, stride=1, pool_type='avg'),
    
    # 分类头 - 保持不变
    FC(output_size=256),
    BatchNorm(),
    Activation('leaky_relu'),
    Dropout(0.5),
    
    FC(output_size=128),
    BatchNorm(),
    Activation('leaky_relu'),
    Dropout(0.3),
    
    FC(output_size=5),
]

backbone1 = [
    # ResBlock(Layers = [
        Conv(filter_num=64,filter_size=7,filter_channel=3,stride=2),
        # BatchNorm(),
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
    # ]),
    # ResBlock(Layers = [
        Conv(filter_num=192,filter_size=3,filter_channel=64,stride=1,same_padding=True),
        # BatchNorm(),
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
    # ]),
    
    ResBlock(Layers = [
        Conv(filter_num=128,filter_size=1,filter_channel=192,stride=1,same_padding=True),
        Activation('leaky_relu'),
        Conv(filter_num=256,filter_size=3,filter_channel=128,stride=1,same_padding=True),
        Activation('leaky_relu'),   
        Conv(filter_num=256,filter_size=1,filter_channel=256,stride=1,same_padding=True),    
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1,same_padding=True),    
        # BatchNorm(),
        Activation('leaky_relu'),

        Pooling(pool_size=2,stride=2,pool_type='max'),
    ]),
]
backbone2 = [
    # ResBlock(Layers = [
        Conv(filter_num=64,filter_size=7,filter_channel=3,stride=2),
        # BatchNorm(),
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
    # ]),
    # ResBlock(Layers = [
        Conv(filter_num=192,filter_size=3,filter_channel=64,stride=1,same_padding=True),
        # BatchNorm(),
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
    # ]),
    
    ResBlock(Layers = [
        Conv(filter_num=128,filter_size=1,filter_channel=192,stride=1,same_padding=True),
        Activation('leaky_relu'),
        Conv(filter_num=256,filter_size=3,filter_channel=128,stride=1,same_padding=True),
        Activation('leaky_relu'),   
        Conv(filter_num=256,filter_size=1,filter_channel=256,stride=1,same_padding=True),    
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1,same_padding=True),    
        # BatchNorm(),
        Activation('leaky_relu'),

        Pooling(pool_size=2,stride=2,pool_type='max'),
    ]),


    ResBlock(Layers=[
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1,same_padding=True),
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1,same_padding=True),        
        Activation('leaky_relu'),
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1,same_padding=True),
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1,same_padding=True),    
        Activation('leaky_relu'),   
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1,same_padding=True),
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1,same_padding=True),    
        Activation('leaky_relu'),
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1,same_padding=True),            
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1,same_padding=True),            
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=1,filter_channel=512,stride=1,same_padding=True),
        Activation('leaky_relu'),
        Conv(filter_num=1024,filter_size=3,filter_channel=512,stride=1,same_padding=True),
        # BatchNorm(),
        Activation('leaky_relu'),

        Pooling(pool_size=2,stride=2,pool_type='max'),                
    ]),
]

cnn = CNN(
    layers=[
            *backbone2,

            *ai_layers,
            Activation('sigmoid')

    ],
    learning_rate=0.001,
    _Adam=True,
)

path = 'models/yolov1_shallow.npz'
# cnn = CNN.load_model(path)
# cnn.train(X_train / 255.0, Y, batch_size=32, epochs=50, loss='binary', save_path=path)
# Y_pred = cnn.predict(X_train)

