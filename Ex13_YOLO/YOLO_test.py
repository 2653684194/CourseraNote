from LoadData import *
from YOLOv1 import *

S = (10,10)
C = 1
B = 2

X,Y = load_data(train_path, S=S, imgsize=IMG_SIZE, B=B, C=C)

yolo = YOLOv1(
    layers=[
        Conv(filter_num=64,filter_size=7,filter_channel=3,stride=2),
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
        Conv(filter_num=192,filter_size=3,filter_channel=64,stride=1),
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
        Conv(filter_num=128,filter_size=1,filter_channel=192,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=256,filter_size=3,filter_channel=128,stride=1),
        Activation('leaky_relu'),   
        Conv(filter_num=256,filter_size=1,filter_channel=256,stride=1),    
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1),    
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1),        
        Activation('leaky_relu'),
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1),    
        Activation('leaky_relu'),   
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1),    
        Activation('leaky_relu'),
        Conv(filter_num=256,filter_size=1,filter_channel=512,stride=1),            
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=3,filter_channel=256,stride=1),            
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=1,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=1024,filter_size=3,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Pooling(pool_size=2,stride=2,pool_type='max'),
        Conv(filter_num=512,filter_size=1,filter_channel=1024,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=1024,filter_size=3,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=512,filter_size=1,filter_channel=1024,stride=1),        
        Activation('leaky_relu'),
        Conv(filter_num=1024,filter_size=3,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=1024,filter_size=3,filter_channel=512,stride=1),
        Activation('leaky_relu'),
        Conv(filter_num=1024,filter_size=3,filter_channel=512,stride=2),   
        Activation('leaky_relu'),

        Conv(filter_num=1024,filter_size=3,filter_channel=1024,stride=1),   
        Activation('leaky_relu'),
        Conv(filter_num=1024,filter_size=3,filter_channel=1024,stride=1),   
        Activation('leaky_relu'),

        FC(output_size=4096),
        Activation('leaky_relu'),
        FC(output_size=S[0]*S[1]*(B*5+C)),
        # Activation('linear'),
    ]
)

X = to_gpu(X)
Y = to_gpu(Y)
        
yolo.explicit_init(Y, S,B,C)
yolo.train(X,Y,epochs=100,batch_size=8)
