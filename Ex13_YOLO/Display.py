import numpy as np
import matplotlib.pyplot as plt

def display_image(X:np.ndarray, Y_pred:np.ndarray, S:tuple=(10,10), B:int=2, C:int=1):
    """
    仅支持单张图片
    X: (H, W, C)
    Y_pred: (1, S[0] * S[1] * (B*5+C))  
    """
    # Y_reshape = Y_pred.reshape(S[0],S[1],B*5+C) # 为了确保映射正确 deepseeks说没必要
    Y_reshape = Y_pred.reshape(-1,B*5+C)
    BoxIndices = np.arange(B).astype(int)
    # print(X.shape)
    if (len(X.shape) == 3):
        width, height = X.shape[0], X.shape[1]
    elif (len(X.shape) == 4):
        width, height = X.shape[2], X.shape[3]

    
    Boxes_X = Y_reshape[..., BoxIndices*5]
    Boxes_Y = Y_reshape[..., BoxIndices*5+1]
    Boxes_W = Y_reshape[..., BoxIndices*5+2]
    Boxes_H = Y_reshape[..., BoxIndices*5+3]
    Boxes_Conf = Y_reshape[..., BoxIndices*5+4]
    Boxes_Class = Y_reshape[..., B*5:]
    thereshold = 0.7
    mask = Boxes_Conf > thereshold
    # print(Boxes_W)
    Boxes_X = Boxes_X[mask] * width
    Boxes_Y = Boxes_Y[mask] * height
    Boxes_W = Boxes_W[mask] * width
    Boxes_H = Boxes_H[mask] * height
    Boxes_Conf = Boxes_Conf[mask]
    Boxes_Class = np.argmax(Boxes_Class[mask.sum(axis=1)>=1], axis=1)

    # 绘制预测框
    plt.imshow(X)
    ax = plt.gca()

    for x,y,w,h,conf,cls in zip(Boxes_X,Boxes_Y,Boxes_W,Boxes_H,Boxes_Conf,Boxes_Class):
        ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, edgecolor='red', linewidth=2))
        ax.text(x-w/2, y-h/2-5, f'Cls:{cls}, Conf:{conf:.2f}', bbox=dict(facecolor='red', alpha=0.5))
    plt.show()


