import numpy as np
# from PIL import Image
import os

"""
设计一个输出main的函数,把数据集的标签按类划分文件输出
"""

def read_xml_annotation(xml_dir_path:str, norm_form:bool=True):
    import xml.etree.ElementTree as ET
    """
    读取单个XML标注文件
    
    Args:
        xml_dir_path: XML文件目录路径
    
    Returns:
        dict: 包含图像信息和标注信息的字典
    """
    if not os.path.exists(xml_dir_path):
        raise FileNotFoundError(f"XML目录不存在: {xml_dir_path}")
    data = {}
    for xml_file in os.listdir(xml_dir_path):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir_path, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像信息
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 获取所有标注对象
            objects = [] # lists that contain dic info of each object
            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                if norm_form:
                    x, y, w, h = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height, (xmax - xmin) / width, (ymax - ymin) / height
                    objects.append({
                        'name': name,
                        'bbox': [x, y, w, h]
                    })
                else:
                    objects.append({
                        'name': name,
                        'bbox': [xmin, ymin, xmax, ymax]
                    })
        data[filename] = {
            'width': width,
            'height': height,
            'objects': objects
        }
    return data

def classify(data:dict, dir:str=None, keep_true:bool=False)->tuple[np.ndarray, list]:
    '''
    整理标注信息输出txt文件
    return:
        classified_files: 形状为(n, len(classes))的数组, 第i行第j列表示第i个文件是否属于第j个类别
        classes: 类别列表
    '''
    n = len(data.items())
    classes = []
    classified_files = np.asarray([])
    # 遍历字典的内容
    for i, filename in enumerate(data.items()):
        for obj in filename['objects']:
            name = obj['name']
            if name not in classes:
                classified_files = np.concatenate((classified_files, np.zeros((n, 1))), axis=1)
                classes.append(name)
            classified_files[i, classes.index(name)] += 1
    if dir is not None and os.path.exists(dir):
        for name in classes:
            if keep_true:
                indices = np.where(classified_files[:, classes.index(name)] >= 1)[0]
                np.savetxt(os.path.join(dir, f"{name}.txt"),
                    np.hstack(indices.reshape(-1,1), np.ones((indices.shape[0], 1))),
                    fmt='%d')
                continue
            np.savetxt(os.path.join(dir, f"{name}.txt"),
                np.hstack(np.arange(n).reshape(-1,1), classified_files[:, classes.index(name)]),
                fmt='%d')
    return classified_files, classes
            
def load_classified_data(imgs_dir:str, lbls_dir:str, img_size:tuple, 
                        classes:list=None, exclude_unclassified:bool=True, N:int=1e6,
                        dtype=np.float32
                        )->tuple[np.ndarray,np.ndarray, list]:
    '''
    加载分类数据
    Args:
        imgs_dir: 图像目录路径 .jpg ...
        lbls_dir: 标注目录路径 .txt
        img_size: 图像尺寸, (height, width)
        classes: 类别列表, None表示所有类别, 可无序
        exclude_unclassified: 是否排除未分类的图像
        N: 最大图像索引范围，因为数据集的图像索引不连续
    Returns:
        X:(N_actual,H,W,C)
        Y:(N_actual,len(classes))        
        classes: 类别列表
    '''
    IMGFILETYPE = '.jpg'
    LBLFILETYPE = '.txt'
    if not os.path.exists(imgs_dir) or not os.path.exists(lbls_dir):
        raise FileNotFoundError(f"图像目录或标注目录不存在: {imgs_dir}, {lbls_dir}")
    # calculate M
    cls_list = []
    for file in sorted(os.listdir(lbls_dir)):
        if not file.endswith(LBLFILETYPE):
            continue
        classname = str(file.split('.')[0].split('_')[0])
        if classes is not None and classname not in classes:
            continue
        if classname not in cls_list:
            cls_list.append(classname)
        
    M = len(cls_list)

    Y = np.zeros((N, M))


        

    for file in sorted(os.listdir(lbls_dir)):
        if not file.endswith(LBLFILETYPE):
            continue
        classname = str(file.split('.')[0].split('_')[0])
        if classes is not None and classname not in classes:
            continue
        j = cls_list.index(classname)
        with open(os.path.join(lbls_dir, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                line = [int(str) for str in parts if str != ''] # 避免空格多的情况

                if int(line[0]) >= N:
                    continue
                try:
                    idx, val = line
                    Y[idx, j] = 1 if val >= 0 else 0 # 实际上是 1，-1，0，这里暂时简化！！！！！！！-----------------------------

                except ValueError: # 'train.txt' 'trainval.txt' 'val.txt' 中没有标签
                    print(f"file: {file}, part:{parts}, line: {line}, j:{j}, Y shape:{Y.shape}")
                    continue

    valid_label_idx = np.any(Y, axis=0)
    Y = Y[:,valid_label_idx] # 过滤掉没有标注的标签, 这个由于‘train.txt’’trainval.txt‘'val.txt'中没有标签
    if exclude_unclassified:
        mask = np.any(Y, axis=1)
        Y = Y[mask]
    N = Y.shape[0] # 实际的图像数量
    X = np.zeros((N, img_size[0], img_size[1], 3),dtype = dtype)
    from PIL import Image
    i=0
    for file in sorted(os.listdir(imgs_dir)):
        if not file.endswith(IMGFILETYPE):
            print(f"form of file in {imgs_dir} is not {IMGFILETYPE}")          
            continue
        k = int(file.split('.')[0])
        if k >= N:
            continue        
        if mask[k] == False: # 避免过大无法加载
            continue

        img = Image.open(os.path.join(imgs_dir, file)).resize(img_size) # (height, width, 3)
        X[i] = np.array(img, dtype=dtype)
        i+=1
    return X, Y, cls_list


                
        
