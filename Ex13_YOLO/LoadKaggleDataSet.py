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
    加载分类数据 - 修复版
    1. 扫描所有图片文件，获取真实存在的文件ID列表
    2. 建立 ID -> Array Index 的映射
    3. 填充标签和图像数据
    '''
    IMGFILETYPE = '.jpg'
    LBLFILETYPE = '.txt'
    
    if not os.path.exists(imgs_dir) or not os.path.exists(lbls_dir):
        raise FileNotFoundError(f"图像目录或标注目录不存在: {imgs_dir}, {lbls_dir}")

    # 1. 扫描所有图片并获取有效 ID
    all_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(IMGFILETYPE)])
    valid_ids = []
    for f in all_files:
        try:
            fid = int(f.split('.')[0])
            valid_ids.append(fid)
        except ValueError:
            continue
            
    # 限制 N
    if len(valid_ids) > N:
        valid_ids = valid_ids[:N]
    
    # 建立映射: real_id -> 数组索引 (0, 1, 2...)
    id_to_idx = {real_id: i for i, real_id in enumerate(valid_ids)}
    num_samples = len(valid_ids)
    
    print(f"Found {num_samples} valid images (limit N={N})")

    # 2. 确定类别列表
    cls_list = []
    if classes is not None:
        cls_list = classes
    else:
        for file in sorted(os.listdir(lbls_dir)):
            if not file.endswith(LBLFILETYPE) or '_' not in file:
                continue
            classname = file.split('_')[0] # Assuming format: classname_train.txt
            if classname not in cls_list:
                cls_list.append(classname)
    
    M = len(cls_list)
    Y = np.zeros((num_samples, M))

    # 3. 读取标签并填充 Y
    for j, classname in enumerate(cls_list):
        # 查找对应的 txt 文件 (e.g. car_train.txt)
        # 这里假设文件名包含类别名，可能需要更精确的匹配逻辑如果 classes 参数传入了
        # 为了稳健，遍历目录找匹配的
        target_file = None
        for file in os.listdir(lbls_dir):
            if file.startswith(classname + '_') and file.endswith(LBLFILETYPE):
                target_file = file
                break
        
        if target_file:
            with open(os.path.join(lbls_dir, target_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    try:
                        real_id = int(parts[0])
                        val = int(parts[1])
                        
                        # 如果这个 ID 在我们需要加载的图片列表中
                        if real_id in id_to_idx:
                            idx = id_to_idx[real_id]
                            # VOC: 1=True, -1=False/Difficult
                            Y[idx, j] = 1 if val == 1 else 0
                    except ValueError:
                        continue
        else:
            print(f"Warning: Annotation file for class '{classname}' not found in {lbls_dir}")

    # 4. 过滤未分类样本 (如果 exclude_unclassified=True)
    # 注意：这可能会导致 X 和 Y 再次缩小
    if exclude_unclassified:
        mask = np.any(Y == 1, axis=1) # 只有当至少有一个标签为 1 时保留
        Y = Y[mask]
        # 更新 valid_ids 和 id_to_idx (虽然 id_to_idx 后面不用了，但为了逻辑完整)
        valid_ids = [valid_ids[i] for i in range(len(valid_ids)) if mask[i]]
        num_samples = len(valid_ids)
        print(f"After excluding unclassified: {num_samples} samples remaining")

    # 5. 加载图片填充 X
    X = np.zeros((num_samples, img_size[0], img_size[1], 3), dtype=dtype)
    from PIL import Image
    
    for i, real_id in enumerate(valid_ids):
        # 构造文件名，假设 VOC 格式是 6 位数字
        filename = f"{real_id:06d}{IMGFILETYPE}" 
        img_path = os.path.join(imgs_dir, filename)
        
        if not os.path.exists(img_path):
            # 尝试非补零格式 (虽然 VOC 都是补零的)
            filename = f"{real_id}{IMGFILETYPE}"
            img_path = os.path.join(imgs_dir, filename)
            
        try:
            img = Image.open(img_path).convert('RGB').resize(img_size)
            X[i] = np.array(img, dtype=dtype)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    return X, Y, cls_list


# ---------------------------------------------------------------------------
# Face Mask Detection 数据集 (XML 标注 → VOC Main 格式 + 加载)
# ---------------------------------------------------------------------------

def _natural_sort_key(s: str):
    """排序用：maksssksksss1, 2, 9, 10, 99 -> 按末尾数字排序"""
    import re
    m = re.search(r'(\d+)$', s)
    return (int(m.group(1)) if m else 0, s)


def build_face_mask_main(
    annotations_dir: str,
    output_main_dir: str,
    image_ext: str = '.png',
    split: str = 'trainval',
) -> tuple:
    """
    从 Face Mask Detection 的 XML 标注生成与 VOC ImageSets/Main 一致的标签文件。

    目录结构建议：
        Face_Mask_Dection/
            annotations/   <- annotations_dir (*.xml)
            images/        <- 图片 (maksssksksss99.png)
        生成：
            ImageSets/Main/  <- output_main_dir
                with_mask_trainval.txt
                without_mask_trainval.txt
                mask_weared_incorrect_trainval.txt
                images_list.txt   # 一行一个文件名 stem，顺序与 ID 000000,000001,... 对应

    Args:
        annotations_dir: 存放 *.xml 的目录
        output_main_dir: 输出目录（如 ImageSets/Main），将写入 class_trainval.txt 和 images_list.txt
        image_ext: 图片扩展名，用于从 XML 的 filename 中取 stem
        split: 文件名后缀，如 'trainval' -> xxx_trainval.txt

    Returns:
        stems: 有序的图片 stem 列表（与 000000, 000001, ... 一一对应）
        classes: 类别名列表
    """
    import xml.etree.ElementTree as ET

    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"标注目录不存在: {annotations_dir}")

    stem_to_classes = {}
    for f in os.listdir(annotations_dir):
        if not f.endswith('.xml'):
            continue
        xml_path = os.path.join(annotations_dir, f)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = root.find('filename')
            if filename is None or not filename.text:
                continue
            stem = os.path.splitext(filename.text.strip())[0]
            names = set()
            for obj in root.findall('object'):
                name_el = obj.find('name')
                if name_el is not None and name_el.text:
                    names.add(name_el.text.strip())
            stem_to_classes[stem] = names
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue

    if not stem_to_classes:
        raise FileNotFoundError(f"未在 {annotations_dir} 中找到有效 XML")

    stems = sorted(stem_to_classes.keys(), key=_natural_sort_key)
    all_classes = set()
    for names in stem_to_classes.values():
        all_classes.update(names)
    classes = sorted(all_classes)

    n = len(stems)
    Y = np.zeros((n, len(classes)), dtype=np.int32)
    for i, stem in enumerate(stems):
        for j, c in enumerate(classes):
            Y[i, j] = 1 if c in stem_to_classes[stem] else -1

    os.makedirs(output_main_dir, exist_ok=True)
    for j, c in enumerate(classes):
        safe_name = c.replace(' ', '_')
        fname = os.path.join(output_main_dir, f"{safe_name}_{split}.txt")
        with open(fname, 'w') as f:
            for i in range(n):
                f.write(f"{i:06d}  {Y[i, j]}\n")
        print(f"Wrote {fname}")

    images_list_path = os.path.join(output_main_dir, "images_list.txt")
    with open(images_list_path, 'w') as f:
        for stem in stems:
            f.write(stem + "\n")
    print(f"Wrote {images_list_path} (n={len(stems)})")
    return stems, classes


def load_face_mask_data(
    imgs_dir: str,
    lbls_dir: str,
    img_size: tuple,
    split: str = 'trainval',
    classes: list = None,
    exclude_unclassified: bool = True,
    N: int = int(1e6),
    dtype=np.float32,
    image_ext: str = '.png',
) -> tuple:
    """
    从 VOC 风格的 ImageSets/Main 加载 Face Mask 数据（需先调用 build_face_mask_main 生成标签）。

    Args:
        imgs_dir: 图片目录，内含 maksssksksss99.png 等（与 images_list.txt 中的 stem + image_ext 对应）
        lbls_dir: Main 目录，内含 xxx_trainval.txt 和 images_list.txt
        img_size: (H, W)
        split: 与 build_face_mask_main 的 split 一致，如 'trainval'
        classes: 类别列表，None 则从 lbls_dir 中 *_trainval.txt 推断
        exclude_unclassified: 是否排除“所有类都为 -1”的样本
        N: 最多加载样本数
        image_ext: 图片扩展名，如 '.png'

    Returns:
        X: (N_actual, H, W, 3)
        Y: (N_actual, n_classes)，每类 1/0
        cls_list: 类别名列表
    """
    from PIL import Image

    if not os.path.exists(imgs_dir):
        raise FileNotFoundError(f"图像目录不存在: {imgs_dir}")
    if not os.path.exists(lbls_dir):
        raise FileNotFoundError(f"标签目录不存在: {lbls_dir}")

    images_list_path = os.path.join(lbls_dir, "images_list.txt")
    if not os.path.exists(images_list_path):
        raise FileNotFoundError(
            f"缺少 {images_list_path}，请先运行 build_face_mask_main 生成 Main 与 images_list.txt"
        )
    with open(images_list_path, 'r') as f:
        stems = [line.strip() for line in f if line.strip()]
    stems = stems[:N]
    n = len(stems)

    if classes is not None:
        cls_list = list(classes)
    else:
        cls_list = []
        suffix = f"_{split}.txt"
        for f in os.listdir(lbls_dir):
            if f == "images_list.txt" or not f.endswith('.txt'):
                continue
            if f.endswith(suffix):
                base = f[:-len(suffix)]  # 类别名可能含下划线，如 with_mask
                cls_list.append(base.replace('_', ' '))
        cls_list = sorted(set(cls_list))

    M = len(cls_list)
    Y = np.zeros((n, M), dtype=np.float32)
    for j, c in enumerate(cls_list):
        safe_name = c.replace(' ', '_')
        fpath = os.path.join(lbls_dir, f"{safe_name}_{split}.txt")
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                    val = int(parts[1])
                    if idx < n:
                        Y[idx, j] = 1 if val == 1 else 0
                except ValueError:
                    continue

    if exclude_unclassified:
        mask = np.any(Y == 1, axis=1)
        Y = Y[mask]
        stems = [s for i, s in enumerate(stems) if mask[i]]
        n = len(stems)
        print(f"After excluding unclassified: {n} samples")
    else:
        print(f"Loaded {n} samples")

    X = np.zeros((n, img_size[0], img_size[1], 3), dtype=dtype)
    for i, stem in enumerate(stems):
        fname = stem + image_ext
        img_path = os.path.join(imgs_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB').resize(img_size)
            X[i] = np.array(img, dtype=dtype)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return X, Y, cls_list