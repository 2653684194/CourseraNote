"""
将 mask_dataset 下的图片与标签统一命名为 img_000001, img_000002, ...
便于 LoadData 等代码按固定格式处理。
"""
import os

MASK_DATASET = os.path.join(os.path.dirname(__file__), "mask_dataset")
SPLITS = ("train", "test")
IMG_DIR = "images"
LABEL_DIR = "labels"
IMG_EXTS = ['.jpg', '.jpeg', '.png']
LABEL_EXT = ".txt"
TMP_PREFIX = "__tmp_rename__"
NEW_PREFIX = "img_"
PAD = 6


def main():
    for split in SPLITS:
        img_dir = os.path.join(MASK_DATASET, IMG_DIR, split)
        label_dir = os.path.join(MASK_DATASET, LABEL_DIR, split)
        if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
            print(f"Skip {split}: dirs not found")
            continue

        # 只保留同时存在图片和标签的样本，按文件名排序
        pairs = []
        for f in sorted(os.listdir(img_dir)):
            if not any(f.lower().endswith(ext) for ext in IMG_EXTS):
                continue
            base = f.rsplit('.', 1)[0]
            img_ext = '.' + f.rsplit('.', 1)[1].lower()
            label_path = os.path.join(label_dir, base + LABEL_EXT)
            if os.path.isfile(label_path):
                pairs.append((base, img_ext))
            else:
                print(f"  [skip] no label for image: {base}{img_ext}")

        n = len(pairs)
        if n == 0:
            print(f"{split}: no pairs found")
            continue

        print(f"{split}: renaming {n} image-label pairs to {NEW_PREFIX}XXXXXX ...")

        # 第一阶段：重命名为临时名，避免新名与旧名冲突
        for i, (base, img_ext) in enumerate(pairs):
            idx = i + 1
            tmp_base = f"{TMP_PREFIX}{idx:0{PAD}d}"
            old_img = os.path.join(img_dir, base + img_ext)
            old_label = os.path.join(label_dir, base + LABEL_EXT)
            tmp_img = os.path.join(img_dir, tmp_base + img_ext)
            tmp_label = os.path.join(label_dir, tmp_base + LABEL_EXT)
            os.rename(old_img, tmp_img)
            os.rename(old_label, tmp_label)

        # 第二阶段：临时名改为最终名
        for i, (_, img_ext) in enumerate(pairs):
            idx = i + 1
            tmp_base = f"{TMP_PREFIX}{idx:0{PAD}d}"
            new_base = f"{NEW_PREFIX}{idx:0{PAD}d}"
            tmp_img = os.path.join(img_dir, tmp_base + img_ext)
            tmp_label = os.path.join(label_dir, tmp_base + LABEL_EXT)
            new_img = os.path.join(img_dir, new_base + img_ext)
            new_label = os.path.join(label_dir, new_base + LABEL_EXT)
            os.rename(tmp_img, new_img)
            os.rename(tmp_label, new_label)

        print(f"  -> {NEW_PREFIX}000001 (and .txt) .. {NEW_PREFIX}{n:0{PAD}d}")

    print("Done.")


if __name__ == "__main__":
    main()
