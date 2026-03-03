"""
将 mask_dataset 下的图片与标签统一命名为 img_000001, img_000002, ...
便于 LoadData 等代码按固定格式处理。
"""
import os

MASK_DATASET = os.path.join(os.path.dirname(__file__), "mask_dataset")
SPLITS = ("train", "test")
IMG_DIR = "images"
LABEL_DIR = "labels"
IMG_EXT = ".jpg"
LABEL_EXT = ".txt"
TMP_PREFIX = "__tmp_"
NEW_PREFIX = "img_"
PAD = 6


def main():
    for split in SPLITS:
        img_dir = os.path.join(MASK_DATASET, IMG_DIR, split)
        label_dir = os.path.join(MASK_DATASET, LABEL_DIR, split)
        if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
            print(f"Skip {split}: dirs not found")
            continue

        # 只保留同时存在图片和标签的样本，按名称排序
        bases = []
        for f in os.listdir(img_dir):
            if not f.lower().endswith(IMG_EXT):
                continue
            base = f[: -len(IMG_EXT)]
            label_path = os.path.join(label_dir, base + LABEL_EXT)
            if os.path.isfile(label_path):
                bases.append(base)
            else:
                print(f"  [skip] no label for image: {base}{IMG_EXT}")

        bases.sort()
        n = len(bases)
        if n == 0:
            print(f"{split}: no pairs found")
            continue

        print(f"{split}: renaming {n} image-label pairs to {NEW_PREFIX}XXXXXX ...")

        # 第一阶段：重命名为临时名，避免新名与旧名冲突
        for i, base in enumerate(bases):
            idx = i + 1
            tmp_base = f"{TMP_PREFIX}{idx:0{PAD}d}"
            old_img = os.path.join(img_dir, base + IMG_EXT)
            old_label = os.path.join(label_dir, base + LABEL_EXT)
            tmp_img = os.path.join(img_dir, tmp_base + IMG_EXT)
            tmp_label = os.path.join(label_dir, tmp_base + LABEL_EXT)
            os.rename(old_img, tmp_img)
            os.rename(old_label, tmp_label)

        # 第二阶段：临时名改为最终名
        for i in range(n):
            idx = i + 1
            tmp_base = f"{TMP_PREFIX}{idx:0{PAD}d}"
            new_base = f"{NEW_PREFIX}{idx:0{PAD}d}"
            tmp_img = os.path.join(img_dir, tmp_base + IMG_EXT)
            tmp_label = os.path.join(label_dir, tmp_base + LABEL_EXT)
            new_img = os.path.join(img_dir, new_base + IMG_EXT)
            new_label = os.path.join(label_dir, new_base + LABEL_EXT)
            os.rename(tmp_img, new_img)
            os.rename(tmp_label, new_label)

        print(f"  -> {new_base}{IMG_EXT} (and .txt) range: {NEW_PREFIX}000001 .. {NEW_PREFIX}{n:0{PAD}d}")

    print("Done.")


if __name__ == "__main__":
    main()
