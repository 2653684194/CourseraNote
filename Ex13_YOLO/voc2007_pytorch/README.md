# VOC 2007 PyTorch (Backbone + Multi-label Head)

本目录为**独立项目**，不修改工作空间原有文件。使用 `archive_2007/VOCtrainval_06-Nov-2007` 数据，PyTorch 复刻 ML_Note13_2 的 backbone（3 个 ResBlock）+ 自设计的 VOC 多标签分类头，完成训练与可视化。

## 数据

- 路径：`../archive_2007/VOCtrainval_06-Nov-2007`（ImageSets/Main/trainval.txt + Annotations + JPEGImages）
- 任务：20 类多标签分类（PASCAL VOC 类别）

## 结构

- `config.py`：数据路径、类别、训练超参
- `dataset.py`：VOC 数据集（解析 XML，返回图像 + 20 维 0/1 标签）
- `model.py`：Backbone（3×ResBlock，与 notebook 骨架一致）+ GAP + FC(20) + sigmoid
- `train.py`：训练循环，保存最佳/最新 checkpoint 与 `history.json`
- `visualize.py`：绘制 loss/accuracy 曲线，并展示部分样本的预测 vs 真实标签

## 运行

```bash
cd voc2007_pytorch
pip install -r requirements.txt   # 如未安装 PyTorch
python train.py                    # 训练
python visualize.py                # 生成 visualizations/train_curves.png 与 predictions.png
```

训练结果与可视化均保存在本目录下 `checkpoints/` 与 `visualizations/`，不修改上级目录任何文件。
