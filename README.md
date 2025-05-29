# VOC 数据集上的 Mask R-CNN 和 Sparse R-CNN 实验

本项目基于 MMDetection 框架，在 VOC 数据集上训练和测试 Mask R-CNN 和 Sparse R-CNN 模型。

## 环境配置

本项目依赖于 MMDetection 框架，请确保已经正确安装了 MMDetection 及其依赖项。

```bash
# 激活环境
conda activate openmm
```

## 数据集准备

本项目使用 VOC2007 和 VOC2012 数据集。数据集应该放在 `mmdetection/data/VOCdevkit` 目录下，目录结构如下：

```
mmdetection/data/VOCdevkit/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── SegmentationObject/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/
    ├── JPEGImages/
    ├── SegmentationClass/
    └── SegmentationObject/
```

## 训练模型

### 训练 Mask R-CNN

```bash
bash tools/train_mask_rcnn_voc.sh
```

### 训练 Sparse R-CNN

```bash
bash tools/train_sparse_rcnn_voc.sh
```

## 测试模型

```bash
bash tools/test_models_voc.sh
```

## 可视化结果

### 可视化 Mask R-CNN 的 proposal boxes 和最终预测结果

```bash
python tools/visualize_mask_rcnn_proposals.py
```

### 可视化 Mask R-CNN 和 Sparse R-CNN 在外部图像上的检测结果

首先，准备三张包含 VOC 类别物体的外部图像，并将它们放在 `external_images` 目录下，命名为 `image1.jpg`、`image2.jpg` 和 `image3.jpg`。

```bash
mkdir -p external_images
# 将外部图像放入 external_images 目录
```

然后运行可视化脚本：

```bash
python tools/visualize_external_images.py
```

### 生成 TensorBoard 可视化

```bash
python tools/visualize_tensorboard.py
```

## 实验设置

- **数据集划分**：使用 VOC2007 和 VOC2012 的 trainval 集合作为训练集，VOC2007 的 test 集合作为测试集
- **网络结构**：
  - Mask R-CNN：使用 ResNet-50 作为骨干网络，FPN 作为特征金字塔网络
  - Sparse R-CNN：使用 ResNet-50 作为骨干网络，FPN 作为特征金字塔网络，6 个级联阶段，100 个提议框
- **Batch Size**：2
- **Learning Rate**：
  - Mask R-CNN：0.01
  - Sparse R-CNN：0.000025
- **优化器**：
  - Mask R-CNN：SGD，动量 0.9，权重衰减 0.0001
  - Sparse R-CNN：AdamW，权重衰减 0.0001
- **训练轮次**：4 个 epoch（由于数据集重复了 3 次，实际上是 12 个 epoch）
- **损失函数**：
  - Mask R-CNN：分类损失（CrossEntropyLoss），边界框回归损失（L1Loss），掩码损失（CrossEntropyLoss）
  - Sparse R-CNN：分类损失（FocalLoss），边界框回归损失（L1Loss），IoU 损失（GIoULoss），掩码损失（CrossEntropyLoss）
- **评价指标**：mAP（使用 VOC 评价模式：11 points）

## 模型权重

训练好的模型权重可以在以下位置找到：

- Mask R-CNN：`work_dirs/mask_rcnn_voc/epoch_4.pth`
- Sparse R-CNN：`work_dirs/sparse_rcnn_voc/epoch_4.pth`

## 可视化结果

可视化结果保存在以下目录：

- Mask R-CNN proposal boxes 和最终预测结果：`work_dirs/mask_rcnn_voc/proposal_vis`
- 外部图像的检测结果：`work_dirs/external_images_vis`
- TensorBoard 可视化：`work_dirs/tensorboard_vis` 