#!/bin/bash

cd /home/add_disk2/qiuxingyu/mmdetection

echo "========================================"
echo "开始训练 Mask R-CNN 模型..."
echo "========================================"

# 多卡训练 Mask R-CNN 模型
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py \
    --work-dir work_dirs/mask_rcnn_voc \
    --launcher pytorch \
    --cfg-options log_level=INFO \
    --auto-scale-lr

# 检查 Mask R-CNN 训练是否成功
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Mask R-CNN 训练完成！"
    echo "========================================"
else
    echo "========================================"
    echo "Mask R-CNN 训练失败，退出脚本！"
    echo "========================================"
    exit 1
fi

# 等待一段时间，让 GPU 冷却
echo "等待 30 秒后开始下一个实验..."
sleep 30

echo "========================================"
echo "开始训练 Sparse R-CNN 模型..."
echo "========================================"

# 多卡训练 Sparse R-CNN 模型
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py \
    --work-dir work_dirs/sparse_rcnn_voc \
    --launcher pytorch \
    --cfg-options log_level=INFO \
    --auto-scale-lr

# 检查 Sparse R-CNN 训练是否成功
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Sparse R-CNN 训练完成！"
    echo "========================================"
else
    echo "========================================"
    echo "Sparse R-CNN 训练失败！"
    echo "========================================"
    exit 1
fi

echo "========================================"
echo "所有实验已完成！"
echo "========================================"

# 训练完成后运行测试
echo "开始测试模型..."

# 测试 Mask R-CNN 模型
echo "测试 Mask R-CNN 模型..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/test.py \
    configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py \
    work_dirs/mask_rcnn_voc/epoch_4.pth \
    --work-dir work_dirs/mask_rcnn_voc \
    --show-dir work_dirs/mask_rcnn_voc/vis \
    --launcher pytorch \
    --cfg-options log_level=INFO

# 测试 Sparse R-CNN 模型
echo "测试 Sparse R-CNN 模型..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/test.py \
    configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py \
    work_dirs/sparse_rcnn_voc/epoch_4.pth \
    --work-dir work_dirs/sparse_rcnn_voc \
    --show-dir work_dirs/sparse_rcnn_voc/vis \
    --launcher pytorch \
    --cfg-options log_level=INFO

echo "========================================"
echo "所有测试已完成！"
echo "========================================" 