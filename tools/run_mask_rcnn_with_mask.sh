#!/bin/bash

cd /home/add_disk2/qiuxingyu/mmdetection

echo "========================================"
echo "开始训练 Mask R-CNN 模型（带实例分割）..."
echo "========================================"

# 多卡训练 Mask R-CNN 模型
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc_coco-format.py \
    --work-dir work_dirs/mask_rcnn_voc_with_mask \
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

echo "========================================"
echo "训练已完成！"
echo "========================================"

# 训练完成后运行测试
echo "开始测试模型..."

# 测试 Mask R-CNN 模型
echo "测试 Mask R-CNN 模型..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/test.py \
    configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc_coco-format.py \
    work_dirs/mask_rcnn_voc_with_mask/epoch_12.pth \
    --work-dir work_dirs/mask_rcnn_voc_with_mask \
    --show-dir work_dirs/mask_rcnn_voc_with_mask/vis \
    --launcher pytorch \
    --cfg-options log_level=INFO

echo "========================================"
echo "测试已完成！"
echo "========================================"

# 可视化 proposal boxes 和实例分割结果
echo "可视化 proposal boxes 和实例分割结果..."
python tools/visualize_mask_rcnn_proposals_and_masks.py

echo "========================================"
echo "所有操作已完成！"
echo "========================================" 