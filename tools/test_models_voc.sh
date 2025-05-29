#!/bin/bash

cd /home/add_disk2/qiuxingyu/mmdetection

# # 激活环境
# source activate openmm

# 测试 Mask R-CNN 模型
echo "Testing Mask R-CNN model..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/test.py \
    configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py \
    work_dirs/mask_rcnn_voc/epoch_4.pth \
    --work-dir work_dirs/mask_rcnn_voc \
    --show-dir work_dirs/mask_rcnn_voc/vis \
    --launcher pytorch \
    --cfg-options log_level=INFO

# 测试 Sparse R-CNN 模型
echo "Testing Sparse R-CNN model..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/test.py \
    configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py \
    work_dirs/sparse_rcnn_voc/epoch_4.pth \
    --work-dir work_dirs/sparse_rcnn_voc \
    --show-dir work_dirs/sparse_rcnn_voc/vis \
    --launcher pytorch \
    --cfg-options log_level=INFO 