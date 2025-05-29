#!/bin/bash

cd /home/add_disk2/qiuxingyu/mmdetection

# # 激活环境
# source activate openmm

# 多卡训练 Sparse R-CNN 模型
# 使用 torch.distributed.launch 启动多卡训练
# 可以根据实际情况修改 --nproc_per_node 参数，设置为您想使用的 GPU 数量
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py \
    --work-dir work_dirs/sparse_rcnn_voc \
    --launcher pytorch \
    --cfg-options log_level=INFO \
    --auto-scale-lr 