#!/bin/bash

# 配置文件和检查点路径
CONFIG="configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc_coco-format.py"
CHECKPOINT="work_dirs/mask_rcnn_voc_coco_format/epoch_12.pth"  # 根据实际训练的最终检查点调整

# 使用单卡测试
GPUS=1
NNODES=1
NODE_RANK=0
PORT=29501
MASTER_ADDR="127.0.0.1"

# 启动测试
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --work-dir work_dirs/mask_rcnn_voc_coco_format_eval \
    --show-dir work_dirs/mask_rcnn_voc_coco_format_eval/visualization \
    --show 