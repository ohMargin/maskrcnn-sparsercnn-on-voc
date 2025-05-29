#!/bin/bash

# 确保预训练模型已下载
CHECKPOINT_DIR="checkpoints"
mkdir -p $CHECKPOINT_DIR

PRETRAINED_MODEL="$CHECKPOINT_DIR/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "下载预训练模型..."
    wget -P $CHECKPOINT_DIR https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
fi

# 使用分布式训练
CONFIG="configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc_coco-format.py"
GPUS=2
NNODES=1
NODE_RANK=0
PORT=29500
MASTER_ADDR="127.0.0.1"

# 确保配置文件中的预训练模型路径正确
sed -i "s|https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth|$PRETRAINED_MODEL|g" $CONFIG

# 启动训练
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch \
    --work-dir work_dirs/mask_rcnn_voc_coco_format 