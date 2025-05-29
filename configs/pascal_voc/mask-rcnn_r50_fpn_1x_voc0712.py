_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]

# 修改类别数为 VOC 的 20 类，并禁用掩码分支
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=None,  # 禁用掩码头
        mask_roi_extractor=None))  # 禁用掩码 ROI 提取器

# 训练计划，VOC 数据集在 `_base_/datasets/voc0712.py` 中重复了 3 次，
# 所以实际 epoch = 4 * 3 = 12
max_epochs = 10
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 学习率调度
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# 默认设置，用于自动缩放学习率
auto_scale_lr = dict(enable=False, base_batch_size=16)

# 修改 VOC 数据集的训练管道
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(dataset=dict(dataset=dict(
    datasets=[
        dict(
            type='VOCDataset',
            data_root='data/VOCdevkit/',
            ann_file='VOC2007/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
            pipeline=train_pipeline,
            backend_args=None),
        dict(
            type='VOCDataset',
            data_root='data/VOCdevkit/',
            ann_file='VOC2012/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2012/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
            pipeline=train_pipeline,
            backend_args=None)
    ]
)))

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline)) 