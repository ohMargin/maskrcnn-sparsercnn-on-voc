_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]

# 修改类别数为 VOC 的 20 类
model = dict(
    # 使用 COCO 预训练的 Mask R-CNN 模型
    init_cfg=dict(type='Pretrained', checkpoint='checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'),
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)))

# 数据集配置
dataset_type = 'CocoDataset'
data_root = 'data/coco/'  # 修改为您的VOC COCO格式数据路径
backend_args = None

# 训练管道
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 测试管道
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 数据加载器配置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',  # 确保这个文件名与您转换后的文件一致
        data_prefix=dict(img='train2017/'),  # 确保这个路径与您的数据结构一致
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',  # 确保这个文件名与您转换后的文件一致
        data_prefix=dict(img='val2017/'),  # 确保这个路径与您的数据结构一致
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',  # 确保这个文件名与您转换后的文件一致
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# 训练计划
max_epochs = 40
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 优化器 - 降低学习率以更好地进行迁移学习
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))  # 学习率从0.02降至0.005

# 默认设置，用于自动缩放学习率
auto_scale_lr = dict(enable=False, base_batch_size=16) 