import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

def main():
    # 配置文件和检查点文件路径
    mask_rcnn_config = 'configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py'
    mask_rcnn_checkpoint = 'work_dirs/mask_rcnn_voc/epoch_4.pth'
    
    sparse_rcnn_config = 'configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py'
    sparse_rcnn_checkpoint = 'work_dirs/sparse_rcnn_voc/epoch_4.pth'
    
    # 输出目录
    output_dir = 'work_dirs/external_images_vis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    mask_rcnn_model = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cuda:0')
    sparse_rcnn_model = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')
    
    # 加载配置文件
    mask_rcnn_cfg = Config.fromfile(mask_rcnn_config)
    sparse_rcnn_cfg = Config.fromfile(sparse_rcnn_config)
    
    # 初始化可视化器
    mask_rcnn_visualizer = VISUALIZERS.build(mask_rcnn_cfg.visualizer)
    mask_rcnn_visualizer.dataset_meta = mask_rcnn_model.dataset_meta
    
    sparse_rcnn_visualizer = VISUALIZERS.build(sparse_rcnn_cfg.visualizer)
    sparse_rcnn_visualizer.dataset_meta = sparse_rcnn_model.dataset_meta
    
    # 下载三张包含 VOC 类别的外部图像
    # 这里我们假设已经有三张图像，路径为 external_images/image1.jpg, external_images/image2.jpg, external_images/image3.jpg
    # 如果没有，你需要手动下载或准备这些图像
    external_images_dir = 'external_images'
    os.makedirs(external_images_dir, exist_ok=True)
    
    # 图像路径列表
    image_paths = [
        os.path.join(external_images_dir, 'image1.jpg'),
        os.path.join(external_images_dir, 'image2.jpg'),
        os.path.join(external_images_dir, 'image3.jpg')
    ]
    
    # 检查图像是否存在，如果不存在，提示用户准备图像
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"请准备外部图像文件: {img_path}")
    
    # 对每张图像进行推理和可视化
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            continue
            
        # 读取图像
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        # Mask R-CNN 推理
        mask_rcnn_result = inference_detector(mask_rcnn_model, img)
        
        # Sparse R-CNN 推理
        sparse_rcnn_result = inference_detector(sparse_rcnn_model, img)
        
        # 可视化 Mask R-CNN 结果
        mask_rcnn_visualizer.add_datasample(
            'result',
            img,
            data_sample=mask_rcnn_result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, f'image{i+1}_mask_rcnn.jpg'),
            pred_score_thr=0.3
        )
        
        # 可视化 Sparse R-CNN 结果
        sparse_rcnn_visualizer.add_datasample(
            'result',
            img,
            data_sample=sparse_rcnn_result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, f'image{i+1}_sparse_rcnn.jpg'),
            pred_score_thr=0.3
        )
        
        print(f'Processed external image {i+1}')

if __name__ == '__main__':
    main() 