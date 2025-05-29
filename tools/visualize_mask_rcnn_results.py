import os
import cv2
import numpy as np
import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Mask R-CNN results')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file path')
    parser.add_argument('--img', help='Image file path')
    parser.add_argument('--out-dir', default='visualization', help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载模型
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # 初始化可视化器
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    # 读取图像
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    
    # 进行推理
    result = inference_detector(model, args.img)
    
    # 可视化结果
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=os.path.join(args.out_dir, 'detection_result.jpg'),
        pred_score_thr=args.score_thr
    )
    
    # 分别可视化边界框和分割掩码
    # 1. 只显示边界框
    result_boxes = result.clone()
    if hasattr(result_boxes.pred_instances, 'masks'):
        result_boxes.pred_instances.masks = None
    
    visualizer.add_datasample(
        'boxes_only',
        img,
        data_sample=result_boxes,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=os.path.join(args.out_dir, 'boxes_only.jpg'),
        pred_score_thr=args.score_thr
    )
    
    # 2. 只显示分割掩码
    if hasattr(result.pred_instances, 'masks') and result.pred_instances.masks is not None:
        result_masks = result.clone()
        result_masks.pred_instances.bboxes = None
        
        visualizer.add_datasample(
            'masks_only',
            img,
            data_sample=result_masks,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(args.out_dir, 'masks_only.jpg'),
            pred_score_thr=args.score_thr
        )
    
    print(f'可视化结果已保存到 {args.out_dir} 目录')

if __name__ == '__main__':
    main() 