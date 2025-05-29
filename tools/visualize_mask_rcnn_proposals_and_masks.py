import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import torch
from mmengine.config import Config
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import cv2

def main():
    # 配置文件和检查点文件路径
    config_file = 'configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc_coco-format.py'
    checkpoint_file = 'work_dirs/mask_rcnn_voc_coco_format/epoch_40.pth'
    
    # 输出目录
    output_dir = 'work_dirs/mask_rcnn_voc_coco_format/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置文件
    cfg = Config.fromfile(config_file)
    
    # 初始化模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # 获取测试数据集中的图像
    test_dataset = cfg.test_dataloader.dataset
    data_root = test_dataset.data_root
    ann_file = test_dataset.ann_file
    
    # 初始化可视化器
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    # 从测试集中选择4张图像
    import json
    with open(os.path.join(data_root, ann_file), 'r') as f:
        coco_data = json.load(f)
    
    # 随机选择4张图像
    import random
    random.seed(42)  # 设置随机种子以确保可重复性
    selected_images = random.sample(coco_data['images'], 4)
    
    for img_info in selected_images:
        img_id = img_info['id']
        img_path = os.path.join(data_root, test_dataset.data_prefix['img'], img_info['file_name'])
        
        # 读取图像
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        # 推理
        result = inference_detector(model, img)
        
        # 可视化最终预测结果（包括边界框）
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, f'{img_id}_bbox.jpg'),
            pred_score_thr=0.3
        )
        
        # 可视化实例分割结果
        # 创建只有掩码没有边界框的结果副本
        mask_result = result.clone()
        if hasattr(mask_result.pred_instances, 'bboxes'):
            # 使用空张量代替None，保持序列类型
            mask_result.pred_instances.bboxes = torch.zeros((0, 4), device=mask_result.pred_instances.bboxes.device)
        
        visualizer.add_datasample(
            'result',
            img,
            data_sample=mask_result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, f'{img_id}_mask.jpg'),
            pred_score_thr=0.3
        )
        
        # 可视化 proposal boxes
        # 创建输入数据
        data = dict(
            inputs=img,
            data_samples=[result]
        )
        
        # 获取特征图
        x = model.extract_feat(model.data_preprocessor(data)['inputs'])
        
        # 获取 proposal boxes
        rpn_results_list = model.rpn_head.predict(x, [result])
        
        # 可视化 proposal boxes
        img_show = img.copy()
        for rpn_results in rpn_results_list:
            bboxes = rpn_results.bboxes
            scores = rpn_results.scores
            
            # 只保留前 100 个 proposal boxes
            keep_idxs = scores.argsort(descending=True)[:100]
            bboxes = bboxes[keep_idxs]
            
            # 绘制 proposal boxes
            for bbox in bboxes:
                bbox_int = bbox.int().tolist()
                cv2.rectangle(img_show, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0, 255, 0), 1)
        
        # 保存图像
        mmcv.imwrite(mmcv.rgb2bgr(img_show), os.path.join(output_dir, f'{img_id}_proposals.jpg'))
        
        # 创建对比图：左边是 proposals，右边是最终检测结果
        # 读取保存的图像
        proposals_img = mmcv.imread(os.path.join(output_dir, f'{img_id}_proposals.jpg'))
        bbox_img = mmcv.imread(os.path.join(output_dir, f'{img_id}_bbox.jpg'))
        mask_img = mmcv.imread(os.path.join(output_dir, f'{img_id}_mask.jpg'))
        
        # 确保所有图像具有相同的大小
        h, w = proposals_img.shape[:2]
        bbox_img = cv2.resize(bbox_img, (w, h))
        mask_img = cv2.resize(mask_img, (w, h))
        
        # 创建对比图
        comparison_img = np.hstack([proposals_img, bbox_img])
        mmcv.imwrite(comparison_img, os.path.join(output_dir, f'{img_id}_comparison_proposals_bbox.jpg'))
        
        # 创建边界框和掩码的对比图
        comparison_img = np.hstack([bbox_img, mask_img])
        mmcv.imwrite(comparison_img, os.path.join(output_dir, f'{img_id}_comparison_bbox_mask.jpg'))
        
        print(f'Processed image {img_id}')

if __name__ == '__main__':
    main() 