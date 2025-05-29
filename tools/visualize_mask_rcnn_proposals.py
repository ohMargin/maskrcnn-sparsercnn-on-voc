import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import cv2

def main():
    # 配置文件和检查点文件路径
    config_file = 'configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py'
    checkpoint_file = 'work_dirs/mask_rcnn_voc/epoch_4.pth'
    
    # 输出目录
    output_dir = 'work_dirs/mask_rcnn_voc/proposal_vis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置文件
    cfg = Config.fromfile(config_file)
    
    # 初始化模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # 获取测试数据集中的图像
    test_dataset = cfg.test_dataloader.dataset
    data_root = test_dataset.data_root
    ann_file = test_dataset.ann_file
    
    # 读取测试集图像列表
    with open(os.path.join(data_root, ann_file), 'r') as f:
        img_ids = [line.strip() for line in f.readlines()]
    
    # 选择4张图像进行可视化
    selected_img_ids = img_ids[:4]
    
    # 初始化可视化器
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    for img_id in selected_img_ids:
        # 构建图像路径
        img_path = os.path.join(data_root, 'VOC2007/JPEGImages', f'{img_id}.jpg')
        
        # 读取图像
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        # 推理
        result = inference_detector(model, img)
        
        # 可视化最终预测结果
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, f'{img_id}_final.jpg'),
            pred_score_thr=0.3
        )
        
        # 可视化 proposal boxes
        # 获取 proposal boxes
        # 注意：这里需要修改模型的前向传播过程来获取 proposal boxes
        # 这里我们使用一个简单的方法：直接从模型的 rpn_head 中获取 proposals
        
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
        
        print(f'Processed image {img_id}')

if __name__ == '__main__':
    main() 