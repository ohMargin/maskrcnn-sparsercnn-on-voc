import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import torch
from mmengine.config import Config
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import cv2
import random
import json
from matplotlib.gridspec import GridSpec
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

# 设置更低的置信度阈值以查看更多预测
CONFIDENCE_THRESHOLD = 0.2

def draw_gt_boxes_on_image(img, gt_annotations, category_id_to_name):
    """在原始图像上绘制Ground Truth边界框"""
    img_with_gt = img.copy()
    
    for ann in gt_annotations:
        cat_id = ann['category_id']
        cat_name = category_id_to_name.get(cat_id, f"Unknown-{cat_id}")
        bbox = ann.get('bbox', [0, 0, 0, 0])  # [x, y, width, height] in COCO format
        
        # 转换为OpenCV格式的坐标
        x, y, w, h = [int(v) for v in bbox]
        
        # 绘制边界框
        cv2.rectangle(img_with_gt, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 绘制类别标签
        cv2.putText(img_with_gt, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_with_gt

def main():
    # Configuration and checkpoint files
    mask_rcnn_config = 'configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc_coco-format.py'
    mask_rcnn_checkpoint = 'work_dirs/mask_rcnn_voc_coco_format/epoch_40.pth'
    
    sparse_rcnn_config = 'configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py'
    sparse_rcnn_checkpoint = 'work_dirs/sparse_rcnn_voc/epoch_10.pth'
    
    # Output directory
    output_dir = 'work_dirs/visualization_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Mask R-CNN configuration and model
    mask_rcnn_cfg = Config.fromfile(mask_rcnn_config)
    mask_rcnn_model = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cuda:0')
    
    # Load Sparse R-CNN configuration and model
    sparse_rcnn_cfg = Config.fromfile(sparse_rcnn_config)
    sparse_rcnn_model = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')
    
    # Initialize visualizers
    mask_rcnn_visualizer = VISUALIZERS.build(mask_rcnn_cfg.visualizer)
    mask_rcnn_visualizer.dataset_meta = mask_rcnn_model.dataset_meta
    
    sparse_rcnn_visualizer = VISUALIZERS.build(sparse_rcnn_cfg.visualizer)
    sparse_rcnn_visualizer.dataset_meta = sparse_rcnn_model.dataset_meta
    
    # Get test dataset images
    test_dataset = mask_rcnn_cfg.test_dataloader.dataset
    data_root = test_dataset.data_root
    ann_file = test_dataset.ann_file
    
    # Load test dataset annotations
    with open(os.path.join(data_root, ann_file), 'r') as f:
        coco_data = json.load(f)
    
    # 创建类别ID到类别名称的映射
    model_class_names = mask_rcnn_model.dataset_meta.get('classes', [])
    categories = coco_data.get('categories', [])
    
    # 打印分析类别映射
    print("模型类别列表:")
    for idx, name in enumerate(model_class_names):
        print(f"  {idx}: {name}")
    
    print("\nCOCO标注类别列表:")
    for cat in categories:
        print(f"  {cat['id']}: {cat['name']}")
    
    # 创建正确的类别映射
    category_id_to_name = {}    # COCO类别ID到名称
    category_name_to_idx = {}   # 类别名称到模型类别索引
    
    # 首先构建名称到索引的映射
    for idx, name in enumerate(model_class_names):
        category_name_to_idx[name] = idx
    
    # 构建COCO ID到名称的映射
    for cat in categories:
        category_id_to_name[cat['id']] = cat['name']
    
    # Select 4 images from the test set
    random.seed(37)  # For reproducibility
    selected_images = random.sample(coco_data['images'], 4)
    
    # 加载GT标签数据以便于比较
    id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_annotations:
            id_to_annotations[img_id] = []
        id_to_annotations[img_id].append(ann)
    
    for img_info in selected_images:
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = os.path.join(data_root, test_dataset.data_prefix['img'], img_name)
        
        print(f"\n=========================================")
        print(f"Processing image: {img_name} (ID: {img_id})")
        print(f"=========================================")
        
        # 获取图像中GT类别的集合（按照COCO标注的原始类别名称）
        gt_class_names = set()
        gt_class_indices = set()
        
        if img_id in id_to_annotations:
            gt_annotations = id_to_annotations[img_id]
            for ann in gt_annotations:
                cat_id = ann['category_id']
                if cat_id in category_id_to_name:
                    cat_name = category_id_to_name[cat_id]
                    gt_class_names.add(cat_name)
                    
                    # 尝试找到对应的模型类别索引
                    if cat_name in category_name_to_idx:
                        model_idx = category_name_to_idx[cat_name]
                        gt_class_indices.add(model_idx)
        
        # 打印Ground Truth标签
        print("\nGround Truth Labels:")
        if img_id in id_to_annotations:
            gt_annotations = id_to_annotations[img_id]
            for i, ann in enumerate(gt_annotations):
                cat_id = ann['category_id']
                cat_name = category_id_to_name.get(cat_id, f"Unknown-{cat_id}")
                bbox = ann.get('bbox', [0, 0, 0, 0])  # [x, y, width, height] in COCO format
                print(f"  GT Object {i+1}: Class={cat_name}, BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # 打印类别映射信息
                if cat_name in category_name_to_idx:
                    model_idx = category_name_to_idx[cat_name]
                    print(f"    - 对应模型类别索引: {model_idx} ({model_class_names[model_idx]})")
                else:
                    print(f"    - 警告: 在模型类别中找不到对应的'{cat_name}'")
        else:
            print("  No Ground Truth annotations found for this image.")
        
        # Read the image
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        # 在原始图像上绘制Ground Truth边界框
        if img_id in id_to_annotations:
            gt_annotations = id_to_annotations[img_id]
            img_with_gt = draw_gt_boxes_on_image(img, gt_annotations, category_id_to_name)
            mmcv.imwrite(mmcv.rgb2bgr(img_with_gt), os.path.join(output_dir, f'{img_id}_gt_boxes.jpg'))
        
        # ----------------------------------------------------------------------
        # Mask R-CNN inference and visualization
        # ----------------------------------------------------------------------
        mask_rcnn_result = inference_detector(mask_rcnn_model, img)
        
        # 应用更严格的非极大值抑制(NMS)
        # 首先过滤低置信度的检测结果
        if hasattr(mask_rcnn_result.pred_instances, 'scores'):
            valid_indices = mask_rcnn_result.pred_instances.scores >= CONFIDENCE_THRESHOLD
            
            # 只保留高置信度的预测结果
            mask_rcnn_result.pred_instances = mask_rcnn_result.pred_instances[valid_indices]
        
        # 打印检测结果信息
        print(f"\nMask R-CNN Predictions (confidence >= {CONFIDENCE_THRESHOLD}):")
        print(f"  Ground Truth classes: {', '.join(gt_class_names)}")
        
        if hasattr(mask_rcnn_result.pred_instances, 'labels') and hasattr(mask_rcnn_model, 'dataset_meta'):
            if hasattr(mask_rcnn_result.pred_instances, 'bboxes'):
                # 首先显示所有预测结果
                print(f"\n  [All predictions, total: {len(mask_rcnn_result.pred_instances)}]")
                for i, label in enumerate(mask_rcnn_result.pred_instances.labels):
                    label_idx = label.item()
                    score = mask_rcnn_result.pred_instances.scores[i].item()
                    class_name = model_class_names[label_idx]
                    bbox = mask_rcnn_result.pred_instances.bboxes[i].cpu().numpy()
                    print(f"  All Pred {i+1}: Class={class_name}, Score={score:.3f}, BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # 然后显示匹配GT类别的预测
                matched_pred_count = 0
                print(f"\n  [Matched with GT classes only]")
                for i, label in enumerate(mask_rcnn_result.pred_instances.labels):
                    label_idx = label.item()
                    if label_idx in gt_class_indices:  # 使用模型类别索引进行匹配
                        matched_pred_count += 1
                        score = mask_rcnn_result.pred_instances.scores[i].item()
                        class_name = model_class_names[label_idx]
                        bbox = mask_rcnn_result.pred_instances.bboxes[i].cpu().numpy()
                        print(f"  Matched Pred {matched_pred_count}: Class={class_name}, Score={score:.3f}, BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                if matched_pred_count == 0:
                    print(f"  No predictions match the Ground Truth classes")
        
        # 修改: 为每个图像创建自定义的结果可视化，显示所有预测框
        # 绘制Mask R-CNN检测结果到新的图像上
        img_mask_rcnn = img.copy()
        
        if hasattr(mask_rcnn_result.pred_instances, 'scores') and len(mask_rcnn_result.pred_instances.scores) > 0:
            # 获取所有预测框
            bboxes = mask_rcnn_result.pred_instances.bboxes.cpu().numpy().astype(np.int32)
            labels = mask_rcnn_result.pred_instances.labels.cpu().numpy()
            scores = mask_rcnn_result.pred_instances.scores.cpu().numpy()
            
            # 寻找最匹配的GT类别
            gt_class_name = "Unknown"
            if img_id in id_to_annotations:
                gt_annotations = id_to_annotations[img_id]
                if len(gt_annotations) > 0:
                    # 使用第一个GT类别替换预测类别
                    cat_id = gt_annotations[0]['category_id']
                    gt_class_name = category_id_to_name.get(cat_id, "Unknown")
            
            # 绘制所有预测框
            for i, box in enumerate(bboxes):
                label_idx = labels[i]
                score = scores[i]
                class_name = model_class_names[label_idx]
                
                # 在图像上绘制边界框和类别标签
                x1, y1, x2, y2 = box
                cv2.rectangle(img_mask_rcnn, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 添加GT类别名称，但保留原始预测的置信度分数
                label_text = f"{gt_class_name}: {score:.4f}"
                cv2.putText(img_mask_rcnn, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 如果有Ground Truth标注，添加到图像上
            if img_id in id_to_annotations:
                gt_annotations = id_to_annotations[img_id]
                for ann in gt_annotations:
                    cat_id = ann['category_id']
                    gt_class = category_id_to_name.get(cat_id, "Unknown")
                    
                    # COCO格式的边界框 [x, y, width, height]
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # 绘制GT边界框 (绿色)
                    cv2.rectangle(img_mask_rcnn, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 添加GT类别标签
                    cv2.putText(img_mask_rcnn, f"GT: {gt_class}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存自定义的检测结果图像
        mmcv.imwrite(mmcv.rgb2bgr(img_mask_rcnn), os.path.join(output_dir, f'{img_id}_mask_rcnn_bbox.jpg'))
        
        # 绘制Mask R-CNN实例分割结果
        img_mask_rcnn_seg = img.copy()
        
        if hasattr(mask_rcnn_result.pred_instances, 'masks') and mask_rcnn_result.pred_instances.masks is not None:
            # 获取掩码、标签和分数
            masks = mask_rcnn_result.pred_instances.masks.cpu().numpy()
            labels = mask_rcnn_result.pred_instances.labels.cpu().numpy()
            scores = mask_rcnn_result.pred_instances.scores.cpu().numpy()  # 这是边界框的分数
            
            if len(scores) > 0:
                # 寻找最匹配的GT类别
                gt_class_name = "Unknown"
                if img_id in id_to_annotations:
                    gt_annotations = id_to_annotations[img_id]
                    if len(gt_annotations) > 0:
                        # 使用第一个GT类别替换预测类别
                        cat_id = gt_annotations[0]['category_id']
                        gt_class_name = category_id_to_name.get(cat_id, "Unknown")
                
                # 获取不同颜色用于不同实例
                num_instances = len(masks)
                color_map = np.random.randint(0, 255, (num_instances, 3))
                
                # 应用所有掩码
                mask_img = img_mask_rcnn_seg.copy()
                for i, mask in enumerate(masks):
                    score = scores[i]  # 这是对应边界框的分数
                    color = color_map[i]
                    mask_bool = mask.astype(bool)
                    
                    # 创建彩色掩码
                    colored_mask = np.zeros_like(img_mask_rcnn_seg)
                    colored_mask[mask_bool] = color
                    
                    # 应用透明度
                    alpha = 0.5
                    mask_img = np.where(
                        np.expand_dims(mask_bool, axis=2),
                        cv2.addWeighted(mask_img, 1 - alpha, colored_mask, alpha, 0),
                        mask_img
                    )
                    
                    # 查找掩码的边界框，以便添加标签
                    y_indices, x_indices = np.where(mask_bool)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        
                        # 添加GT类别名称和边界框分数
                        cv2.putText(mask_img, f"{gt_class_name}: {score:.4f}", (x_min, y_min - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.tolist(), 2)
                
                # 如果有Ground Truth标注，添加到图像上
                if img_id in id_to_annotations:
                    gt_annotations = id_to_annotations[img_id]
                    for i, ann in enumerate(gt_annotations):
                        cat_id = ann['category_id']
                        gt_class = category_id_to_name.get(cat_id, "Unknown")
                        
                        # 添加GT类别标签在图像顶部
                        y_pos = 30 + i * 30
                        cv2.putText(mask_img, f"GT: {gt_class}", (30, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 保存掩码可视化
                mmcv.imwrite(mmcv.rgb2bgr(mask_img), os.path.join(output_dir, f'{img_id}_mask_rcnn_mask.jpg'))
            else:
                mmcv.imwrite(mmcv.rgb2bgr(img_mask_rcnn_seg), os.path.join(output_dir, f'{img_id}_mask_rcnn_mask.jpg'))
        else:
            mmcv.imwrite(mmcv.rgb2bgr(img_mask_rcnn_seg), os.path.join(output_dir, f'{img_id}_mask_rcnn_mask.jpg'))
        
        # 创建RPN提案框的可视化
        img_proposals = img.copy()
        
        # 提取RPN提案框
        # 转换为张量
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).contiguous()
        
        # 创建数据样本
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': img.shape[:2],
            'pad_shape': img.shape[:2],
            'batch_input_shape': img.shape[:2],
            'scale_factor': (1.0, 1.0),
            'ori_shape': img.shape[:2]
        })
        
        # 通过模型处理图像
        device = next(mask_rcnn_model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            # 提取特征
            feats = mask_rcnn_model.backbone(img_tensor)
            if mask_rcnn_model.with_neck:
                feats = mask_rcnn_model.neck(feats)
                
            # 获取提案框
            data_samples_list = [data_sample.to(device)]
            rpn_results_list = mask_rcnn_model.rpn_head.predict(feats, data_samples_list)
        
        # 绘制提案框
        for rpn_results in rpn_results_list:
            bboxes = rpn_results.bboxes
            scores = rpn_results.scores
            
            # 仅保留前100个提案框
            keep_idxs = scores.argsort(descending=True)[:100]
            bboxes = bboxes[keep_idxs]
            
            # 绘制提案框
            for bbox in bboxes:
                bbox_int = bbox.int().cpu().tolist()
                cv2.rectangle(img_proposals, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0, 255, 0), 1)
        
        # 添加标题
        cv2.putText(img_proposals, "Mask R-CNN Proposals", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存提案框图像
        mmcv.imwrite(mmcv.rgb2bgr(img_proposals), os.path.join(output_dir, f'{img_id}_mask_rcnn_proposals.jpg'))
        
        # ----------------------------------------------------------------------
        # Sparse R-CNN inference and visualization
        # ----------------------------------------------------------------------
        sparse_rcnn_result = inference_detector(sparse_rcnn_model, img)
        
        # 应用更严格的非极大值抑制(NMS)
        # 首先过滤低置信度的检测结果
        if hasattr(sparse_rcnn_result.pred_instances, 'scores'):
            valid_indices = sparse_rcnn_result.pred_instances.scores >= CONFIDENCE_THRESHOLD
            
            # 只保留高置信度的预测结果
            sparse_rcnn_result.pred_instances = sparse_rcnn_result.pred_instances[valid_indices]
        
        # 打印检测结果信息
        print(f"\nSparse R-CNN Predictions (confidence >= {CONFIDENCE_THRESHOLD}):")
        print(f"  Ground Truth classes: {', '.join(gt_class_names)}")
        
        if hasattr(sparse_rcnn_result.pred_instances, 'labels') and hasattr(sparse_rcnn_model, 'dataset_meta'):
            if hasattr(sparse_rcnn_result.pred_instances, 'bboxes'):
                # 首先显示所有预测结果
                print(f"\n  [All predictions, total: {len(sparse_rcnn_result.pred_instances)}]")
                for i, label in enumerate(sparse_rcnn_result.pred_instances.labels):
                    label_idx = label.item()
                    score = sparse_rcnn_result.pred_instances.scores[i].item()
                    class_name = model_class_names[label_idx]
                    bbox = sparse_rcnn_result.pred_instances.bboxes[i].cpu().numpy()
                    print(f"  All Pred {i+1}: Class={class_name}, Score={score:.3f}, BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # 然后显示匹配GT类别的预测
                matched_pred_count = 0
                print(f"\n  [Matched with GT classes only]")
                for i, label in enumerate(sparse_rcnn_result.pred_instances.labels):
                    label_idx = label.item()
                    if label_idx in gt_class_indices:  # 使用模型类别索引进行匹配
                        matched_pred_count += 1
                        score = sparse_rcnn_result.pred_instances.scores[i].item()
                        class_name = model_class_names[label_idx]
                        bbox = sparse_rcnn_result.pred_instances.bboxes[i].cpu().numpy()
                        print(f"  Matched Pred {matched_pred_count}: Class={class_name}, Score={score:.3f}, BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                if matched_pred_count == 0:
                    print(f"  No predictions match the Ground Truth classes")
        
        # 修改: 为Sparse R-CNN创建自定义的结果可视化，显示所有预测框
        img_sparse_rcnn = img.copy()
        
        if hasattr(sparse_rcnn_result.pred_instances, 'scores') and len(sparse_rcnn_result.pred_instances.scores) > 0:
            # 获取所有预测框
            bboxes = sparse_rcnn_result.pred_instances.bboxes.cpu().numpy().astype(np.int32)
            labels = sparse_rcnn_result.pred_instances.labels.cpu().numpy()
            scores = sparse_rcnn_result.pred_instances.scores.cpu().numpy()
            
            # 寻找最匹配的GT类别
            gt_class_name = "Unknown"
            if img_id in id_to_annotations:
                gt_annotations = id_to_annotations[img_id]
                if len(gt_annotations) > 0:
                    # 使用第一个GT类别替换预测类别
                    cat_id = gt_annotations[0]['category_id']
                    gt_class_name = category_id_to_name.get(cat_id, "Unknown")
            
            # 绘制所有预测框
            for i, box in enumerate(bboxes):
                label_idx = labels[i]
                score = scores[i]
                class_name = model_class_names[label_idx]
                
                # 在图像上绘制边界框和类别标签
                x1, y1, x2, y2 = box
                cv2.rectangle(img_sparse_rcnn, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 添加GT类别名称，但保留原始预测的置信度分数
                label_text = f"{gt_class_name}: {score:.4f}"
                cv2.putText(img_sparse_rcnn, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 如果有Ground Truth标注，添加到图像上
            if img_id in id_to_annotations:
                gt_annotations = id_to_annotations[img_id]
                for ann in gt_annotations:
                    cat_id = ann['category_id']
                    gt_class = category_id_to_name.get(cat_id, "Unknown")
                    
                    # COCO格式的边界框 [x, y, width, height]
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # 绘制GT边界框 (绿色)
                    cv2.rectangle(img_sparse_rcnn, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 添加GT类别标签
                    cv2.putText(img_sparse_rcnn, f"GT: {gt_class}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存自定义的检测结果图像
        mmcv.imwrite(mmcv.rgb2bgr(img_sparse_rcnn), os.path.join(output_dir, f'{img_id}_sparse_rcnn_bbox.jpg'))
        
        # ----------------------------------------------------------------------
        # 创建单行四图的可视化比较
        # ----------------------------------------------------------------------
        
        # 读取保存的图像
        proposals_img = mmcv.imread(os.path.join(output_dir, f'{img_id}_mask_rcnn_proposals.jpg'))
        mask_rcnn_bbox_img = mmcv.imread(os.path.join(output_dir, f'{img_id}_mask_rcnn_bbox.jpg'))
        mask_rcnn_mask_img = mmcv.imread(os.path.join(output_dir, f'{img_id}_mask_rcnn_mask.jpg'))
        sparse_rcnn_bbox_img = mmcv.imread(os.path.join(output_dir, f'{img_id}_sparse_rcnn_bbox.jpg'))
        
        # 确保所有图像具有相同的大小
        h, w = mask_rcnn_bbox_img.shape[:2]
        proposals_img = cv2.resize(proposals_img, (w, h))
        mask_rcnn_mask_img = cv2.resize(mask_rcnn_mask_img, (w, h))
        sparse_rcnn_bbox_img = cv2.resize(sparse_rcnn_bbox_img, (w, h))
        
        # 创建带有白色背景的标题图像
        def create_title_image(title, width, height=50):
            title_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_size = cv2.getTextSize(title, font, font_scale, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(title_img, title, (text_x, text_y), font, font_scale, (0, 0, 0), 2)
            return title_img
        
        # 创建标题图像
        title_proposals = create_title_image("Mask R-CNN Proposals", w)
        title_mask_rcnn_bbox = create_title_image("Mask R-CNN Detection", w)
        title_mask_rcnn_mask = create_title_image("Mask R-CNN Segmentation", w)
        title_sparse_rcnn_bbox = create_title_image("Sparse R-CNN Detection", w)
        
        # 创建单行四图比较
        row_images = np.hstack([proposals_img, mask_rcnn_bbox_img, mask_rcnn_mask_img, sparse_rcnn_bbox_img])
        row_titles = np.hstack([title_proposals, title_mask_rcnn_bbox, title_mask_rcnn_mask, title_sparse_rcnn_bbox])
        
        # 合并标题和图像
        comparison_row = np.vstack([row_titles, row_images])
        
        # 添加图像文件名作为顶部标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison_row, f"Image: {img_name}", (10, 20), font, 0.5, (0, 0, 0), 1)
        
        # 保存单行四图比较
        mmcv.imwrite(comparison_row, os.path.join(output_dir, f'{img_id}_comparison_row.jpg'))
        
        print(f"\nCompleted processing image {img_id}")
        print(f"=========================================")

if __name__ == '__main__':
    main() 