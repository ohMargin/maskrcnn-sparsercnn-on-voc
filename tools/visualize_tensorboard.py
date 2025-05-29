import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def extract_json_data(json_file_path, tags):
    """从指定的JSON日志文件中提取数据"""
    if not os.path.exists(json_file_path):
        print(f"警告：JSON文件不存在: {json_file_path}")
        return {}
    
    print(f"使用JSON日志文件: {json_file_path}")
    
    # 读取JSON文件
    data = {}
    try:
        with open(json_file_path, 'r') as f:
            lines = f.readlines()
            
        # 解析每一行JSON数据
        records = []
        for line in lines:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue
        
        # 提取所需的标签数据
        available_tags = set()
        for record in records:
            available_tags.update(record.keys())
        
        print(f"可用的标签: {available_tags}")
        
        for tag in tags:
            if tag in available_tags:
                steps = []
                values = []
                for record in records:
                    if tag in record and 'step' in record:
                        steps.append(record['step'])
                        values.append(record[tag])
                
                if steps:  # 只有当有数据时才添加
                    data[tag] = {
                        'step': steps,
                        'value': values
                    }
            else:
                print(f"警告: 标签 '{tag}' 不在日志中")
                
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
    
    return data

def plot_curves(data, output_dir, title):
    """绘制曲线图"""
    if not data:
        print(f"警告: 没有数据可以绘制图表 '{title}'")
        return
        
    plt.figure(figsize=(12, 8))
    
    for tag, values in data.items():
        plt.plot(values['step'], values['value'], label=tag)
    
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_")}.png'), dpi=300)
    plt.close()

def main():
    # 指定的JSON日志文件路径
    mask_rcnn_json_file = 'work_dirs/mask_rcnn_voc/20250525_155324/vis_data/20250525_155324.json'
    sparse_rcnn_json_file = 'work_dirs/sparse_rcnn_voc/20250526_075415/vis_data/20250526_075415.json'
    
    # 输出目录
    output_dir = 'work_dirs/tensorboard_vis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取 Mask R-CNN 的数据
    mask_rcnn_tags = [
        'loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox',
        'acc', 'pascal_voc/mAP', 'pascal_voc/AP50'
    ]
    mask_rcnn_data = extract_json_data(mask_rcnn_json_file, mask_rcnn_tags)
    
    # 提取 Sparse R-CNN 的数据
    # Sparse R-CNN有多个阶段(s0-s5)，每个阶段都有自己的loss
    sparse_rcnn_tags = [
        'loss', 
        's0.loss_cls', 's1.loss_cls', 's2.loss_cls', 's3.loss_cls', 's4.loss_cls', 's5.loss_cls',
        's0.loss_bbox', 's1.loss_bbox', 's2.loss_bbox', 's3.loss_bbox', 's4.loss_bbox', 's5.loss_bbox',
        's0.loss_iou', 's1.loss_iou', 's2.loss_iou', 's3.loss_iou', 's4.loss_iou', 's5.loss_iou',
        's0.pos_acc', 's1.pos_acc', 's2.pos_acc', 's3.pos_acc', 's4.pos_acc', 's5.pos_acc',
        'pascal_voc/mAP', 'pascal_voc/AP50'
    ]
    sparse_rcnn_data = extract_json_data(sparse_rcnn_json_file, sparse_rcnn_tags)
    
    # 绘制 Mask R-CNN 的 loss 曲线
    mask_rcnn_loss_data = {tag: values for tag, values in mask_rcnn_data.items() if 'loss' in tag}
    plot_curves(mask_rcnn_loss_data, output_dir, 'Mask R-CNN Loss Curves')
    
    # 绘制 Sparse R-CNN 的总loss曲线
    sparse_rcnn_total_loss_data = {tag: values for tag, values in sparse_rcnn_data.items() if tag == 'loss'}
    plot_curves(sparse_rcnn_total_loss_data, output_dir, 'Sparse R-CNN Total Loss Curve')
    
    # 绘制 Sparse R-CNN 的分类loss曲线
    sparse_rcnn_cls_loss_data = {tag: values for tag, values in sparse_rcnn_data.items() if 'loss_cls' in tag}
    plot_curves(sparse_rcnn_cls_loss_data, output_dir, 'Sparse R-CNN Classification Loss Curves')
    
    # 绘制 Sparse R-CNN 的边界框loss曲线
    sparse_rcnn_bbox_loss_data = {tag: values for tag, values in sparse_rcnn_data.items() if 'loss_bbox' in tag}
    plot_curves(sparse_rcnn_bbox_loss_data, output_dir, 'Sparse R-CNN Bounding Box Loss Curves')
    
    # 绘制 Sparse R-CNN 的IoU loss曲线
    sparse_rcnn_iou_loss_data = {tag: values for tag, values in sparse_rcnn_data.items() if 'loss_iou' in tag}
    plot_curves(sparse_rcnn_iou_loss_data, output_dir, 'Sparse R-CNN IoU Loss Curves')
    
    # 绘制 Sparse R-CNN 的准确率曲线
    sparse_rcnn_acc_data = {tag: values for tag, values in sparse_rcnn_data.items() if 'pos_acc' in tag}
    plot_curves(sparse_rcnn_acc_data, output_dir, 'Sparse R-CNN Accuracy Curves')
    
    # 绘制 Mask R-CNN 的 mAP 曲线
    mask_rcnn_map_data = {tag: values for tag, values in mask_rcnn_data.items() if 'mAP' in tag or 'AP' in tag}
    plot_curves(mask_rcnn_map_data, output_dir, 'Mask R-CNN mAP Curves')
    
    # 绘制 Sparse R-CNN 的 mAP 曲线
    sparse_rcnn_map_data = {tag: values for tag, values in sparse_rcnn_data.items() if 'mAP' in tag or 'AP' in tag}
    plot_curves(sparse_rcnn_map_data, output_dir, 'Sparse R-CNN mAP Curves')
    
    # 对比 Mask R-CNN 和 Sparse R-CNN 的 mAP
    if 'pascal_voc/mAP' in mask_rcnn_data and 'pascal_voc/mAP' in sparse_rcnn_data:
        plt.figure(figsize=(12, 8))
        plt.plot(mask_rcnn_data['pascal_voc/mAP']['step'], mask_rcnn_data['pascal_voc/mAP']['value'], label='Mask R-CNN mAP')
        plt.plot(sparse_rcnn_data['pascal_voc/mAP']['step'], sparse_rcnn_data['pascal_voc/mAP']['value'], label='Sparse R-CNN mAP')
        plt.title('Mask R-CNN vs Sparse R-CNN mAP')
        plt.xlabel('Steps')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'Mask_RCNN_vs_Sparse_RCNN_mAP.png'), dpi=300)
        plt.close()
    
    # 对比 Mask R-CNN 和 Sparse R-CNN 的 AP50
    if 'pascal_voc/AP50' in mask_rcnn_data and 'pascal_voc/AP50' in sparse_rcnn_data:
        plt.figure(figsize=(12, 8))
        plt.plot(mask_rcnn_data['pascal_voc/AP50']['step'], mask_rcnn_data['pascal_voc/AP50']['value'], label='Mask R-CNN AP50')
        plt.plot(sparse_rcnn_data['pascal_voc/AP50']['step'], sparse_rcnn_data['pascal_voc/AP50']['value'], label='Sparse R-CNN AP50')
        plt.title('Mask R-CNN vs Sparse R-CNN AP50')
        plt.xlabel('Steps')
        plt.ylabel('AP50')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'Mask_RCNN_vs_Sparse_RCNN_AP50.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    main() 