import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
import random
import shutil
random.seed(49)

# 反向映射：索引到类别名
VOC_ID_TO_NAME = {
    0: "background",
    1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
    6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"
}

def parse_voc_xml(xml_path):
    """解析 VOC XML 标注文件，提取实例信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text)
        truncated = int(obj.find('truncated').text)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO格式[x,y,w,h]
            'difficult': difficult,
            'truncated': truncated,
            'segmentation': None  # 从分割掩码中提取
        })

    return width, height, objects


def extract_instance_masks(mask_path, objects):
    """
    :param mask_path: 单通道PNG掩码路径
    :param objects: parse_voc_xml()返回的对象列表（需含bbox）
    :return: 更新后的objects列表（含segmentation字段）
    """
    # 加载单通道掩码
    mask = np.array(Image.open(mask_path))  # shape: [H, W]
    instance_id = 1
    for obj in objects:
        # 生成当前实例的二值掩码
        binary_mask = (mask == instance_id).astype(np.uint8) * 255
        # 提取轮廓
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            contour = contour.squeeze(1)  # 去除冗余维度
            if len(contour) < 3:
                continue  # 忽略点或线
            polygons.append(contour.flatten().tolist())
        if polygons:
            obj['segmentation'] = polygons
        instance_id += 1
    return objects


def merge_voc_instance_segmentation(val_ratio=0.1):
    """合并 VOC 2007 和 2012 的实例分割标注为 COCO 格式"""
    # COCO 类别定义
    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in VOC_ID_TO_NAME.items() if idx != 0  # 跳过背景
    ]

    images = {'train': [], 'val': []}
    annotations = {'train': [], 'val': []}
    annotation_id = {'train': 1, 'val': 1}

    main_path = os.path.join(os.getcwd(), 'data')
    coco_json_annotations = os.path.join(main_path, 'coco', 'annotations')
    coco_train2017 = os.path.join(main_path, 'coco', 'train2017')
    coco_val2017 = os.path.join(main_path, 'coco', 'val2017')
    mkdir(coco_json_annotations)
    mkdir(coco_train2017)
    mkdir(coco_val2017)

    for year in ['2007', '2012']:
        anno_dir = os.path.join(main_path, 'VOCdevkit', f'VOC{year}', 'Annotations')
        jpeg_dir = os.path.join(main_path, 'VOCdevkit', f'VOC{year}', 'JPEGImages')
        seg_dir = os.path.join(main_path, 'VOCdevkit', f'VOC{year}', 'SegmentationObject')

        for xml_file in tqdm(os.listdir(anno_dir), desc=f"处理 VOC{year}"):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(anno_dir, xml_file)
            image_id = os.path.splitext(xml_file)[0]
            seg_path = os.path.join(seg_dir, f'{image_id}.png')

            if not os.path.exists(seg_path):
                continue

            isTrain = random.random() > val_ratio
            # 解析XML和分割掩码
            width, height, objects = parse_voc_xml(xml_path)
            extract_instance_masks(seg_path, objects)

            # 添加图像信息
            if isTrain:
                _images = images['train']
                _annotations = annotations['train']
                _annotation_id = annotation_id['train']
                target_path = os.path.join(coco_train2017, f"{image_id}.jpg")
            else:
                _images = images['val']
                _annotations = annotations['val']
                _annotation_id = annotation_id['val']
                target_path = os.path.join(coco_val2017, f"{image_id}.jpg")
            shutil.copy(os.path.join(jpeg_dir, f"{image_id}.jpg"), target_path)
            _images.append({
                "id": image_id,
                "file_name": f"{image_id}.jpg",
                "width": width,
                "height": height
            })

            # 添加标注信息
            for obj in objects:
                if not obj['segmentation']:
                    continue

                _annotations.append({
                    "id": _annotation_id,
                    "image_id": image_id,
                    "category_id": next(
                        c["id"] for c in categories if c["name"] == obj["name"]
                    ),
                    "segmentation": obj["segmentation"],
                    "area": obj["bbox"][2] * obj["bbox"][3],  # 近似面积
                    "bbox": obj["bbox"],
                    "iscrowd": 0
                })
                _annotation_id += 1
            if isTrain:
                annotation_id['train'] = _annotation_id
            else:
                annotation_id['val'] = _annotation_id

    # 保存为 COCO 格式
    with open(os.path.join(coco_json_annotations, 'instances_train2017.json'), 'w') as f:
        json_str = json.dumps({
            "images": images['train'],
            "annotations": annotations['train'],
            "categories": categories
        })
        f.write(json_str)
    with open(os.path.join(coco_json_annotations, 'instances_val2017.json'), 'w') as f:
        json_str = json.dumps({
            "images": images['val'],
            "annotations": annotations['val'],
            "categories": categories
        })
        f.write(json_str)
        
      
# 新建文件夹
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' ----- folder created')
        return True
    else:
        print(path + ' ----- folder existed')
        return False

if __name__ == "__main__":
    merge_voc_instance_segmentation()