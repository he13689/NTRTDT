import os
import json
import cv2

# 定义类别（根据实际情况修改）
categories = [
    {"id": 1, "name": "broken"},
    {"id": 2, "name": "loosen"},
    {"id": 3, "name": "kites"},
    {"id": 4, "name": "balloon"},
    {"id": 5, "name": "cloth"},
    {"id": 6, "name": "tree"},
    {"id": 7, "name": "others"},
    # 继续添加其他类别
]

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    将YOLO格式的bbox转换为COCO格式
    """
    x_center, y_center, width, height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    return [x_min, y_min, width, height]

def convert_yolo_to_coco(images_dir, labels_dir, output_json):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1

    for img_filename in os.listdir(images_dir):
        if not img_filename.endswith(('.jpg', '.png')):
            continue
        img_id = int(os.path.splitext(img_filename)[0].split('_')[-1])
        img_name = os.path.splitext(img_filename)[0]
        img_path = os.path.join(images_dir, img_filename)
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape

        # 添加图像信息
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "height": img_height,
            "width": img_width,
        })

        label_filename = f"{img_name}.txt"
        label_path = os.path.join(labels_dir, label_filename)

        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0]) + 1  # 假设类别ID从0开始，需要加1
                    yolo_bbox = [float(x) for x in parts[1:]]
                    coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

                    # 添加标注信息
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": coco_bbox,
                        "area": coco_bbox[2] * coco_bbox[3],
                        "iscrowd": 0,
                    })
                    annotation_id += 1

    # 保存为JSON文件
    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

# 使用示例
images_dir = 'test/strands_images/'  # 图像文件夹路径
# images_dir = 'test/hanging_images/'  # 图像文件夹路径
labels_dir = 'test/labels/'  # 标签文件夹路径
output_json = 'test/test_strands.json'  # 输出的JSON文件路径
# output_json = 'test/test_hanging.json'  # 输出的JSON文件路径

convert_yolo_to_coco(images_dir, labels_dir, output_json)
