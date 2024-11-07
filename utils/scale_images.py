import os
import cv2


def resize_images_in_folder(folder_path, scale=0.5):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理图片文件
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            file_path = os.path.join(folder_path, filename)

            # 读取图像
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to read {filename}")
                continue

            # 获取图像的尺寸
            height, width = image.shape[:2]

            if height >= 3000 or width >= 3000:

                # 计算新的尺寸
                new_dimensions = (int(width * scale), int(height * scale))

                # 调整图像大小
                resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

                # 保存调整后的图像
                cv2.imwrite(file_path, resized_image)
                print(f"Resized {filename} to {new_dimensions}")


# 指定文件夹路径
folder_path = "data_new_yolo/images"

# 调用函数进行图像缩小
resize_images_in_folder(folder_path, scale=0.5)
