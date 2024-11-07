# 筛选指定类别，抛弃无用类别
import os


def filter_yolo_labels(folder_path, target_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            tar_path = os.path.join(target_path, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()

            filtered_lines = []
            for line in lines:
                parts = line.split()
                label = int(parts[0])
                if 0 <= label <= 6:
                    filtered_lines.append(line)

            if len(filtered_lines) == 0:
                 continue

            with open(tar_path, "w") as file:
                file.writelines(filtered_lines)


# 指定文件夹路径
folder_path = "data_yolo/Annotations/val"
target_path = "data_new_yolo/Annotations/val"

# 调用函数进行标签过滤
filter_yolo_labels(folder_path, target_path)
