# 划分voc数据集
import os
import random
import xml
from xml.dom import minidom

VOC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', ]


def generate_train_val_test_txt():
    xml_file_path = "data2_voc/Annotations"  # xml文件路径
    save_Path = "data2_voc/ImageSets/Main"
    ############################################3
    ftrainval = open(os.path.join(save_Path, 'trainval.txt'), 'r')
    trainval = ftrainval.readlines()
    num = len(trainval)
    ftrainval.close()
    ftrain = open(os.path.join(save_Path, 'train.txt'), 'r')
    train = ftrain.readlines()
    ftrain.close()

    for idx in range(len(VOC_CLASSES)):  # 每一个类单独处理
        class_name = VOC_CLASSES[idx]
        # 创建txt
        class_trainval = open(os.path.join(save_Path, str(class_name) + '_trainval.txt'), 'w')
        class_test = open(os.path.join(save_Path, str(class_name) + '_test.txt'), 'w')
        class_train = open(os.path.join(save_Path, str(class_name) + '_train.txt'), 'w')
        class_val = open(os.path.join(save_Path, str(class_name) + '_val.txt'), 'w')
        for k in trainval:
            xml_name = k.split('\n')[0]  # xml的名称
            print(xml_name)
            xml_path = os.path.join(xml_file_path, xml_name + '.xml')
            ##################################################
            # 将获取的xml文件名送入到dom解析
            dom = xml.dom.minidom.parse(xml_path)  # 输入xml文件具体路径
            root = dom.documentElement
            # 获取xml object标签<name>
            object_name = root.getElementsByTagName('name')
            object_names = [object_name[i].childNodes[0].data for i in range(len(object_name))]
            if len(object_name) > 0 and class_name in object_names:  # 存在object（矩形框并且class_name在object_name列表中
                if k in trainval:
                    class_trainval.write(xml_name + ' ' + str(1) + "\n")
                    if k in train:
                        class_train.write(xml_name + ' ' + str(1) + "\n")
                    else:
                        class_val.write(xml_name + ' ' + str(1) + "\n")
                else:
                    class_test.write(xml_name + ' ' + str(1) + "\n")
            # else:
            #     if k in trainval:
            #         class_trainval.write(xml_name + ' ' + str(-1) + "\n")
            #         if k in train:
            #             class_train.write(xml_name + ' ' + str(-1) + "\n")
            #         else:
            #             class_val.write(xml_name + ' ' + str(-1) + "\n")
            #     else:
            #         class_test.write(xml_name + ' ' + str(-1) + "\n")
        class_trainval.close()
        class_test.close()
        class_train.close()
        class_val.close()  # 1类的.txt编辑好了
    #################################################


generate_train_val_test_txt()
