import os

from utils.common import generate_colors

model_yaml = 'configs/rtdetr/rtdetr_insulator.yml'  # 模型yaml配置文件
resume = False  # 继续训练
use_amp = True  # amp加速
tuning = 'weights/rtdetr_r50vd_6x_coco_from_paddle.pth'  # 预训练模型位置

conf_thres = .32  # 置信度筛选
iou_type = 'NIOU'  # NIOU MPDIOU
new_loss = False
discount_factor = .5  # 加权box损失

test_only = False
chs = 3
device = 'cpu'
half = False
v8_weights = 'weights/yolom.pt'
v7_weights = 'weights/yolov7.pth'
v10_weights = 'weights/yolov10_l.pt'
save_counter = 0
colors = generate_colors(8)  # 根据类别选择颜色数量
best_ap = 0.300
counter = 0
images_list = os.listdir('data/strands/')
