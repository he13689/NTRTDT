import time

import torch
from loguru import logger
import config
from src.core import YAMLConfig
from utils.trainer import DetTrainer

logger.add('output/custom18/test_log.txt')
# config.model_yaml = 'configs/rtdetr/rtdetr_custom13.yml'
# config.resume = 'output/custom18/best_model.pth'
config.model_yaml = 'configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
config.resume = 'output/rtdetr50/best_model.pth'
config.test_only = True
config.half = False

cfg = YAMLConfig(
    config.model_yaml,
    resume=config.resume,
    use_amp=config.use_amp,
    tuning=config.tuning
)

ckpt = torch.load(config.resume, map_location='cpu')

msg = cfg.model.load_state_dict(ckpt)
print(msg)

logger.info('start infer ...')
cfg.model.to(config.device)
cfg.model.backbone.half().to(config.device)
cfg.model.encoder.half().to(config.device)
cfg.model.decoder.half().to(config.device)
cfg.model.eval()

for _ in range(10):
    image = torch.HalfTensor(8, 3, 640, 640).to(config.device)
    stime = time.time()
    cfg.model(image)
    etime = time.time()
    last_time = etime - stime
    bs = 8
    logger.error(f'the model run {bs} samples uses {last_time} secs, per image takes {last_time / bs} secs')
