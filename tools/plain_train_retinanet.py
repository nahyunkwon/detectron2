import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
# from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances
# from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2 import model_zoo

# # from detectron2.modeling import RETINANET_NERF_REGISTRY

# # from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

# # print(META_ARCH_REGISTRY)

# # from detectron2.modeling.meta_arch.retinanet_nerf import RetinaNet_Nerf

# # import RetinaNet_Nerf
# # print(detectron2.modeling.meta_arch.RetinaNet_Nerf)
# # from detectron2.modeling.meta_arch import RetinaNet_Nerf

# # from detectron2.modeling.meta_arch import RetinaNet_Nerf




# # epoch = 1
# # thresh = 0.7
# # cfg = get_cfg()
# # cfg.merge_from_file('../configs/retinanet_nerf.yaml')

# yaml_f = "retinanet_R_50_FPN_3x.yaml"
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/{}".format(yaml_f)))

data_root = r"/media/nahyun/HDD//data_100/" # data home path 
test_images = r'/media/nahyun/HDD/realDB/test/images'

register_coco_instances("retinanet_train", {}, os.path.join(data_root, "instances_train.json"), os.path.join(data_root, 'train', 'images'))
register_coco_instances("retinanet_val", {}, os.path.join(data_root, "instances_val.json"), os.path.join(data_root, 'val', 'images'))

register_coco_instances("retinanet_test", {}, os.path.join(data_root, "instances_test.json"), test_images)

yaml_f = "retinanet_R_50_FPN_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/{}".format(yaml_f)))
cfg.DATASETS.TRAIN = ("retinanet_train",)
cfg.DATASETS.TEST = ('retinanet_test', )
cfg.DATALOADER.NUM_WORKERS = 2

# cfg.MODEL.ROI_HEADS.NAME = 'NeRFHeads'
# cfg.MODEL.ROI_HEADS.NERF_WEIGHTS = "/home/yunhan/scratchT/nerf_weights/nerf_det/nerf-pytorch/logs/reduced_nerf_weights.pth"

FINETUNE_FROM = "COCO" # choose from ["COCO", "ImageNet""]

if FINETUNE_FROM == "COCO":
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/{}".format(yaml_f))  # Let training initialize from model zoo

version = "NeRF_10kRandFeat_002lr_15000step"
if FINETUNE_FROM == "ImageNet":
    version += "_ft{}".format(FINETUNE_FROM)

cfg.SOLVER.IMS_PER_BATCH = 12
ITERS_IN_ONE_EPOCH = int(20000 / cfg.SOLVER.IMS_PER_BATCH)
cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 3) - 1  # 3 epochs
# cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH) -1
cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 

print(trainer.model)

trainer.resume_or_load(resume=False)
trainer.train()


# cfg.DATASETS.TRAIN = ('retinanet_train', ) # the comma is necessary
# cfg.DATASETS.TEST = ('retinanet_test', )

# cfg.MODEL.ROI_HEADS.NAME = 'NeRFHeads'

#  # cfg.TEST.EVAL_PERIOD = 500
# cfg.MODEL.RETINANET.NUM_CLASSES = 100

# cfg.MODEL.ROI_HEADS.NAME = 'NeRFROIHeads'
# # cfg.MODEL.ROI_HEADS.NERF_WEIGHTS = "/media/nahyun/HDD/nerf_weights/nerf_weights_10k.pth"

# cfg.SOLVER.IMS_PER_BATCH = 12
# ITERS_IN_ONE_EPOCH = int(20000 / cfg.SOLVER.IMS_PER_BATCH)
# cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 3) - 1  # 3 epochs
# # cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH) -1
# cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
# cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH


# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()



