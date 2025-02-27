#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

import torch

import logging
import os
from collections import OrderedDict
from detectron2 import model_zoo

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
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
from detectron2.modeling import GeneralizedRCNNWithTTA


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    ## COCO pretrained weights
    yaml_f = "retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/{}".format(yaml_f)))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/{}".format(yaml_f))

    ## ImageNet pretrained weights
    # cfg.merge_from_file(args.config_file)
    logger = logging.getLogger("detectron2.trainer")

    logger.info('NK: args.opts', args.opts)
    
    cfg.merge_from_list(args.opts)
    
    default_setup(cfg, args)
    
    data_root = r"/media/nahyun/HDD//data_100/" # data home path 
    test_images = r'/media/nahyun/HDD/realDB/test/images'

    register_coco_instances("retinanet_train", {}, os.path.join(data_root, "instances_train.json"), os.path.join(data_root, 'train', 'images'))
    register_coco_instances("retinanet_val", {}, os.path.join(data_root, "instances_val.json"), os.path.join(data_root, 'val', 'images'))

    register_coco_instances("retinanet_test", {}, os.path.join(data_root, "instances_test.json"), test_images)

    cfg.DATASETS.TRAIN = ("retinanet_train", )
    cfg.DATASETS.TEST = ("retinanet_test", )

    # cfg.TEST.EVAL_PERIOD = 500
    cfg.MODEL.RETINANET.NUM_CLASSES = 100
    
    # cfg.MODEL.ROI_HEADS.NAME = 'NeRFROIHeads'
    # cfg.MODEL.ROI_HEADS.NERF_WEIGHTS = "/media/nahyun/HDD/nerf_weights/nerf_weights_10k.pth"

    cfg.SOLVER.IMS_PER_BATCH = 12
    ITERS_IN_ONE_EPOCH = int(20000 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 3) - 1  # 3 epochs
    # cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH) -1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup(args)

    torch.cuda.empty_cache()

    x

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

#### Run this command
#### python3 train_retinanet.py --num-gpu 2 --config-file ../configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml