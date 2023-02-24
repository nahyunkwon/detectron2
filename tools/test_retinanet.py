#import the COCO Evaluator to use the COCO Metrics
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import os
import sys

def test(weights):
    #register your data
    data_root = r"/mnt/data_100/" # data home path

    # register_coco_instances("retinanet_train", {}, os.path.join(data_root, "instances_train.json"), os.path.join(data_root, 'train', 'images'))
    # register_coco_instances("retinanet_val", {}, os.path.join(data_root, "instances_val.json"), os.path.join(data_root,'val/images'))

    try:
        register_coco_instances("retinanet_test", {}, os.path.join(data_root, "instances_test.json"), r'/mnt/realDB/test/images')
    except:
        pass
    #load the config file, configure the threshold value, load weights 
    cfg = get_cfg()
    cfg.MODEL.RETINANET.NUM_CLASSES=100

    cfg.merge_from_file("../configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = weights

    # Create predictor
    predictor = DefaultPredictor(cfg)

    #Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator("retinanet_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "retinanet_test")

    #Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)


def resize():
    import cv2

    # im = cv2.imread(r'/mnt/realDB/test_1/images/test_000.jpg')
    # print('test_1/images/test_000.jpg')
    # print(im.shape)

    # print('test_2/images/test_000.jpg')
    # im = cv2.imread(r'/mnt/realDB/test_2/images/test_000.jpg')
    # print(im.shape)

    # print('test_4/images/test_000.jpg')
    # im = cv2.imread(r'/mnt/realDB/test_4/images/test_000.jpg')
    # print(im.shape)

    # print('test_8/images/test_000.jpg')
    # im = cv2.imread(r'/mnt/realDB/test_8/images/test_000.jpg')
    # print(im.shape)

    import cv2
    import numpy as np

    test = r'/mnt/realDB/test_1/images/'
    test_reduced = r'/mnt/realDB/test/images/'

    test_raw = os.listdir(test)

    for r in test_raw:
        print(test + r)
        img = cv2.imread(test + r)

        res = cv2.resize(img, dsize=(1024, 768)) #, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(test_reduced + r, res)


if __name__ == "__main__":
    print('1st')
    test("./output/model_0001664.pth")
    print('2nd')
    test("./output/model_0003329.pth")
    print('3rd')
    test("./output/model_0004994.pth")
    print('final')
    # test("./output/retinanet_imagenet.pth")

    # main()