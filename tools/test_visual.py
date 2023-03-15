from detectron2.data.datasets import register_coco_instances
import os
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog

data_root = "/media/nahyun/HDD/accesslens/AccessLens_COCO/"
register_coco_instances("accesslens_train", {}, os.path.join(data_root, "instances_train.json"), os.path.join(data_root, 'train/images'))


import random
from detectron2.utils.visualizer import Visualizer
import cv2

metadata = MetadataCatalog.get("accesslens_train")
dataset_dicts = DatasetCatalog.get("accesslens_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite('./test.jpg', vis.get_image()[:, :, ::-1])