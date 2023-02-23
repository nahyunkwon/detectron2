# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

data_root = r"/media/nahyun/HDD/data_100/" # data home path

register_coco_instances("retinanet_train", {}, data_root + "instances_train.json", data_root+'train/images')
register_coco_instances("retinanet_val", {}, data_root + "instances_val.json", data_root+'val/images')

data = DatasetCatalog.get("retinanet_train")