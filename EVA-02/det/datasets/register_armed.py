from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Update these paths to your dataset locations.
register_coco_instances(
    "armed_train",
    {},
    "/path/to/armed_train.json",
    "/path/to/images",
)
register_coco_instances(
    "armed_val",
    {},
    "/path/to/armed_val.json",
    "/path/to/images",
)

MetadataCatalog.get("armed_train").thing_classes = ["Armed", "Unarmed", "Gun"]
MetadataCatalog.get("armed_val").thing_classes = ["Armed", "Unarmed", "Gun"]
