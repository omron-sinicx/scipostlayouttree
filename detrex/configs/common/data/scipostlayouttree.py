import itertools

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper

from .coco_relation import register_scipost
from .relation_evaluator import RelationEvaluator
from .tree_evaluator import TreeEvaluator

dataloader = OmegaConf.create()

# register_coco_instances("my_dataset_train", {}, '/path/to/train.json', '/path/to/train/images')
# register_coco_instances("my_dataset_test", {}, '/path/to/test.json', '/path/to/test/images')

# register_coco_instances(
#     "scipostlayouttree_train",
#     {},
#     "/scipostlayout/poster/png/train.json",
#     "/scipostlayout/poster/png/train"
# )

# register_coco_instances(
#     "scipostlayouttree_val",
#     {},
#     "/scipostlayout/poster/png/dev.json",
#     "/scipostlayout/poster/png/dev"
# )

register_scipost(
    "scipostlayouttree_train",
    "/scipostlayout/poster/png/train_tree_ocr.json",
    "/scipostlayout/poster/png/train"
)

register_scipost(
    "scipostlayouttree_val",
    "/scipostlayout/poster/png/dev_tree_ocr.json",
    "/scipostlayout/poster/png/dev"
)

register_scipost(
    "scipostlayouttree_test",
    "/scipostlayout/poster/png/test_tree_ocr.json",
    "/scipostlayout/poster/png/test"
)

dataloader.train = L(build_detection_train_loader)(
    # dataset=L(get_detection_dataset_dicts)(names="my_dataset_train"),
    dataset=L(get_detection_dataset_dicts)(names="scipostlayouttree_train", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        # augmentation=[
        #     # L(T.RandomFlip)(),
        #     L(T.ResizeShortestEdge)(
        #         short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
        #         max_size=1333,
        #         sample_style="choice",
        #     ),
        # ],
        augmentation=[
            # L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1024,
                sample_style="choice",
            ),
        ],
        # augmentation_with_crop=[
        #     L(T.RandomFlip)(),
        #     L(T.ResizeShortestEdge)(
        #         short_edge_length=(400, 500, 600),
        #         sample_style="choice",
        #     ),
        #     L(T.RandomCrop)(
        #         crop_type="absolute_range",
        #         crop_size=(384, 600),
        #     ),
        #     L(T.ResizeShortestEdge)(
        #         short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
        #         max_size=1333,
        #         sample_style="choice",
        #     ),
        # ],
        augmentation_with_crop=None,
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    # dataset=L(get_detection_dataset_dicts)(names="my_dataset_test", filter_empty=False),
    dataset=L(get_detection_dataset_dicts)(names="scipostlayouttree_val", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        # augmentation=[
        #     L(T.ResizeShortestEdge)(
        #         short_edge_length=800,
        #         max_size=1333,
        #     ),
        # ],
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1024,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = [
    L(COCOEvaluator)(
        dataset_name="${dataloader.test.dataset.names}",
        output_dir=None,
    ),
    L(RelationEvaluator)(
        output_dir=None,
    ),
    L(TreeEvaluator)(
        output_dir=None,
    ),
]
