import copy
import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L
from detectron2.modeling import FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import VIT_Backbone

from projects.relation.modeling.gtbbox_relation_predictor import GTBBoxRelationPredictor

model = L(GTBBoxRelationPredictor)(
    backbone=L(FPN)(
        bottom_up=L(VIT_Backbone)(
            name="dit_base_patch16",
            out_features=["layer3", "layer5", "layer7", "layer11"],
            drop_path=0.1,
            img_size=[224, 224],
            pos_type="abs",
            model_kwargs={
                "use_checkpoint": False,
            },
            config_path=None,
            image_only=True,
            cfg=None,
        ),
        in_features=["layer3", "layer5", "layer7", "layer11"],
        out_channels=256,
        # norm="GN",
        norm="",
        top_block=L(LastLevelMaxPool)(),
        fuse_type="sum",
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=None,
    embed_dim=1024,
    num_classes=8,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    input_format="RGB",
    device="cuda",
)
