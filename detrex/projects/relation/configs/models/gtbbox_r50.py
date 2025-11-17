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

from projects.relation.modeling.gtbbox_relation_predictor import GTBBoxRelationPredictor

model = L(GTBBoxRelationPredictor)(
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=False,
                norm="FrozenBN",
            ),
            out_features=["res3", "res4", "res5"],
            freeze_at=1,
        ),
        in_features=["res3", "res4", "res5"],
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
