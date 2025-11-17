import copy
import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L
from detectron2.modeling import FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import SwinTransformer

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.relation.modeling.gtbbox_relation_predictor import GTBBoxRelationPredictor

model = L(GTBBoxRelationPredictor)(
    backbone=L(FPN)(
        bottom_up=L(SwinTransformer)(
            pretrain_img_size=384,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=12,
            out_indices=(0, 1, 2, 3),
        ),
        in_features=["p1", "p2", "p3"],
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
