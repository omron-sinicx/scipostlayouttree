import copy
import torch.nn as nn
from functools import partial

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.relation.modeling.gtbbox_relation_predictor import GTBBoxRelationPredictor


# ViT Base Hyper-params
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1


model = L(GTBBoxRelationPredictor)(
    backbone=L(SimpleFeaturePyramid)(
        net=L(ViT)(  # Single-scale ViT backbone
            img_size=1024,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=dp,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
        ),
        in_feature="${.net.out_feature}",
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
        top_block=L(LastLevelMaxPool)(),
        norm="LN",
        square_pad=1024,
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
