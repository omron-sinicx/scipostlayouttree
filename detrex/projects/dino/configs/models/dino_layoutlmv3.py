from functools import partial
import torch.nn as nn
import detectron2
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import SimpleFeaturePyramid
from detectron2.modeling import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from transformers import LayoutLMv3Model

from .dino_r50 import model


class LayoutLMv3Backbone(Backbone):
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        super().__init__()
        self.model = LayoutLMv3Model.from_pretrained(model_name)
        self._out_feature = "last_hidden_state"
        self._out_channels = self.model.config.hidden_size
        self._stride = 16

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        outputs = self.model(pixel_values=x)
        hidden = outputs.last_hidden_state  # [B, 197, C]

        hidden = hidden[:, 1:, :]  # [B, 196, C]

        B, N, C = hidden.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Cannot reshape: {N} tokens not square"

        feat = hidden.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        return {self._out_feature: feat}

    def output_shape(self):
        return {
            self._out_feature: ShapeSpec(
                channels=self._out_channels,
                stride=self._stride
            )
        }


# Simple Feature Pyramid from LayoutLMv3 backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(LayoutLMv3Backbone)(model_name="microsoft/layoutlmv3-base"),
    in_feature="last_hidden_state",
    out_channels=256,
    scale_factors=(2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

model.neck.input_shapes = {
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}
model.neck.in_features = ["p3", "p4", "p5", "p6"]
model.neck.num_outs = 4
model.transformer.num_feature_levels = 4
