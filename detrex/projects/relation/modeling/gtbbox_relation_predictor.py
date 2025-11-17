# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead
from detectron2.layers import ShapeSpec

from .drgg import DRGGHead


class GTBBoxRelationPredictor(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        embed_dim: int,
        num_classes: int,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        device="cuda",
        input_format: Optional[str] = "RGB",
        use_position_emb: bool = False,
        use_class_emb: bool = False,
        use_bbox_emb: bool = False,
        use_tf_enc: bool = False,
        use_text_emb: bool = False,
        text_encoder: str = None,
        relation_loss_weight: float = 1.0,
        beam_width: int = 1,
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.embed_dim = embed_dim

        # define classification head and box head
        self.num_classes = num_classes

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # the period for visualizing training samples
        self.input_format = input_format
        
        # bbox pooler
        # The pooler is used to extract features from the backbone for each gt box.
        pooler_resolution = 7
        pooler_scales     = [1.0 / self.backbone._out_feature_strides[f] for f in self.backbone._out_features]
        sampling_ratio    = 0
        pooler_type       = "ROIAlignV2"

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.channels = self.backbone.output_shape()[self.backbone._out_features[0]].channels
        roi_feat_spec = ShapeSpec(channels=self.channels, height=pooler_resolution, width=pooler_resolution)
        self.box_head = FastRCNNConvFCHead(
            input_shape=roi_feat_spec,
            conv_dims=[],
            fc_dims=[self.embed_dim, self.embed_dim]
        )

        # self.bbox_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(self.embed_dim, nhead=8, batch_first=True, norm_first=True),
        #     num_layers=6
        # )

        # Relation DRGG head
        self.use_position_emb = use_position_emb
        self.use_class_emb = use_class_emb
        self.use_bbox_emb = use_bbox_emb
        self.use_tf_enc = use_tf_enc
        self.use_text_emb = use_text_emb
        self.relation_loss_weight = relation_loss_weight
        num_feature_levels = len(self.backbone._out_features)
        self.relation_head = DRGGHead(
            embed_dim=self.embed_dim,
            input_channels=self.channels,
            num_feature_levels=num_feature_levels,
            num_classes=self.num_classes,
            use_position_emb=self.use_position_emb,
            use_class_emb=self.use_class_emb,
            use_bbox_emb=self.use_bbox_emb,
            use_text_emb=self.use_text_emb,
            use_tf_enc=self.use_tf_enc,
            text_encoder=text_encoder,
            num_stages=1,
            relation_loss_weight=self.relation_loss_weight,
        )

        self.beam_width = beam_width

    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # backbone + fpn features
        features = self.backbone(images.tensor)  # output feature dict

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        feature_list = [features[f] for f in self.backbone._out_features]

        # box pooling with gt bboxes
        feature_list_per_image = [
            [fmap[i].unsqueeze(0) for fmap in feature_list]
            for i in range(len(gt_boxes))
        ]
        box_features = [
            self.pooler(per_image_features, [boxes])
            for per_image_features, boxes in zip(feature_list_per_image, gt_boxes)
        ]
        box_vectors = [self.box_head(f) for f in box_features]
        box_feats_list = [[v] for v in box_vectors]

        if self.training:
            _, rel_loss = self.relation_head(
                fpn_feats=feature_list,
                box_feats_list=box_feats_list,
                pred_boxes=[t["boxes"] for t in targets],
                pred_classes=[t["labels"] for t in targets],
                targets=gt_instances
            )
        else:
            results = []
            for tgt in gt_instances:
                inst = Instances(tgt.image_size)
                inst.pred_boxes = tgt.gt_boxes
                inst.pred_classes = tgt.gt_classes
                inst.scores = torch.ones(len(inst), device=inst.pred_boxes.device)
                inst.pred_box_indices = torch.arange(len(inst), device=inst.pred_boxes.device)
                if self.relation_head.use_text_emb == True:
                    inst.pred_text_input_ids = tgt.gt_text_input_ids
                    inst.pred_text_attention_mask = tgt.gt_text_attention_mask
                results.append({"instances": inst})
            results, _ = self.relation_head(
                fpn_feats=feature_list,
                box_feats_list=box_feats_list,
                proposals=results
            )

        if self.training:
            return rel_loss
        else:
            return results

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.pred_box_indices = topk_boxes[i]
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets
