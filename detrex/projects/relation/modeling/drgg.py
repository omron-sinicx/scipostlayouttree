from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.distributed as dist
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads import CascadeROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures import pairwise_iou
import detectron2.modeling.roi_heads.fast_rcnn as frcnn
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from transformers import AutoModel, AutoTokenizer

from detrex.layers import box_cxcywh_to_xyxy


class DRGGRelationFeatureExtractor(nn.Module):
    def __init__(self, feat_dim, pool_out_dim=32, hidden_dim=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.pool_out_dim = pool_out_dim
        self.hidden_dim = hidden_dim or pool_out_dim
        self.avgpool = nn.AdaptiveAvgPool1d(pool_out_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(pool_out_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.hidden_dim, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (N, feet_dim)
        pooled = self.avgpool(x)        # (N, pool_out_dim)
        pooled = self.mlp1(pooled)      # (N, hidden_dim)
        upsampled = self.mlp2(pooled)   # (N, feat_dim)
        fused = F.relu(upsampled+x)     # (N, feat_dim)
        return fused


class DRGGWeightedFeatureAggregator(nn.Module):
    def __init__(self, num_stages: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_stages))

    def forward(self, feats_list):
        assert len(feats_list) == self.logits.numel()

        stacked = torch.stack(feats_list, dim=0)

        weight = F.softmax(self.logits, dim=0)
        while weight.dim() < stacked.dim():
            weight = weight.unsqueeze(-1)

        out = (weight * stacked).sum(dim=0)
        return out


class DRGGHead(nn.Module):
    def __init__(self, embed_dim, input_channels, num_feature_levels, num_classes, use_position_emb=False, use_class_emb=False, use_bbox_emb=False, use_text_emb=False, use_tf_enc=False, text_encoder=None, num_stages=1, relation_loss_weight=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.num_feature_levels = num_feature_levels
        self.box_dim = 128

        D = self.embed_dim
        C = self.input_channels
        L = self.num_feature_levels
        B = self.box_dim

        self.num_classes = num_classes
        self.use_bbox_emb = use_bbox_emb
        self.use_position_emb = use_position_emb
        self.use_class_emb = use_class_emb
        if self.use_bbox_emb:
            self.use_position_emb = True
            self.use_class_emb = True
        self.use_text_emb = use_text_emb
        self.use_tf_enc = use_tf_enc
        self.num_stages = num_stages

        self.root_proj = nn.Sequential(
            nn.Linear(C*L, C*L),
            nn.ReLU(),
            nn.Linear(C*L, D)
        )
        self.iou_threshold = 0.75

        self.row_drgg_extractor = DRGGRelationFeatureExtractor(D)
        self.col_drgg_extractor = DRGGRelationFeatureExtractor(D)

        if self.use_position_emb == True:
            self.bbox_embed = nn.Sequential(
                nn.Linear(4, B),
                nn.ReLU(),
                nn.Linear(B, B)
            )
            D += B

        if self.use_class_emb == True:
            # +1 for background
            self.class_embed = nn.Embedding(self.num_classes+1, B)
            D += B

        self.bbox_dim = B

        if self.use_text_emb == True:
            if text_encoder is None:
                raise ValueError("text_encoder must be provided if use_text_emb is True")
            else:
                self.text_encoder = AutoModel.from_pretrained(text_encoder, trust_remote_code=False, use_safetensors=True)
            D += self.text_encoder.config.hidden_size

        if self.use_tf_enc == True:
            self.row_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(D, nhead=8, batch_first=True, norm_first=True),
                num_layers=6
            )
            self.col_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(D, nhead=8, batch_first=True, norm_first=True),
                num_layers=6
            )

        self.drgg_aggregator = DRGGWeightedFeatureAggregator(self.num_stages)

        self.relation_head = nn.Sequential(
            nn.Linear(D*2, D*2),
            nn.ReLU(),
            nn.Linear(D*2, 2)
        )

        self.relation_loss_weight = relation_loss_weight

    @staticmethod
    def _hungarian_match(gt_boxes, gt_cls, pred_boxes, pred_cls, iou_thr):
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return {}, {}

        iou_mat = pairwise_iou(gt_boxes, pred_boxes)          # (G, P)
        cls_eq  = (gt_cls.view(-1, 1) == pred_cls.view(1, -1))
        valid   = cls_eq & (iou_mat >= iou_thr)

        cost = np.where(valid.numpy(), 1-iou_mat.numpy(), 1e6)
        row, col = linear_sum_assignment(cost)

        gt2pred, pred2gt = {}, {}
        for g, p in zip(row, col):
            if cost[g, p] < 1e6:
                gt2pred[int(g)] = int(p)
                pred2gt[int(p)] = int(g)
        return gt2pred, pred2gt, iou_mat

    def forward(self, fpn_feats, box_feats_list, pred_boxes=None, pred_logits=None, pred_classes=None, targets=None, proposals=None):
        outputs = defaultdict(list)
        losses = {}

        if self.training:
            if pred_classes is None:
                pred_classes = pred_logits.argmax(dim=-1)
            rel_loss = self._relation_loss(fpn_feats, box_feats_list, pred_boxes, pred_classes, targets)
            losses.update(rel_loss)
        else:
            outputs = self._relation_inference(fpn_feats, box_feats_list, proposals)

        return outputs, losses

    def _relation_loss(self, fpn_feats, box_feats_list, pred_boxes_list, pred_classes_list, targets):
        device = fpn_feats[0].device
        D = self.embed_dim
        loss_p = torch.tensor(0.0, device=device)
        loss_n = torch.tensor(0.0, device=device)
        acc_p = torch.tensor(0.0, device=device)
        acc_n = torch.tensor(0.0, device=device)
        match_rate = torch.tensor(0.0, device=device)
        num_valid_samples = 0

        for img_idx, target in enumerate(targets):
            image_height, image_width = target.image_size

            # (1,D)
            root = self.root_proj(
                torch.cat([f[img_idx].flatten(1).mean(-1) for f in fpn_feats])
            ).unsqueeze(0)

            feats_img_list = box_feats_list[img_idx]
            pred_boxes = pred_boxes_list[img_idx]
            pred_classes = pred_classes_list[img_idx]

            gt_boxes = target.gt_boxes
            gt_boxes.tensor[:, 0] /= image_width
            gt_boxes.tensor[:, 1] /= image_height
            gt_boxes.tensor[:, 2] /= image_width
            gt_boxes.tensor[:, 3] /= image_height
            gt_classes = target.gt_classes
            pred_boxes = Boxes(box_cxcywh_to_xyxy(pred_boxes_list[img_idx].detach().clone()))
            pred_classes = pred_classes_list[img_idx]
            gt2pred, pred2gt, iou_mat = self._hungarian_match(
                gt_boxes=gt_boxes.to("cpu"),
                gt_cls=gt_classes.to("cpu"),
                pred_boxes=pred_boxes.to("cpu"),
                pred_cls=pred_classes.to("cpu"),
                iou_thr=self.iou_threshold
            )

            G = len(gt_boxes)

            ROOT_ID = 0

            gt_parent = target.gt_parents + 1  # (G,)
            gt_next = target.gt_next + 1       # (G,)

            best_idx = []
            tgt_parent = []
            tgt_next = []
            temp_pred_boxes = []
            is_detected = torch.ones(G+1, dtype=torch.bool, device=device)
            for g in range(G):
                if g in gt2pred:
                    pred_idx = gt2pred[g]
                    best_idx.append(pred_idx)
                    tgt_parent.append(gt_parent[g])
                    tgt_next.append(gt_next[g])
                    temp_pred_boxes.append(pred_boxes.tensor[pred_idx])
                else:
                    best_idx.append(-1)
                    tgt_parent.append(G)
                    tgt_next.append(ROOT_ID)
                    temp_pred_boxes.append(torch.zeros(4, device=device))
                    is_detected[g+1] = 0.0

            tgt_parent = torch.cat(
                [
                    torch.tensor([G], device=device),
                    torch.tensor(tgt_parent, device=device)
                ],
                dim=0
            )
            tgt_next = torch.cat(
                [
                    torch.tensor([1], device=device),
                    torch.tensor(tgt_next, device=device)
                ],
                dim=0
            )
            pred_boxes = torch.stack(temp_pred_boxes, dim=0)  # (G, 4)

            # BBox Embedding
            if self.use_position_emb:
                root_bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=pred_boxes.dtype, device=device).unsqueeze(0)
                root_position_embed = self.bbox_embed(root_bbox)  # (1, B)
                normalized_pred_boxes = pred_boxes.clone()
                position_embed = self.bbox_embed(normalized_pred_boxes)
                all_position_embed = torch.cat([root_position_embed, position_embed], dim=0)  # (K+1, B)

            # BBox Embedding
            if self.use_class_emb:
                root_class_embed = self.class_embed(torch.tensor([self.num_classes], device=device))
                class_embed = self.class_embed(gt_classes)
                all_class_embed = torch.cat([root_class_embed, class_embed], dim=0)

            # Text Embedding
            if self.use_text_emb:
                text_input_ids = target.gt_text_input_ids.to(device)
                text_attention_mask = target.gt_text_attention_mask.to(device)

                text_embed = self.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask
                ).last_hidden_state
                text_embed = (text_embed * text_attention_mask.unsqueeze(-1)).sum(1) / text_attention_mask.sum(1).unsqueeze(-1)
                text_embed = torch.cat([torch.zeros_like(text_embed[0]).unsqueeze(0), text_embed], dim=0)

            Fl_list = []
            for feats_img in feats_img_list:
                F_box = []
                for idx in best_idx:
                    if idx == -1:
                        F_box.append(torch.zeros(D, device=device))
                    else:
                        F_box.append(feats_img[idx])

                F_box = torch.stack(F_box, 0)       # (G, D)
                F_all = torch.cat([root, F_box], 0) # (G+1, D)

                row_feats = self.row_drgg_extractor(F_all)
                col_feats = self.col_drgg_extractor(F_all)

                if self.use_position_emb:
                    row_feats = torch.cat([row_feats, all_position_embed], dim=-1)
                    col_feats = torch.cat([col_feats, all_position_embed], dim=-1)

                if self.use_class_emb:
                    row_feats = torch.cat([row_feats, all_class_embed], dim=-1)
                    col_feats = torch.cat([col_feats, all_class_embed], dim=-1)

                if self.use_text_emb:
                    row_feats = torch.cat([row_feats, text_embed], dim=-1)
                    col_feats = torch.cat([col_feats, text_embed], dim=-1)

                if self.use_tf_enc:
                    row_feats = self.row_encoder(
                        row_feats.unsqueeze(0), src_key_padding_mask=~is_detected.unsqueeze(0)
                    ).squeeze(0)
                    col_feats = self.col_encoder(
                        col_feats.unsqueeze(0), src_key_padding_mask=~is_detected.unsqueeze(0)
                    ).squeeze(0)

                # (G+1, 1, D) → (G+1, G+1, D)
                row_exp = row_feats.unsqueeze(1).expand(-1, G+1, -1)
                # (1, G+1, D) → (G+1, G+1, D)
                col_exp = col_feats.unsqueeze(0).expand(G+1, -1, -1)

                # (G+1, G+1, D*2)
                Fl_concat = torch.cat([row_exp, col_exp], dim=-1)
                Fl_list.append(Fl_concat)

            # (G+1, G+1, D*2)
            F_agg = self.drgg_aggregator(Fl_list)

            # ---------- loss -------------------------------
            logits = self.relation_head(F_agg)
            logits_p = logits[:, :, 0]
            logits_n = logits[:, :, 1]
            loss_p_all = F.cross_entropy(logits_p, tgt_parent.long(), reduction='none')  # shape: (G+1,)
            loss_n_all = F.cross_entropy(logits_n, tgt_next.long(), reduction='none')

            # Root next
            loss_n += loss_n_all[ROOT_ID] * is_detected[ROOT_ID] * is_detected[tgt_next[ROOT_ID]]

            # 1 - G-1
            for g in range(1, G):
                loss_p += loss_p_all[g] * is_detected[g] * is_detected[tgt_parent[g]]
                loss_n += loss_n_all[g] * is_detected[g] * is_detected[tgt_next[g]]

            # G (Last) parent
            loss_p += loss_p_all[G] * is_detected[G] * is_detected[tgt_parent[G]]

            loss_p /= max(1, is_detected.sum()-1)
            loss_n /= max(1, is_detected.sum()-1)

            if is_detected.sum() > 1:
                num_valid_samples += 1

            with torch.no_grad():
                acc_p += ((logits_p[1:].argmax(1) == tgt_parent[1:]) * is_detected[1:]).sum() / G
                acc_n += ((logits_n[:-1].argmax(1) == tgt_next[:-1]) * is_detected[:-1]).sum() / G
                match_rate += len(gt2pred) / G

        return {
            "loss_parents": loss_p / max(1, num_valid_samples) * self.relation_loss_weight,
            "loss_next_idx": loss_n / max(1, num_valid_samples) * self.relation_loss_weight,
            "loss_accuracy_parents": acc_p / max(1, num_valid_samples),
            "loss_accuracy_next_idx": acc_n / max(1, num_valid_samples),
            "loss_match_rate": match_rate / max(1, num_valid_samples),
        }

    @torch.no_grad()
    def _relation_inference(self, fpn_feats, box_feats_list, proposals):
        device = fpn_feats[0].device
        results = []
        for img_idx, inst in enumerate(proposals):
            inst = inst["instances"]
            M = len(inst)
            pos_indices = inst.scores > 0.5
            N = pos_indices.sum().item()

            if N == 0:
                # No predictions passed the score threshold
                inst.pred_parent_logits = torch.full((M, 1), -1e6, device=device)
                inst.pred_next_logits = torch.full((M, 1), -1e6, device=device)
                inst.pred_root_next_logits = torch.full((M,), -1e6, device=device)
                continue

            box_indices = inst.pred_box_indices[pos_indices]

            feats_img_list = [feats[box_indices] for feats in box_feats_list[img_idx]]

            image_height, image_width = inst.image_size
            pred_boxes = inst.pred_boxes[pos_indices].tensor
            pred_classes = inst.pred_classes[pos_indices]

            # BBox Embedding
            if self.use_position_emb:
                root_bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=pred_boxes.dtype, device=device).unsqueeze(0)
                root_position_embed = self.bbox_embed(root_bbox)  # (1, B)
                normalized_pred_boxes = pred_boxes.clone()
                normalized_pred_boxes[:, 0] /= image_width
                normalized_pred_boxes[:, 1] /= image_height
                normalized_pred_boxes[:, 2] /= image_width
                normalized_pred_boxes[:, 3] /= image_height
                position_embed = self.bbox_embed(normalized_pred_boxes)
                all_position_embed = torch.cat([root_position_embed, position_embed], dim=0)  # (N+1, B)

            # BBox Embedding
            if self.use_class_emb:
                root_class_embed = self.class_embed(torch.tensor([self.num_classes], device=device))
                class_embed = self.class_embed(pred_classes)
                all_class_embed = torch.cat([root_class_embed, class_embed], dim=0)

            # Text Embedding
            if self.use_text_emb:
                text_input_ids = inst.pred_text_input_ids.to(device)
                text_attention_mask = inst.pred_text_attention_mask.to(device)

                text_embed = self.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask
                ).last_hidden_state
                text_embed = (text_embed * text_attention_mask.unsqueeze(-1)).sum(1) / text_attention_mask.sum(1).unsqueeze(-1)
                text_embed = torch.cat([torch.zeros_like(text_embed[0]).unsqueeze(0), text_embed], dim=0)

            # (1,D)
            root = self.root_proj(
                torch.cat([f[img_idx].flatten(1).mean(-1) for f in fpn_feats])
            ).unsqueeze(0)

            Fl_list = []
            for feats_img in feats_img_list:
                # (N+1, D)
                F_all = torch.cat([root, feats_img], 0)

                row_feats = self.row_drgg_extractor(F_all)
                col_feats = self.col_drgg_extractor(F_all)

                if self.use_position_emb:
                    row_feats = torch.cat([row_feats, all_position_embed], dim=-1)
                    col_feats = torch.cat([col_feats, all_position_embed], dim=-1)

                if self.use_class_emb:
                    row_feats = torch.cat([row_feats, all_class_embed], dim=-1)
                    col_feats = torch.cat([col_feats, all_class_embed], dim=-1)

                if self.use_text_emb:
                    row_feats = torch.cat([row_feats, text_embed], dim=-1)
                    col_feats = torch.cat([col_feats, text_embed], dim=-1)

                if self.use_tf_enc:
                    row_feats = self.row_encoder(row_feats.unsqueeze(0)).squeeze(0)
                    col_feats = self.col_encoder(col_feats.unsqueeze(0)).squeeze(0)

                # (N+1, 1, D) → (N+1, N+1, D)
                row_exp = row_feats.unsqueeze(1).expand(-1, N+1, -1)
                # (1, N+1, D) → (N+1, N+1, D)
                col_exp = col_feats.unsqueeze(0).expand(N+1, -1, -1)

                # (N+1, N+1, D*2)
                Fl_concat = torch.cat([row_exp, col_exp], dim=-1)
                Fl_list.append(Fl_concat)

            # (N+1, N+1, D*2)
            F_agg = self.drgg_aggregator(Fl_list)

            logits = self.relation_head(F_agg)
            logits_p = logits[:, :, 0]
            logits_n = logits[:, :, 1]

            pad_value = -1e6
            dummy_parent_logits = torch.full((M, N+1), pad_value, device=device)
            dummy_next_logits = torch.full((M, N+1), pad_value, device=device)
            dummy_root_logits = torch.full((M,), pad_value, device=device)

            dummy_parent_logits[pos_indices] = logits_p[1:]
            dummy_next_logits[pos_indices] = logits_n[1:]
            dummy_root_logits[pos_indices] = logits_n[0, 1:]

            inst.pred_parent_logits = dummy_parent_logits
            inst.pred_next_logits = dummy_next_logits
            inst.pred_root_next_logits = dummy_root_logits

        return proposals
