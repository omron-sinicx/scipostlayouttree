from detrex.config import get_config
from detectron2.layers import ShapeSpec
from .models.gtbbox_r50 import model

# get default config
dataloader = get_config("common/data/scipostlayouttree.py").dataloader
optimizer = get_config("common/optim.py").AdamW
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
lr_multiplier = get_config("common/coco_schedule.py").lr_warmup_cosine_decay_multiplier_24ep
train = get_config("common/train.py").train

# max training iterations
train.max_iter = 11000
train.eval_period = 1000
train.log_period = 20
train.checkpointer.period = 1000

# gradient clipping for training
# train.clip_grad.enabled = True
# train.clip_grad.params.max_norm = 0.1
# train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
# optimizer.lr = 1e-4
# optimizer.betas = (0.9, 0.999)
# optimizer.weight_decay = 1e-4
# optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

optimizer.lr = 2e-4
optimizer.weight_decay = 0.05
optimizer.betas = (0.9, 0.999)
optimizer.params.lr_factor_func = lambda _: 1.0

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 1.0
train.clip_grad.params.norm_type = 2

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# modify neck config
# model.neck.input_shapes = {
#     "p1": ShapeSpec(channels=224),
#     "p2": ShapeSpec(channels=448),
#     "p3": ShapeSpec(channels=896),
# }
model.backbone.in_features = ["res3", "res4", "res5"]

# modify training config
train.init_checkpoint = "./dino_r50_4scale_24ep.pth"
train.load_backbone_only = True
train.output_dir = "./output/dino_r50_4scale_24ep"

model.use_bbox_emb = True
model.use_text_emb = True
model.text_encoder = "allenai/scibert_scivocab_uncased"
model.use_tf_enc = True
