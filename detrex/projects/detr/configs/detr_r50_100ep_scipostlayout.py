from detrex.config import get_config
from .models.detr_r50 import model

dataloader = get_config("common/data/scipostlayout.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "./converted_detr_r50_500ep.pth"
train.output_dir = "./output/detr_r50_100ep"
train.max_iter = 11000
train.eval_period = 1000
train.log_period = 20
train.checkpointer.period = train.eval_period

# modify lr_multiplier
lr_multiplier.scheduler.milestones = [7000, 11000]

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16
dataloader.train.total_batch_size = 64

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

model.num_classes = 9
model.criterion.num_classes = model.num_classes
model.num_queries = 100
