from detrex.config import get_config
from .deformable_detr_r50_50ep import train, dataloader, optimizer, lr_multiplier, model

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep

# modify model config
model.with_box_refine = True
model.as_two_stage = True
model.num_classes = 9
model.criterion.num_classes = model.num_classes
model.num_queries = 100

dataloader = get_config("common/data/scipostlayout.py").dataloader

# modify training config
train.init_checkpoint = "./deformable_detr_r50_two_stage_50ep_new.pth"
train.output_dir = "./output/deformable_detr_r50_two_stage_24ep"
train.max_iter = 11000
train.eval_period = 1000
train.log_period = 20
train.checkpointer.period = train.eval_period

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
