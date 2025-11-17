from detrex.config import get_config
from .deformable_detr_r50_50ep import train, dataloader, optimizer, lr_multiplier, model

# modify model config
model.with_box_refine = True
model.num_classes = 9
model.criterion.num_classes = model.num_classes
model.num_queries = 100

dataloader = get_config("common/data/scipostlayout.py").dataloader

# modify training config
train.init_checkpoint = "./deformable_detr_with_box_refinement_50ep_new.pth"
train.output_dir = "./output/deformable_detr_with_box_refinement_50ep"

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
