from functools import partial

from ..common.coco_loader_lsj_1024 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

# Pretrained model: EVA02 BSL on COCO
train.init_checkpoint = ""

# Image size: 1024 (same as base config)
model.backbone.net.img_size = 1024
model.backbone.square_pad = 1024
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 768
model.backbone.net.depth = 12
model.backbone.net.num_heads = 12
model.backbone.net.mlp_ratio = 4 * 2 / 3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.1

# 2, 5, 8, 11 for global attention
model.backbone.net.window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10]

# Optimizer: AdamW, Base LR: 5e-7
optimizer.lr = 5e-7
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

# Steps: 10000
train.max_iter = 10000
lr_multiplier.scheduler.milestones = [
    train.max_iter * 8 // 10,
    train.max_iter * 9 // 10,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

# Batch Size: 2
dataloader.test.num_workers = 0
dataloader.train.total_batch_size = 2
