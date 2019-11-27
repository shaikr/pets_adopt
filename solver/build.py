# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch


# def make_optimizer(cfg, model):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#     optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
#     return optimizer

def make_optimizer(cfg, model):
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(model.parameters(), lr=cfg.SOLVER.BASE_LR,
                                                                momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_scheduler(cfg, optimizer):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.LR_DECAY)
    return scheduler
