# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.example_inference import inference
from modeling import build_model
from utils.logger import setup_logger
import torch.nn.functional as F


def resnet18_embedding_model_forward(model, x):
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1)
    return out


def embedding(cfg, model, val_loader, logger):
    device = cfg.MODEL.DEVICE
    logger.info("Start inferencing")
    for i, batch in enumerate(val_loader):
        images, labels = batch
        images.to(device)
        with torch.no_grad():
            embeddings = resnet18_embedding_model_forward(model, images)
        pass


def main():
    num_gpus = 1
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR + '_test'
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    val_loader = make_data_loader(cfg, is_train=False)

    embedding(cfg, model, val_loader, logger)


if __name__ == '__main__':
    main()
