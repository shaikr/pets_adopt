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
import numpy as np
import tqdm
from torch import nn
from torchvision.models import resnet18

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.example_inference import inference
from modeling import build_model
from utils.logger import setup_logger
import torch.nn.functional as F


def resnet18_embedding_model_forward(model, x):
    embedding_model = nn.Sequential(*(list(model.children())[:-1]))
    out = embedding_model(x)
    return out


def embedding(cfg, model, val_loader, logger, output_dir):
    device = cfg.MODEL.DEVICE
    logger.info("Start inferencing")
    for i, batch in tqdm.tqdm(enumerate(val_loader)):
        images, labels, pet_ids = batch
        images.to(device)
        with torch.no_grad():
            embeddings = resnet18_embedding_model_forward(model, images)
            for pet_id, vec in zip(pet_ids, embeddings):
                np.save(os.path.join(output_dir, pet_id + '.npy'), vec.squeeze().numpy())


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
    # model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    print('imageneto')
    # val_loader = make_data_loader(cfg, is_train=False, with_pet_ids=True)
    val_loader = make_data_loader(cfg, is_train=True, with_pet_ids=True, all_data=True)

    embedding(cfg, model, val_loader, logger, output_dir)


if __name__ == '__main__':
    main()
