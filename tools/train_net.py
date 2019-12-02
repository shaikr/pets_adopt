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
from torchsummary import summary
import torch.nn.functional as F

from solver.build import make_scheduler

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.example_trainer import do_train
from modeling import build_model
from solver import make_optimizer

from utils.logger import setup_logger


def train(cfg):
    model = build_model(cfg)
    # summary(model.cuda(), (3, 224,224))
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    arguments = {}

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    def fixed_binary_cross_entropy(input, target):
        one_hot = torch.cuda.FloatTensor(target.size(0), 1, target.size(2), target.size(3)).zero_()
        target = one_hot.scatter_(1, target.data, 1)
        return F.binary_cross_entropy_with_logits(input.squeeze(), target,
                                                  weight=torch.FloatTensor([2, 1]).cuda().repeat(target.shape[0], 1))

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        fixed_binary_cross_entropy,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg)


if __name__ == '__main__':
    main()
