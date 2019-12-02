# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
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

import torch
import numpy as np


def train(cfg):
    model = build_model(cfg)
    # summary(model.cuda(), (3, 224,224))
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    arguments = {}

    label = 'BinaryLabel'

    train_loader = make_data_loader(cfg, is_train=True, label_column=label)
    val_loader = make_data_loader(cfg, is_train=False, label_column=label)

    def fixed_binary_cross_entropy(input, target):
        return F.binary_cross_entropy_with_logits(input.squeeze(), target)

    def fixed_cross_entropy(input, target):
        return F.cross_entropy(input, target.squeeze())

    def CB_loss(logits, labels, samples_per_cls=[2377, 6689], no_of_classes=2, loss_type='sigmoid', beta=0.999):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        labels_one_hot = F.one_hot(labels.squeeze(), no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot.cpu()
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)
        weights = weights.cuda()

        if loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits.squeeze(), target=labels_one_hot, weight=weights)
        elif loss_type == "softmax":
            pred = logits.squeeze().softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        else:
            raise
        return cb_loss

    # loss_fn = fixed_cross_entropy
    loss_fn = CB_loss

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
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
