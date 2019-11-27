# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging
import os

import torch
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.metrics.accumulation import VariableAccumulation
import numpy as np

def inference(
        cfg,
        model,
        val_loader
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("template_model.inference")
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model,  # metrics={'accuracy': Accuracy()},
                                            device=device)

    def concat_(a, x):
        if len(a.size()) == 0:
            return a + x.cpu()
        else:
            return torch.cat((a, x.cpu()))

    pred_accumulator = VariableAccumulation(op=concat_, output_transform=lambda x: x[0])
    pred_accumulator.attach(evaluator, 'pred_accumulator')

    label_accumulator = VariableAccumulation(op=concat_, output_transform=lambda x: x[1])
    label_accumulator.attach(evaluator, 'label_accumulator')

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        preds = pred_accumulator.compute()[0]
        labels = label_accumulator.compute()[0]

        np.save(os.path.join(cfg.OUTPUT_DIR, 'preds.npy'), preds)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'labels.npy'), labels)
        print(preds.size())
        # now do whatever you would like with computed metrics

    # adding handlers using `evaluator.on` decorator API
    # @evaluator.on(Events.EPOCH_COMPLETED)
    # def print_validation_results(engine):
    #     metrics = evaluator.state.metrics
    #     avg_acc = metrics['accuracy']
    #     logger.info("Validation Results - Accuracy: {:.3f}".format(avg_acc))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_iter(engine):
        iter = (engine.state.iteration - 1) % len(val_loader) + 1
        logger.info("Iteration[{}/{}]"
                    .format(iter, len(val_loader)))

    evaluator.run(val_loader)
