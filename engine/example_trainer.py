# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage, mean_squared_error
from ignite.contrib.metrics.regression import manhattan_distance
import numpy as np
from ignite.contrib.handlers.visdom_logger import VisdomLogger

from metrics.precision import Precision
from metrics.recall import Recall


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={
        # 'accuracy': Accuracy(),
        'precision': Precision(cfg.THRESHOLD),
        'recall': Recall(cfg.THRESHOLD),
        'ce_loss': Loss(loss_fn)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, 'resnet18_bce', checkpoint_period, n_saved=1000, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))

    from visdom import Visdom
    viz = Visdom()

    @trainer.on(Events.ITERATION_COMPLETED)
    def viz_iteration_loss(engine):
        iteration = engine.state.iteration - 1
        viz.line(np.array([engine.state.metrics['avg_loss']]), np.array([iteration]), win='iter_loss',
                 env='DenseFusionModel',
                 update='append', opts={'title': 'iter_loss'})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['ce_loss']
        recall = metrics['recall']
        precision = metrics['precision']
        logger.info("Training Results - Epoch: {}  Avg Loss: {:.4f}   Recall: {:.4f}   Precision: {:.4f}"
                    .format(engine.state.epoch, avg_loss, recall, precision))
        epoch = engine.state.epoch
        env_name = 'end_task_train'
        viz.line(np.array([avg_loss]), np.array([epoch]), win='train_epoch_loss',
                 env=env_name, update='append', name='train_epoch_loss', opts={'title': 'train_epoch_loss'})
        viz.line(np.array([recall]), np.array([epoch]), win='train_recall',
                 env=env_name, update='append', name='train_recall', opts={'title': 'train_epoch_recall'})
        viz.line(np.array([precision]), np.array([epoch]), win='train_precision',
                 env=env_name, update='append', name='train_precision', opts={'title': 'train_epoch_precision'})
        viz.line(np.array([precision]), np.array([recall]), win='train_ROC',
                 env=env_name, update='append', name='train_ROC', opts={'title': 'train_ROC'})


    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics['ce_loss']
            recall = metrics['recall']
            precision = metrics['precision']
            logger.info("Validation Results - Epoch: {} Avg Loss: {:.4f}   Recall: {:.4f}   Precision: {:.4f}"
                        .format(engine.state.epoch, avg_loss, recall, precision)
                        )
            epoch = engine.state.epoch
            viz.line(np.array([avg_loss]), np.array([epoch]), win='val_epoch_loss',
                     env='DenseFusionModel', update='append', name='val_epoch_loss', opts={'title': 'val_epoch_loss'})
            viz.line(np.array([recall]), np.array([epoch]), win='val_recall',
                     env='DenseFusionModel', update='append', name='val_recall', opts={'title': 'val_epoch_recall'})
            viz.line(np.array([precision]), np.array([epoch]), win='val_precision',
                     env='DenseFusionModel', update='append', name='val_precision', opts={'title': 'val_epoch_precision'})
            viz.line(np.array([precision]), np.array([recall]), win='val_ROC',
                     env='DenseFusionModel', update='append', name='val_ROC', opts={'title': 'val_ROC'})

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
