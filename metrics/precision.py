import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.accuracy import _BaseClassification


class Precision(_BaseClassification):
    """

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) for multilabel cases.
    """

    def __init__(self, threshold, output_transform=lambda x: x, is_multilabel=False):
        self.threshold = threshold
        self.true_positives = None
        self.false_positives = None
        super(Precision, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel)

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        super(Precision, self).reset()

    def update(self, output):
        y_pred_raw, y = output
        y_pred = torch.exp(y_pred_raw) / (torch.exp(y_pred_raw) + 1) > self.threshold
        self.false_positives += torch.all(torch.cat([(y_pred == 1), (y == 0)[:, None]], axis=1), axis=1).sum()
        self.true_positives += torch.all(torch.cat([(y_pred == 1), (y == 1)[:, None]], axis=1), axis=1).sum()

    def compute(self):
        if self.false_positives + self.true_positives == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return float(self.true_positives) / float(self.false_positives + self.true_positives)
