import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.accuracy import _BaseClassification


class Recall(_BaseClassification):
    """

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) for multilabel cases.
    """

    def __init__(self, threshold, output_transform=lambda x: x, is_multilabel=False):
        self.threshold = threshold
        self.true_positives = None
        self.false_negatives = None
        super(Recall, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel)

    def reset(self):
        self.true_positives = 0
        self.false_negatives = 0
        super(Recall, self).reset()

    def update(self, output):
        y_pred_raw, y = output
        y_pred = torch.sigmoid(y_pred_raw).squeeze() > self.threshold
        pred_true = y_pred == 1
        pred_false = y_pred == 0
        label_true = y == 1

        self.false_negatives += (pred_false * label_true).sum()
        self.true_positives += (pred_true* label_true).sum()
        # self.false_negatives += torch.all(torch.cat([(y_pred == 0), (y == 1)[:, None]], axis=1), axis=1).sum()
        # self.true_positives += torch.all(torch.cat([(y_pred == 1), (y == 1)[:, None]], axis=1), axis=1).sum()

    def compute(self):
        if self.false_negatives + self.true_positives == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return float(self.true_positives) / float(self.false_negatives + self.true_positives)
