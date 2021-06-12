"Author: Sagar Chakraborty"
from ignite.engine import Engine
import os
from torch.autograd import Variable

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from text_cleaning import cleanup_text
from model import RNN

from pydoc import locate
from torch.nn.parallel import DataParallel

from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Precision, Recall, Loss
import glob
import logging

def create_supervised_evaluator(model, inference_fn, metrics={}, cuda=False):
    """
    Factory function for creating an evaluator for supervised models.
    Extended version from ignite's create_supervised_evaluator
    Args:
        model (torch.nn.Module): the model to train
        inference_fn (function): inference function
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    engine = Engine(inference_fn)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class ModelLoader(object):
    def __init__(self, model, dirname, filename_prefix):
        self._dirname = dirname
        self._fname_prefix = filename_prefix
        self._model = model
        self._fname = os.path.join(dirname, filename_prefix)
        self.skip_load = False

        # Ensure model is not None
        if not isinstance(model, nn.Module):
            raise ValueError("model should be an object of nn.Module")

        # Ensure that dirname exists
        if not os.path.exists(dirname):
            self.skip_load = True
            logging.warning(
                "Dir '{}' is not found, skip restoring model".format(dirname)
            )

        if len(glob.glob(self._fname + "_*")) == 0:
            self.skip_load = True
            logging.warning(
                "File '{}-*.pth' is not found, skip restoring model".format(self._fname)
            )

    def _load(self, path):
        if not self.skip_load:
            models = sorted(glob.glob(path))
            latest_model = models[-1]

            try:
                if isinstance(self._model, nn.parallel.DataParallel):
                    self._model.module.load_state_dict(torch.load(latest_model))
                else:
                    self._model.load_state_dict(torch.load(latest_model))
                print("Successfull loading {}!".format(latest_model))
            except Exception as e:
                logging.exception(
                    "Something wrong while restoring the model: %s" % str(e)
                )

    def __call__(self, engine, infix_name):
        path = self._fname + "_" + infix_name + "_*"

        self._load(path=path)


