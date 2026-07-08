"""Use built-in datasets for online continual learning.

In OCL datastreams are irreversible sequences of examples following a
non-stationary data distribution. Learners in OCL can only learn from a single
pass through the datastream but are expected to perform well on any portion of
the datastream.

Portions of the datastream where the data distribution is relatively stationary
are called *tasks*.

A common way to construct an OCL dataset for experimentation is to group the
classes of a classification dataset into tasks. Known as the *class-incremental*
scenario, the learner is presented with a sequence of tasks where each task
contains a new subset of the classes.

For example :class:`SplitMNIST` splits the MNIST dataset into five tasks where
each task contains two classes:

>>> from capymoa.ocl.datasets import SplitMNIST
>>> scenario = SplitMNIST(preload_test=False)
>>> scenario.task_schedule
[{1, 4}, {5, 7}, {9, 3}, {0, 8}, {2, 6}]


To get the usual CapyMOA stream object for training:

>>> instance = scenario.stream.next_instance()
>>> instance
LabeledInstance(
    Schema(SplitMNIST10/5),
    x=[0. 0. 0. ... 0. 0. 0.],
    y_index=4,
    y_label='4'
)

CapyMOA streams flatten the data into a feature vector:

>>> instance.x.shape
(784,)

You can access the PyTorch datasets for each task:

>>> x, y = scenario.test_tasks[0][0]
>>> x.shape
torch.Size([1, 28, 28])
>>> y
1
"""

from ._base import _BuiltInCIScenario
from ._tiny import RotatedTinyMNIST, TinySplitMNIST
from ._vision import (
    DomainCIFAR100,
    RotatedFashionMNIST,
    RotatedMNIST,
    SplitCIFAR10,
    SplitCIFAR100,
    SplitFashionMNIST,
    SplitMNIST,
)
from ._vit import DomainCIFAR100ViT, SplitCIFAR10ViT, SplitCIFAR100ViT

__all__ = [
    "_BuiltInCIScenario",
    "SplitMNIST",
    "RotatedMNIST",
    "TinySplitMNIST",
    "RotatedTinyMNIST",
    "SplitCIFAR100ViT",
    "SplitCIFAR10ViT",
    "DomainCIFAR100ViT",
    "SplitFashionMNIST",
    "RotatedFashionMNIST",
    "SplitCIFAR10",
    "SplitCIFAR100",
    "DomainCIFAR100",
]
