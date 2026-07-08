from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from capymoa.datasets import get_download_dir
from capymoa.ocl.util.data import group_indicies
from capymoa.stream import TorchStream

from ._base import (
    _BuiltInCIScenario,
    _BuiltInRotatedDomainScenario,
    _TorchVisionDownload,
)


class SplitMNIST(_TorchVisionDownload, _BuiltInCIScenario):
    """Split MNIST dataset for online class incremental learning.

    **References:**

    #. LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit
       database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.1307]
    std = [0.3081]
    shape = [1, 28, 28]
    dataset_type = datasets.MNIST


class RotatedMNIST(_TorchVisionDownload, _BuiltInRotatedDomainScenario):
    """Rotated MNIST where each task applies a fixed image rotation.

    .. figure:: /images/RotatedMNIST.jpg
       :alt: RotatedMNIST task illustration.
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.1307]
    std = [0.3081]
    shape = [1, 28, 28]
    dataset_type = datasets.MNIST


class SplitFashionMNIST(_TorchVisionDownload, _BuiltInCIScenario):
    """Split Fashion MNIST dataset for online class incremental learning.

    **References:**

    #. Xiao, H., Rasul, K., & Vollgraf, R. (2017, August 28). Fashion-MNIST:
       a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.286]
    std = [0.353]
    shape = [1, 28, 28]
    dataset_type = datasets.FashionMNIST


class RotatedFashionMNIST(_TorchVisionDownload, _BuiltInRotatedDomainScenario):
    """Domain-incremental FashionMNIST where each task applies a fixed image rotation.

    .. figure:: /images/RotatedFashionMNIST.jpg
       :alt: RotatedFashionMNIST task illustration.
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.286]
    std = [0.353]
    shape = [1, 28, 28]
    dataset_type = datasets.FashionMNIST


class SplitCIFAR10(_TorchVisionDownload, _BuiltInCIScenario):
    """Split CIFAR-10 dataset for online class incremental learning.

    **References:**

    #. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny
       Images.
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.491, 0.482, 0.447]
    std = [0.247, 0.243, 0.262]
    shape = [3, 32, 32]
    dataset_type = datasets.CIFAR10


class SplitCIFAR100(_TorchVisionDownload, _BuiltInCIScenario):
    """Split CIFAR-100 dataset for online class incremental learning.

    **References:**

    #. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny
       Images.
    """

    num_classes = 100
    default_task_count = 10
    mean = [0.507, 0.487, 0.441]
    std = [0.267, 0.256, 0.276]
    shape = [3, 32, 32]
    dataset_type = datasets.CIFAR100


class DomainCIFAR100(_TorchVisionDownload, _BuiltInCIScenario):
    """Domain incremental CIFAR-100 variant with 20 classes per task.

    This dataset has exactly 5 tasks. Each task contains one fine-grained class from
    each CIFAR-100 superclass (20 classes per task), while labels are remapped to the 20
    superclass IDs. For example, the "flowers" superclass contains various types of
    flowers.

    .. figure:: /images/DomainCIFAR100.jpg
       :alt: DomainCIFAR100 task illustration.

    Note that the groupings are subjective based on the original CIFAR-100's coarse
    labels.

    **References:**

    #. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
    """

    _CIFAR100_CLASS_TO_SUPERCLASS: List[int] = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]  # fmt: skip
    _CIFAR100_SUPERCLASS_CLASSES: List[List[int]] = [[4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [54, 62, 70, 82, 92], [9, 10, 16, 28, 61], [0, 51, 53, 57, 83], [22, 39, 40, 86, 87], [5, 20, 25, 84, 94], [6, 7, 14, 18, 24], [3, 42, 43, 88, 97], [12, 17, 37, 68, 76], [23, 33, 49, 60, 71], [15, 19, 21, 31, 38], [34, 63, 64, 66, 75], [26, 45, 77, 79, 99], [2, 11, 35, 46, 98], [27, 29, 44, 78, 93], [36, 50, 65, 74, 80], [47, 52, 56, 59, 96], [8, 13, 48, 58, 90], [41, 69, 81, 85, 89]]  # fmt: skip
    """CIFAR100 superclasses as defined in the original dataset.
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    classes = [
        "aquatic_mammals",
        "fish",
        "flowers",
        "food_containers",
        "fruit_and_vegetables",
        "household_electrical_devices",
        "household_furniture",
        "insects",
        "large_carnivores",
        "large_man-made_outdoor_things",
        "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores",
        "medium_mammals",
        "non-insect_invertebrates",
        "people",
        "reptiles",
        "small_mammals",
        "trees",
        "vehicles_1",
        "vehicles_2",
    ]
    """The 20 superclasses of CIFAR-100, which are used as the labels in this scenario."""

    num_classes = 20
    default_task_count = 5
    mean = [0.507, 0.487, 0.441]
    std = [0.267, 0.256, 0.276]
    shape = [3, 32, 32]
    dataset_type = datasets.CIFAR100

    def __init__(
        self,
        shuffle_data: bool = True,
        seed: int = 0,
        directory: Path = get_download_dir(),
        auto_download: bool = True,
        train_transform: Optional[Callable[[Any], Tensor]] = None,
        test_transform: Optional[Callable[[Any], Tensor]] = None,
        normalize_features: bool = False,
    ):
        """Create the DomainCIFAR100 scenario.

        This scenario always uses 5 tasks and 20 superclass labels. Each task
        contains one fine-grained class from each superclass.

        :param shuffle_data: If True, shuffles class order within each
            superclass before forming tasks, and shuffles samples within each
            task for training.
        :param seed: Random seed for reproducible shuffling.
        :param directory: Directory where CIFAR-100 is stored/downloaded.
        :param auto_download: If True, downloads CIFAR-100 when missing.
        :param train_transform: Optional transform applied to training images.
        :param test_transform: Optional transform applied to test images.
        :param normalize_features: If True, applies dataset normalization after
            the provided transforms.
        """
        if train_transform is None:
            train_transform = ToTensor()
        if test_transform is None:
            test_transform = ToTensor()

        if normalize_features and self.mean is not None and self.std is not None:
            normalize = Normalize(self.mean, self.std)
            train_transform = Compose([train_transform, normalize])
            test_transform = Compose([test_transform, normalize])
        elif normalize_features:
            raise ValueError(
                "Cannot normalize features since mean and std are not defined."
            )

        self.num_tasks = self.default_task_count
        all_classes = set(range(self.num_classes))
        self.task_schedule = [set(all_classes) for _ in range(self.num_tasks)]

        generator = torch.Generator().manual_seed(seed)

        def target_transform(y: Any) -> int:
            return self._CIFAR100_CLASS_TO_SUPERCLASS[int(y)]

        train_dataset = self._download_dataset(
            True,
            directory,
            auto_download,
            train_transform,
            target_transform=target_transform,
        )
        test_dataset = self._download_dataset(
            False,
            directory,
            auto_download,
            test_transform,
            target_transform=target_transform,
        )

        superclass_classes = torch.asarray(self._CIFAR100_SUPERCLASS_CLASSES)
        # shuffle the order of classes
        if shuffle_data:
            for i, classes in enumerate(superclass_classes):
                superclass_classes[i] = classes[
                    torch.randperm(classes.size(0), generator=generator)
                ]

        self.train_tasks = self._build_domain_tasks(
            train_dataset,
            superclass_classes.T.tolist(),
            shuffle_data,
            generator,
        )
        self.test_tasks = self._build_domain_tasks(
            test_dataset,
            superclass_classes.T.tolist(),
            False,
            generator,
        )

        self.stream = TorchStream.from_classification(
            ConcatDataset(self.train_tasks),
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=str(self),
            shape=self.shape,
            class_names=self.classes,
        )
        self.schema = self.stream.get_schema()
        self.task_mask = torch.ones(
            (self.num_tasks, self.num_classes), dtype=torch.bool
        )

    @staticmethod
    def _build_domain_tasks(
        dataset: Dataset[Tuple[Tensor, Tensor]],
        task_fine_classes: Sequence[Sequence[int]],
        shuffle_data: bool,
        generator: torch.Generator,
    ) -> List[Dataset[Tuple[Tensor, Tensor]]]:
        targets = cast(torch.LongTensor, torch.asarray(dataset.targets))  # type: ignore
        grouped_indices = group_indicies(
            targets, task_fine_classes, shuffle=shuffle_data, rng=generator
        )
        tasks: List[Dataset[Tuple[Tensor, Tensor]]] = []
        for indices in grouped_indices:
            tasks.append(
                cast(
                    Dataset[Tuple[Tensor, Tensor]],
                    Subset(dataset, indices.tolist()),
                )
            )
        return tasks
