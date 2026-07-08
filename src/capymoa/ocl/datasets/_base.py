from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Generator, Type, cast

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, v2, Lambda, Normalize, ToTensor
from torchvision.transforms.functional import rotate

from capymoa.datasets import get_download_dir, download_unpacked
from capymoa.instance import LabeledInstance
from capymoa.ocl.util.data import (
        (
    class_incremental_schedule,
    class_schedule_to_task_mask,
    get_targets,
    partition_by_schedule, task_free_class_incremental
    ),
)
from capymoa.stream import Stream, TorchStream
from capymoa.stream._stream import Schema

from json import dump as json_dump, load as json_load
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
import requests
import zipfile
import shutil
import os


_SOURCES = {
    "capymoa_tiny_mnist": "https://www.dropbox.com/scl/fi/ry3mqtic4gr02u8kux5yz/capymoa_tiny_mnist.tar.gz?rlkey=khdrktr0ulmjcpbkbhejwfq36&st=0icbomup&dl=1",
    "CIFAR100-vit_base_patch16_224_augreg_in21k": "https://www.dropbox.com/scl/fi/twk8c21xgs5j13xxmcm7q/CIFAR100-vit_base_patch16_224_augreg_in21k.tar.gz?rlkey=xbg7olp440szekvooenes8dhp&st=cznv0q5t&dl=1",
    "CIFAR10-vit_base_patch16_224_augreg_in21k": "https://www.dropbox.com/scl/fi/adxx5u399klcugqk3xlix/CIFAR10-vit_base_patch16_224_augreg_in21k.tar.gz?rlkey=ozfddbomkyt78oyco3c11hz4f&st=xd24ewmr&dl=1",
    "TinyImagenet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "MiniImageNet": "https://www.kaggle.com/api/v1/datasets/download/ctrnngtrung/miniimagenet" 
}


class _PreloadedDataset(TensorDataset):
    def __getitems__(self, indices: Sequence[int]) -> Tuple[Tensor, ...]:
        """Get items from the preloaded dataset."""
        return tuple(tensor[indices] for tensor in self.tensors)

    def collate_fn(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Collate function for PyTorch ``DataLoader``.

        Is the identity function, since the data is already preloaded
        and batched correctly.
        """
        return batch


class _BuiltInCIScenario(ABC):
    """Abstract base class for built-in class incremental OCL datasets.

    This abstract base class is for easily built-in class-incremental continual
    learning datasets.
    """

    task_schedule: Sequence[Set[int]]
    """A sequence of sets containing the classes for each task.

    In online continual learning your learner may not have access to this
    attribute. It is provided for evaluation and debugging.
    """

    num_classes: int
    """The number of classes in the dataset."""

    default_task_count: int
    """The default number of tasks in the dataset."""

    mean: Optional[Sequence[float]]
    """The mean of the features in the dataset used for normalization."""

    std: Optional[Sequence[float]]
    """The standard deviation of the features in the dataset used for normalization."""

    default_train_transform: Optional[Callable[[Any], Tensor]] = ToTensor()
    """The default transform to apply to the dataset."""

    default_test_transform: Optional[Callable[[Any], Tensor]] = ToTensor()
    """The default transform to apply to the dataset."""

    schema: Schema
    """A schema describing the format of the data."""

    stream: Stream[LabeledInstance]
    """Stream containing each task in sequence."""

    shape: Sequence[int]
    """The shape of each input example."""

    task_mask: Tensor
    """A mask for the output for each task of shape (num_tasks, num_classes)"""

    def __init__(
        self,
        num_tasks: Optional[int] = None,
        shuffle_tasks: bool = True,
        shuffle_data: bool = True,
        seed: int = 0,
        directory: Path = get_download_dir(),
        auto_download: bool = True,
        train_transform: Optional[Callable[[Any], Tensor]] = None,
        test_transform: Optional[Callable[[Any], Tensor]] = None,
        normalize_features: bool = False,
        preload_test: bool = True,
        preload_train: bool = False,
        task_option: Optional[str] = None,
    ):
        """Create a new online continual learning datamodule.

        :param num_tasks: The number of tasks to partition the dataset into,
            defaults to :attr:`default_task_count`.
        :param shuffle_tasks: Should the contents and order of the tasks be
            shuffled, defaults to True.
        :param shuffle_data: Should the training dataset be shuffled.
        :param seed: Seed for shuffling the tasks, defaults to 0.
        :param directory: The directory to download the dataset to, defaults to
            :func:`capymoa.datasets.get_download_dir`.
        :param auto_download: Should the dataset be automatically downloaded
            if it does not exist, defaults to True.
        :param train_transform: A transform to apply to the training dataset,
            defaults to :attr:`default_train_transform`.
        :param test_transform: A transform to apply to the test dataset,
            defaults to :attr:`default_test_transform`.
        :param normalize_features: Should the features be normalized. This
            normalization step is after all other transformations.
        :param preload_test: Should the test dataset be preloaded into CPU memory.
            Helps with memory locality and speed, but increases memory usage.
            Preloading the test dataset is recommended since it is small
            and is used multiple times in evaluation.
        :param preload_train: Should the training dataset be preloaded into CPU memory.
            Helps with memory locality and speed, but increases memory usage.
            Preloading the training dataset is not recommended, since it is large
            and each sample is only seen once in online continual learning.
        """
        assert self.num_classes
        assert self.default_task_count

        if num_tasks is None:
            num_tasks = self.default_task_count
        if train_transform is None:
            train_transform = self.default_train_transform
        if test_transform is None:
            test_transform = self.default_test_transform
        if normalize_features and self.mean is not None and self.std is not None:
            normalize = Normalize(self.mean, self.std)
            # If transforms are provided, compose them with the normalization
            # transform. Otherwise, just use the normalization transform.
            train_transform = (
                Compose([train_transform, normalize]) if train_transform else normalize
            )
            test_transform = (
                Compose([test_transform, normalize]) if test_transform else normalize
            )
        elif normalize_features:
            raise ValueError(
                "Cannot normalize features since mean and std are not defined."
            )
        self.num_tasks = num_tasks
        self.task_option = task_option

        generator = torch.Generator().manual_seed(seed)
        self.task_schedule = class_incremental_schedule(
            self.num_classes,
            num_tasks,
            shuffle=shuffle_tasks,
            generator=generator,
        )

        train_dataset = self._download_dataset(
            True, directory, auto_download, train_transform
        )
        test_dataset = self._download_dataset(
            False, directory, auto_download, test_transform
        )
        if self.task_option is not None:
            self.train_tasks = task_free_class_incremental(
                train_dataset,
                shuffle=shuffle_data,
                rng=generator,
            )
            self.test_tasks = task_free_class_incremental(test_dataset)
            print(f"Task-free class-incremental with {len(self.train_tasks)} tasks.")
        else:
            self.train_tasks = partition_by_schedule(
                train_dataset,
                self.task_schedule,
                shuffle=shuffle_data,
                rng=generator,
            )
            self.test_tasks = partition_by_schedule(test_dataset, self.task_schedule)
            print(f"Task-based class-incremental with {len(self.train_tasks)} tasks.")

        if preload_train:
            self.train_tasks = self._preload_datasets(self.train_tasks)
        if preload_test:
            self.test_tasks = self._preload_datasets(self.test_tasks)

        # Create streams for training and testing
        self.stream = TorchStream.from_classification(
            ConcatDataset(self.train_tasks),
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=str(self),
            shape=self.shape,
        )
        self.schema = self.stream.get_schema()
        self.task_mask = class_schedule_to_task_mask(
            self.task_schedule,
            self.num_classes,
        )

    @staticmethod
    def _preload_datasets(
        datasets: Sequence[Dataset[Tuple[Tensor, Tensor]]],
    ) -> Sequence[TensorDataset]:
        """Preload a sequence of datasets into memory.

        :param datasets: A sequence of datasets to preload.
        :return: A sequence of TensorDatasets containing the preloaded data.
        """
        return [_BuiltInCIScenario._preload_dataset(dataset) for dataset in datasets]

    @staticmethod
    def _preload_dataset(dataset: Dataset[Tuple[Tensor, Tensor]]) -> TensorDataset:
        """Preload the dataset into memory.

        :param dataset: The dataset to preload.
        :return: A TensorDataset containing the preloaded data.
        """
        xs, ys = zip(*dataset)
        return _PreloadedDataset(torch.stack(xs), torch.tensor(ys))

    @classmethod
    @abstractmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.num_classes}/{self.num_tasks}"

    def train_loaders(
        self,
        batch_size: int,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> Sequence[DataLoader[Tuple[Tensor, Tensor]]]:
        """Get the training streams for the scenario.

        * The order of the tasks is fixed and does not change between iterations.
          The datasets themselves are shuffled in :func:`__init__` if `shuffle_data`
          is set to True. This is because the order of data is important in
          online learning since the learner can only see each example once.

        :param batch_size: Collects vectors in batches of this size.
        :param kwargs: Additional keyword arguments to pass to the DataLoader.
        :return: A data loader for each task.
        """
        return cast(
            List[DataLoader[Tuple[Tensor, Tensor]]],
            [
                DataLoader(
                    task,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    **kwargs,
                    collate_fn=getattr(task, "collate_fn", None),
                )
                for task in self.train_tasks
            ],
        )

    def test_loaders(
        self,
        batch_size: int,
        **kwargs: Any,
    ) -> Sequence[DataLoader[Tuple[Tensor, Tensor]]]:
        """Get the training streams for the scenario.

        :param batch_size: Collects vectors in batches of this size.
        :param kwargs: Additional keyword arguments to pass to the DataLoader.
        :return: A data loader for each task.
        """
        return cast(
            List[DataLoader[Tuple[Tensor, Tensor]]],
            [
                DataLoader(
                    task,
                    batch_size=batch_size,
                    shuffle=False,
                    **kwargs,
                    collate_fn=getattr(task, "collate_fn", None),
                )
                for task in self.test_tasks
            ],
        )


class _TorchVisionDownload:
    """Shared torchvision dataset downloader for classification scenarios."""

    dataset_type: Type[Dataset]
    shape = [1, 28, 28]

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        dataset_type = cast(Any, cls.dataset_type)
        return dataset_type(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
            target_transform=target_transform,
        )


class _BuiltInRotatedDomainScenario(_BuiltInCIScenario):
    """Base class for domain-incremental scenarios with fixed task rotations."""

    default_rotation_max = 180.0

    def __init__(
        self,
        num_tasks: Optional[int] = None,
        rotations: Optional[Sequence[float]] = None,
        shuffle_data: bool = True,
        seed: int = 0,
        directory: Path = get_download_dir(),
        auto_download: bool = True,
        train_transform: Optional[Callable[[Any], Tensor]] = None,
        test_transform: Optional[Callable[[Any], Tensor]] = None,
        normalize_features: bool = False,
        preload_test: bool = True,
        preload_train: bool = False,
    ):
        if num_tasks is None:
            num_tasks = self.default_task_count
        if num_tasks <= 0:
            raise ValueError("Number of tasks should be greater than 0")

        if train_transform is None:
            train_transform = self.default_train_transform
        if test_transform is None:
            test_transform = self.default_test_transform

        if rotations is None:
            step = self.default_rotation_max / num_tasks
            rotations = tuple(i * step for i in range(num_tasks))
        elif len(rotations) != num_tasks:
            raise ValueError("Length of `rotations` must match `num_tasks`")

        normalize = None
        if normalize_features and self.mean is not None and self.std is not None:
            normalize = Normalize(self.mean, self.std)
        elif normalize_features:
            raise ValueError(
                "Cannot normalize features since mean and std are not defined."
            )

        self.num_tasks = num_tasks
        self.rotations = tuple(float(angle) for angle in rotations)
        all_classes = set(range(self.num_classes))
        self.task_schedule = [set(all_classes) for _ in range(self.num_tasks)]

        generator = torch.Generator().manual_seed(seed)
        self.train_tasks = []
        self.test_tasks = []
        for angle in self.rotations:
            train_dataset = self._download_dataset(
                True,
                directory,
                auto_download,
                self._task_transform(angle, train_transform, normalize),
class TinySplitMNIST(_BuiltInCIScenario):
    """A lower resolution and smaller version of the SplitMNIST dataset for testing.

    You should use :class:`SplitMNIST` instead, this dataset is intended for testing
    and documentation purposes.

    - 16x16 resolution
    - 100 training samples per class
    - 20 testing samples per class
    - 10 classes
    - 5 tasks
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.1307]
    std = [0.3081]
    default_train_transform = None
    default_test_transform = None
    _dataset_key = "capymoa_tiny_mnist"
    shape = [1, 16, 16]

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        ((train_x, train_y), (test_x, test_y)) = download_numpy_dataset(
            dataset_name=cls._dataset_key,
            url=_SOURCES[cls._dataset_key],
            auto_download=auto_download,
            downloads=directory,
        )
        if train:
            return TensorDatasetWithTransform(
                torch.from_numpy(train_x).float().unsqueeze(1) / 255.0,
                torch.from_numpy(train_y).long(),
                transform=transform,
            )
        else:
            return TensorDatasetWithTransform(
                torch.from_numpy(test_x).float().unsqueeze(1) / 255.0,
                torch.from_numpy(test_y).long(),
                transform=transform,
            )


class SplitCIFAR100ViT(_BuiltInCIScenario):
    """CIFAR100 encoded by a Vision Transformer (ViT).

    * Encoded using the ``vit_base_patch16_224_augreg_in21k`` pre-trained
      backbone [1]_.
    * 768 dimensional features (extracted from the last layer of the ViT).
    * 100 classes.
    * 50,000 training samples
    * 10,000 testing samples
    * Useful for developing and evaluating prototype based continual
      learning algorithms.

    ..  [1] Model card for ``vit_base_patch16_224.augreg_in21k``
        https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k
    """

    num_classes = 100
    default_task_count = 10
    default_train_transform = None
    default_test_transform = None
    shape = [768]
    _dataset_key = "CIFAR100-vit_base_patch16_224_augreg_in21k"

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        ((train_x, train_y), (test_x, test_y)) = download_numpy_dataset(
            dataset_name=cls._dataset_key,
            url=_SOURCES[cls._dataset_key],
            auto_download=auto_download,
            downloads=directory,
        )
        if train:
            return TensorDatasetWithTransform(
                torch.from_numpy(train_x).float(),
                torch.from_numpy(train_y).long(),
                transform=transform,
            )
            if shuffle_data:
                train_dataset = self._shuffle_dataset(train_dataset, generator)

            test_dataset = self._download_dataset(
                False,
                directory,
                auto_download,
                self._task_transform(angle, test_transform, normalize),
            )
            self.train_tasks.append(train_dataset)
            self.test_tasks.append(test_dataset)

        if preload_train:
            self.train_tasks = self._preload_datasets(self.train_tasks)
        if preload_test:
            self.test_tasks = self._preload_datasets(self.test_tasks)

        self.stream = TorchStream.from_classification(
            ConcatDataset(self.train_tasks),
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=str(self),
            shape=self.shape,
        )
        self.schema = self.stream.get_schema()
        self.task_mask = class_schedule_to_task_mask(
            self.task_schedule,
            self.num_classes,
        )

    @staticmethod
    def _task_transform(
        angle: float,
        base_transform: Optional[Callable[[Any], Tensor]],
        normalize: Optional[Normalize],
    ) -> Callable[[Any], Tensor]:
        transforms: list[Callable[[Any], Any]] = []
        if base_transform is not None:
            transforms.append(base_transform)
        transforms.append(Lambda(lambda x, a=angle: rotate(x, a)))
        if normalize is not None:
            transforms.append(normalize)
        return Compose(transforms)

    @staticmethod
    def _shuffle_dataset(
        dataset: Dataset[Tuple[Tensor, Tensor]],
        generator: torch.Generator,
        else:
            return TensorDatasetWithTransform(
                torch.from_numpy(test_x).float(),
                torch.from_numpy(test_y).long(),
                transform=transform,
            )


class SplitCIFAR10ViT(SplitCIFAR100ViT):
    """CIFAR10 encoded by a Vision Transformer (ViT).

    * Encoded using the ``vit_base_patch16_224_augreg_in21k`` pre-trained
      backbone [1]_.
    * 768 dimensional features (extracted from the last layer of the ViT).
    * 10 classes.
    * 50,000 training samples
    * 10,000 testing samples
    * Useful for developing and evaluating prototype based continual learning
      algorithms.

    ..  [1] Model card for ``vit_base_patch16_224.augreg_in21k``
        https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k
    """

    _dataset_key = "CIFAR10-vit_base_patch16_224_augreg_in21k"

    num_classes = 10
    default_task_count = 5
    shape = [768]


class SplitFashionMNIST(_BuiltInCIScenario):
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

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        return datasets.FashionMNIST(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
        )


class SplitCIFAR10(_BuiltInCIScenario):
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

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        return datasets.CIFAR10(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
        )

class _CustomDataLoader(DataLoader):
    def __getitem__(self, index):
        return self.dataset[index]
    
class SplitCIFAR100(_BuiltInCIScenario):
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

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        return datasets.CIFAR100(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
        )


class _CustomDataLoader(DataLoader):
    def __getitem__(self, index):
        return self.dataset[index]
    

class SplitTinyImagenet(_BuiltInCIScenario):
    _dataset_key = "tiny-imagenet-200"
    num_classes = 200
    default_task_count = 100
    labels_to_wnids: dict[int, str] = {}
    
    _num_workers = 64
    
    @classmethod
    def _download_dataset(cls, train, directory, auto_download, transform):
        try:
            path = download_unpacked(_SOURCES["TinyImagenet"], get_download_dir())
            tmp_path = path / cls._dataset_key
            
            for file in os.listdir(tmp_path):
                os.rename(tmp_path/file, path/file)
            
            os.rmdir(tmp_path)

            val = path/"val"
            with open(val/'val_annotations.txt') as v:
                for fn, lb in (l.strip().split()[:2] for l in v.readlines()):
                    if not os.path.exists(val/lb):
                        os.mkdir(val/lb)
                    os.rename(val/f'images/{fn}', val/lb/fn)
            
            os.rmdir(val/"images")
            shutil.rmtree(path/"test", True)
        
        except FileExistsError:
            path = get_download_dir() / cls._dataset_key

        if train:
            ds = datasets.ImageFolder(path/"train", transform)
        else:
            ds = datasets.ImageFolder(path/"val", transform)
        
        return _CustomDataLoader(ds)

class SplitMiniImagenet(_BuiltInCIScenario):
    """Split MiniImagenet dataset for online class incremental learning.

    **References:**

    #
    """
    num_classes = 100
    default_task_count = 10
    mean = [0.485, 0.456, 0.406]  
    std = [0.229, 0.224, 0.225]  
    default_train_transform = v2.Compose([v2.Resize((244, 244)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    default_test_transform = v2.Compose([v2.Resize((244, 244)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    shape = [3, 244, 244]

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        
        if 'imagenet' not in os.listdir(directory):
            cls._download_and_extract(_SOURCES["MiniImageNet"], directory)

        if train:
            train = Dataset(directory/"imagenet/train", transform = transform)
            
            return train
        else:
            test = Dataset(directory/"imagenet/test", transform = transform)
            
            return test

    def _download_and_extract(self, url, extract_to):
        """Download the zip file and extract it."""
        
        zip_path = extract_to / "imagenet.zip"
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print("Downloading ImageNet from Kaggle...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        os.remove(zip_path)
        print("Done. Files extracted to:", extract_to)

    
class Dataset(ImageFolder):
    def __init__(self, path, transform=None):
        super().__init__(path, transform=transform)
        self.path = path
        path_idx2label = pd.read_csv(os.path.join(self.path, '..', 'map_clsloc.txt'), sep=' ', header=None)
        self.idx2label = dict(zip(path_idx2label[0], path_idx2label[1]))