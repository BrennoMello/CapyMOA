import torch
from torch import Tensor
from torch.nn import functional as F

from capymoa.base import BatchClassifier
from capymoa.ocl.util._replay import ReservoirLogitSampler
from capymoa.ocl.base import TrainTaskAware, TestTaskAware


from typing import Callable

class DER(BatchClassifier, TrainTaskAware, TestTaskAware):
    #TODO: Change the docstring to reflect the new name of the class
    """Continual learning via Dark Experience Replay.

    Dark Experience Replay (DER) [#f0]_ is a replay continual learning
    strategy that combines data augmentation with repeated training on each
    batch to mitigate catastrophic forgetting.

    * Coreset Selection: Reservoir sampling is used to select a fixed-size
      buffer of past examples.

    * Coreset Retrieval: During training, the learner samples uniformly from the
      buffer of past examples.

    * Coreset Exploitation: The learner trains on the current batch of examples
      and the sampled buffer examples, performing multiple optimization steps
      per-batch using random augmentations of the examples. The original paper uses
      RandAugment [#f1]_ for augmentation but any randomized augmentation can be used.
      But the choice of augmentation is important and should be chosen based on the
      problem domain.

    * Not :class:`~capymoa.ocl.base.TrainTaskAware` or
      :class:`~capymoa.ocl.base.TestTaskAware`, but will proxy it to the wrapped
      learner.

    >>> from capymoa.ann import Perceptron
    >>> from capymoa.classifier import Finetune
    >>> from capymoa.ocl.strategy import RAR
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop
    >>> import torchvision.transforms as T
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> scenario = TinySplitMNIST()
    >>> model = Perceptron(scenario.schema)
    >>> # You should use more complex augmentations for more challenging problems.
    >>> augment = T.Compose([
    ...     T.RandomRotation(10),
    ... ])
    >>> learner = RAR(Finetune(scenario.schema, model), augment=augment, repeats=5)
    >>> results = ocl_train_eval_loop(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results.accuracy_final*100:.1f}%")
    45.0%

    Usually more complex augmentations are used such as random crops and
    rotations.

    .. [#f0] Zhang, Yaqian, Bernhard Pfahringer, Eibe Frank, Albert Bifet, Nick
       Jin Sean Lim, and Yunzhe Jia. “A Simple but Strong Baseline for Online
       Continual Learning: Repeated Augmented Rehearsal.” In Advances in Neural
       Information Processing Systems 35: Annual Conference on Neural
       Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA,
       November 28 - December 9, 2022, edited by Sanmi Koyejo, S. Mohamed, A.
       Agarwal, Danielle Belgrave, K. Cho, and A. Oh, 2022.
       https://doi.org/10.5555/3600270.3601344.

    .. [#f1] Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). Randaugment:
       Practical automated data augmentation with a reduced search space. 2020 IEEE/CVF
       Conference on Computer Vision and Pattern Recognition Workshops (CVPRW),
       3008-3017. https://doi.org/10.1109/CVPRW50498.2020.00359
    """

    def __init__(
        self,
        learner: BatchClassifier,
        augment: Callable[[Tensor], Tensor],
        coreset_size: int = 200,
        alpha: float = 0.3,
        repeats: int = 1,
    ) -> None:
        """Initialize Dark Experience Replay.

        :param learner: Underlying learner to be trained with RAR.
        :param coreset_size: Size of the coreset buffer.
        :param repeats: Number of times to repeat training on each batch, defaults to 1.
        """

        super().__init__(learner.schema)
        num_features = learner.schema.get_num_attributes()
        self.learner = learner
        
        self.repeats = repeats
        self.coreset = ReservoirLogitSampler(
            coreset_size,
            num_features,
            learner.schema.get_num_classes(),
            rng=torch.Generator().manual_seed(learner.random_seed),
        )
        self.augment = augment
        self.alpha = alpha
        self.shape = learner.schema.shape

    def train_step(self, x_fresh: Tensor, y_fresh: Tensor) -> None:

        y_fresh = y_fresh.to(self.learner.device, self.learner.y_dtype)
        n = x_fresh.shape[0]

        x_fresh = x_fresh.view(-1, *self.shape)
        x_fresh_aug = self.augment(x_fresh)
        outputs_logits = self.learner.predict_logits(x_fresh_aug)
        loss_fresh = self.learner.criterion(outputs_logits, y_fresh)

        if self.coreset.count != 0:
            x_replay, y_replay, logits_replay = self.coreset.sample(n)
            logits_replay = logits_replay.to(self.learner.device, self.learner.x_dtype)

            x_replay = x_replay.view(-1, *self.shape)
            x_replay_aug = self.augment(x_replay)

            outputs_logits_replay = self.learner.predict_logits(x_replay_aug)
            loss_mse = self.alpha * F.mse_loss(outputs_logits_replay, logits_replay)
            loss_fresh += loss_mse

        self.learner.update_learner(loss_fresh)

        self.coreset.update(x_fresh, y_fresh, outputs_logits.detach())

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        for i in range(self.repeats):
            self.train_step(x, y)
        

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        x = x.to(self.learner.device, self.learner.x_dtype)
        return self.learner.batch_predict_proba(x)

    def on_test_task(self, task_id: int):
        if isinstance(self.learner, TestTaskAware):
            self.learner.on_test_task(task_id)

    def on_train_task(self, task_id: int):
        if isinstance(self.learner, TrainTaskAware):
            self.learner.on_train_task(task_id)
