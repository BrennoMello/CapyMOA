import math
import os
import torch
import numpy as np
from torch import Tensor
from typing import List, Optional, Tuple

from capymoa.base import BatchClassifier
from capymoa.ocl.base import TrainTaskAware, TestTaskAware
from capymoa.ocl.util._coreset import ReservoirSampler


class ExperienceReplay(BatchClassifier, TrainTaskAware, TestTaskAware):
    """Experience Replay.

    Experience Replay (ER) [#f0]_ is a replay continual learning strategy.

    * Uses a replay buffer to store past experiences and samples from it during training
      to mitigate catastrophic forgetting.
    * The replay buffer is implemented using reservoir sampling, which allows for
      uniform sampling over the entire stream [#f1]_.
    * Not :class:`capymoa.ocl.base.TrainTaskAware` or
      :class:`capymoa.ocl.base.TestTaskAware`, but will proxy it to the wrapped learner.

    >>> from capymoa.ann import Perceptron
    >>> from capymoa.classifier import Finetune
    >>> from capymoa.ocl.strategy import ExperienceReplay
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> scenario = TinySplitMNIST()
    >>> model = Perceptron(scenario.schema)
    >>> learner = ExperienceReplay(Finetune(scenario.schema, model))
    >>> results = ocl_train_eval_loop(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results.accuracy_final*100:.1f}%")
    33.0%

    .. [#f0] `Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019).
              Experience replay for continual learning. Advances in neural information
              processing systems, 32. <https://arxiv.org/abs/1811.11682>`_
    .. [#f1] `Jeffrey S. Vitter. 1985. Random sampling with a reservoir. ACM Trans. Math.
              Softw. 11, 1 (March 1985), 37–57. <https://doi.org/10.1145/3147.3165>`_
    """

    def __init__(
        self, learner: BatchClassifier, buffer_size: int = 200, repeat: int = 1
    ) -> None:
        """Initialize the Experience Replay strategy.

        :param learner: The learner to be wrapped for experience replay.
        :param buffer_size: The size of the replay buffer, defaults to 200.
        :param repeat: The number of times to repeat the training data in each batch,
            defaults to 1.
        """
        super().__init__(learner.schema, learner.random_seed)
        #: The wrapped learner to be trained with experience replay.
        self.learner = learner
        self._buffer = ReservoirSampler(
            capacity=buffer_size,
            features=self.schema.get_num_attributes(),
            rng=torch.Generator().manual_seed(learner.random_seed),
        )
        self.repeat = repeat

    def batch_train(self, x: Tensor, y: Tensor, train_task_id: int) -> None:
        # update the buffer with the new data
        self._buffer.update(x, y)

        for _ in range(self.repeat):
            # sample from the buffer and construct training batch
            replay_x, replay_y = self._buffer.sample(x.shape[0])
            train_x = torch.cat((x, replay_x), dim=0)
            train_y = torch.cat((y, replay_y), dim=0)
            train_x = train_x.to(self.learner.device, dtype=self.learner.x_dtype)
            train_y = train_y.to(self.learner.device, dtype=self.learner.y_dtype)
            self._log_batches_train(train_y, train_task_id)
            self.learner.batch_train(train_x, train_y)

    def _log_batches_train(self, train_y: Tensor, train_task_id: int):
        # count number of each classes in train_y
        class_counts = train_y.bincount(minlength=self.learner.schema.get_num_classes())
        # log the class counts in a debug file
        # Transform class_counts to a more readable format
        class_counts_str = ",".join(f"{count}" for i, count in enumerate(class_counts))
        os.makedirs("debug", exist_ok=True)
        with open(f"debug/train_batches_y_{self.__class__.__name__}.log", "a") as f:
            f.write(f"{train_task_id},{class_counts_str}\n")

    def batch_predict_proba(self, x: Tensor) -> Tensor:
        x = x.to(self.learner.device, dtype=self.learner.x_dtype)
        return self.learner.batch_predict_proba(x)

    def on_test_task(self, task_id: int):
        if isinstance(self.learner, TestTaskAware):
            self.learner.on_test_task(task_id)

    def on_train_task(self, task_id: int):
        if isinstance(self.learner, TrainTaskAware):
            self.learner.on_train_task(task_id)

    def __str__(self) -> str:
        return f"ExperienceReplay(buffer_size={self._buffer.capacity})"


class ExperienceDelayReplay(BatchClassifier, TrainTaskAware, TestTaskAware):
    """Experience Replay Based on Delayed Importance Sampling.

    Experience Delay Replay (EDR) [#f0]_ is a replay continual learning strategy.

    * Uses a replay buffer to store past experiences and samples from it during training
      to mitigate catastrophic forgetting.
    * The replay buffer is implemented using reservoir sampling, which allows for
      uniform sampling over the entire stream [#f1]_.
    * Not :class:`capymoa.ocl.base.TrainTaskAware` or
      :class:`capymoa.ocl.base.TestTaskAware`, but will proxy it to the wrapped learner.

    >>> from capymoa.ann import Perceptron
    >>> from capymoa.classifier import Finetune
    >>> from capymoa.ocl.strategy import ExperienceReplay
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> scenario = TinySplitMNIST()
    >>> model = Perceptron(scenario.schema)
    >>> learner = ExperienceReplay(Finetune(scenario.schema, model))
    >>> results = ocl_train_eval_loop(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results.accuracy_final*100:.1f}%")
    33.0%

    .. [#f0] `Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019).
              Experience replay for continual learning. Advances in neural information
              processing systems, 32. <https://arxiv.org/abs/1811.11682>`_
    .. [#f1] `Jeffrey S. Vitter. 1985. Random sampling with a reservoir. ACM Trans. Math.
              Softw. 11, 1 (March 1985), 37–57. <https://doi.org/10.1145/3147.3165>`_
    """
    
    def __init__(
        self, learner: BatchClassifier, buffer_size: int = 200, repeat: int = 1,
        k: float = 0.01
    ) -> None:
        """Initialize the Experience Replay strategy.

        :param learner: The learner to be wrapped for experience replay.
        :param buffer_size: The size of the replay buffer, defaults to 200.
        :param repeat: The number of times to repeat the training data in each batch,
            defaults to 1.
        """
        super().__init__(learner.schema, learner.random_seed)
        #: The wrapped learner to be trained with experience replay.
        self.learner = learner
        self._buffer = ReservoirSampler(
            capacity=buffer_size,
            features=self.schema.get_num_attributes(),
            rng=torch.Generator().manual_seed(learner.random_seed),
        )
        self.repeat = repeat
        self._step = 0
        self.k = k
    
    def penalize_imp(self, loss, delay, k=0.01):
        return loss*(math.exp(-self.k * delay))

    def categorical_crossentropy(self, y_true, y_pred):
        # Avoid log(0) by adding a small epsilon value
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.sum(y_true * np.log10(y_pred))
        
        return loss

    #TODO: Test influence of k on the importance
    # Range: 0.01 to 0.1
    def instance_importance(self, true_label, predicted_probs, 
                                delay, k=0.01):
        # When the loss was bigger than 0.9
        #loss = 1 - self.categorical_crossentropy(true_label, predicted_probs)
        # print(f"K value: {k}")
        loss = self.categorical_crossentropy(true_label, predicted_probs)
        
        importance = self.penalize_imp(loss, delay, k)

        return importance
    
    def select_random_indices(self, batches: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        #join all batches
        xb_join = torch.cat([b[0] for b in batches], dim=0)
        yb_join = torch.cat([b[1] for b in batches], dim=0)

        #size of batch
        n = batches[0][0].shape[0]
        count = xb_join.shape[0]
        indices = torch.randint(0, count, (n,))
        
        xb_selected = xb_join[indices]
        yb_selected = yb_join[indices]

        return xb_selected, yb_selected
    
    def batch_train(self, batches: List[Tuple[Tensor, Tensor]],
                    delay: int, task_id: int) -> None:

        self._step += 1
        
        batch_size = batches[0][0].shape[0]
        train_instances = list()

        # if self._buffer.count == 0:
        #     #select random instances to update the reservoir
        #     x_buffer, y_buffer = self.select_random_indices(batches)
        #     x_buffer = x_buffer.view(x_buffer.shape[0], -1)
        #     self._buffer.update(x_buffer, y_buffer)
        
        for instance in batches:

            x_ = instance[0]
            y_ = instance[1]
            x_ = x_.view(x_.shape[0], -1)
            yb_pred_proba = instance[2]
            self._buffer.update(x_, y_)
            if delay > 0:
                # print(f"Batch Delay: {delay}")              
                for j in range(len(y_)):
                    y = y_[j].item()
                    # x = x_[j]
                    # TODO: Generate one hot encoding for the true label
                    num_classes = self.schema.get_num_classes()
                    true_label_one_hot = np.eye(num_classes)[y]
                    # predicted_probs = instance[4][j]
                    predicted_probs = yb_pred_proba[j]

                    instance_importance = self.instance_importance(true_label_one_hot, predicted_probs, delay)
                    # print(f"Instance importance: {instance_importance}")
                    train_instances.append((x_[j], y, instance_importance))
            else:
                # print("No delay for instance, adding to training instances")
                for j in range(len(y_)):
                    y = y_[j].item()
                    train_instances.append((x_[j], y, torch.iinfo(torch.int32).max))
        
        #sort the train instances by importance
        train_instances = sorted(
            train_instances,
            key=lambda item: item[2],  # Sort by importance
            reverse=True,  # Highest importance first
        )

        # if len(train_instances) > batch_size*2:
        #     # If the number of instances is greater than the batch size, we need to sample
        #     # the instances based on their importance
        #     print(f"Number of train instances {len(train_instances)} is greater than batch size {batch_size*2}")
        #     # for instance in train_instances:
        #     #     with open(f"debug/train_instance_{task_id}_{_step}.txt", "a") as f:
        #     #         f.write(f"{instance[2]}\n")

        #     train_instances = train_instances[:batch_size*2]

        #     # for instance in train_instances:
        #     #     with open(f"debug/train_instance_importance_{task_id}_{_step}.txt", "a") as f:
        #     #         f.write(f"{instance[2]}\n")

        train_instances = train_instances[:batch_size]
        
        #####################----------------########################## 
        replay_x, replay_y = self._buffer.sample(batch_size)
        train_x = torch.stack([instance[0] for instance in train_instances], dim=0)
            
        # print(f"Number of train instances: {len(train_instances)}")
        train_x = torch.cat((train_x, replay_x), dim=0).to(self.learner.device)
        train_y = torch.tensor([instance[1] for instance in train_instances])
        train_y = torch.cat((train_y, replay_y), dim=0).to(self.learner.device)
        #####################----------------########################## 
        
        # #  update reservoir
        # for instance in train_instances:
        #     x = instance[0].unsqueeze(0)
        #     y = torch.tensor([instance[1]])
        #     self._buffer.update(x, y)

          
        # select random instances to update the reservoir
        # x_buffer, y_buffer = self.select_random_indices(batches)
        # x_buffer = x_buffer.view(x_buffer.shape[0], -1)
        # self._buffer.update(x_buffer, y_buffer)

        # for instance in batches:

        #     x_up = instance[0]
        #     y_up = instance[1]
        #     x_up = x_up.view(x_up.shape[0], -1)
            
        #     self._buffer.update(x_up, y_up)

        self._log_batches_train(train_y, task_id)
   
        return self.learner.batch_train(train_x, train_y)

    def _log_batches_train(self, train_y: Tensor, train_task_id: int):
        # count number of each classes in train_y
        class_counts = train_y.bincount(minlength=self.learner.schema.get_num_classes())
        # log the class counts in a debug file
        # Transform class_counts to a more readable format
        class_counts_str = ",".join(f"{count}" for i, count in enumerate(class_counts))
        os.makedirs("debug", exist_ok=True)
        with open(f"debug/train_batches_y_{self.__class__.__name__}.log", "a") as f:
            f.write(f"{train_task_id},{class_counts_str}\n")     

    def batch_predict_proba(self, x: Tensor) -> Tensor:
        x = x.to(self.learner.device, dtype=self.learner.x_dtype)
        return self.learner.batch_predict_proba(x)

    def on_test_task(self, task_id: int):
        if isinstance(self.learner, TestTaskAware):
            self.learner.on_test_task(task_id)

    def on_train_task(self, task_id: int):
        if isinstance(self.learner, TrainTaskAware):
            self.learner.on_train_task(task_id)

    def __str__(self) -> str:
        return f"ExperienceReplay(buffer_size={self._buffer.capacity})"
