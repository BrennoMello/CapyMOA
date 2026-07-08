import math
import os
import torch
import numpy as np
from torch import Tensor, nn
from torchvision.transforms import v2
from typing import List, Optional, Tuple, Literal

from capymoa.base import BatchClassifier
from capymoa.classifier import Finetune
from capymoa.ocl.base import TrainTaskAware, TestTaskAware
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


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
    32.5%

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
            #TODO: refactor shape of buffer sample
            replay_x = replay_x.view(-1, *self._buffer.original_shape)

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
        k: float = 0.01, criterion_loss: nn.Module = nn.CrossEntropyLoss()
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
        self.device = learner.device
        self.criterion_loss = criterion_loss
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
        # loss = 1 - self.categorical_crossentropy(true_label, predicted_probs)
        # print(f"K value: {k}")
        # loss_old = self.categorical_crossentropy(true_label, predicted_probs)

        if isinstance(predicted_probs, np.ndarray):
            predicted_probs = torch.from_numpy(predicted_probs).float()
        
        true_label = torch.tensor([true_label], dtype=torch.long).to(self.device)

        # Confirm device
        predicted_probs = predicted_probs.to(self.device)
        if predicted_probs.dim() == 1:
            predicted_probs = predicted_probs.unsqueeze(0)
        
        #TODO: Develop ACE loss to calculate importance
        loss = self.criterion_loss(predicted_probs, true_label)
        
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
                    # y = y_[j].item()
                    y = y_[j]
                    # x = x_[j]
                    
                    # TODO: Generate one hot encoding for the true label
                    # num_classes = self.schema.get_num_classes()
                    # true_label_one_hot = np.eye(num_classes)[y]
                    
                    # predicted_probs = instance[4][j]
                    predicted_probs = yb_pred_proba[j]

                    instance_importance = self.instance_importance(y, predicted_probs, delay)

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

    def batch_mixed_train(self, batches: List[Tuple[Tensor, Tensor]],
                         task_id: int) -> None:

        self._step += 1
        
        batch_size = batches[0][0].shape[0]
        train_instances = list()

        for instance in batches:

            x_ = instance[0]
            y_ = instance[1]
            
            # x_ = x_.view(x_.shape[0], -1)
            
            yb_pred_proba = instance[2]
            self._buffer.update(x_, y_)
            delay = instance[3]
            if delay > 0:
                # if delay == 1:
                #     print(f"Batch Delay: {delay}")

                for j in range(len(y_)):
                    y = y_[j].item()
                    # x = x_[j]
                    
                    # TODO: Generate one hot encoding for the true label
                    # num_classes = self.schema.get_num_classes()
                    # true_label_one_hot = np.eye(num_classes)[y]
                    
                    # predicted_probs = instance[4][j]
                    predicted_probs = yb_pred_proba[j]

                    instance_importance = self.instance_importance(y, predicted_probs, delay)

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
        # print(f"Sorted train instances: {len(train_instances)}")

        train_instances = train_instances[:batch_size]
        
        #####################----------------########################## 
        replay_x, replay_y = self._buffer.sample(batch_size)
        replay_x = replay_x.view(-1, *self._buffer.original_shape)
        train_x = torch.stack([instance[0] for instance in train_instances], dim=0)
            
        # print(f"Final train instances: {len(train_instances)}")
        train_x = torch.cat((train_x, replay_x), dim=0).to(self.learner.device)
        train_y = torch.tensor([instance[1] for instance in train_instances])
        train_y = torch.cat((train_y, replay_y), dim=0).to(self.learner.device)
        #####################----------------########################## 
        
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

    def batch_predict_logits(self, x: Tensor) -> Tensor:
        x = x.to(self.learner.device, dtype=self.learner.x_dtype)
        return self.learner.predict_logits(x)
    
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


class ExperienceReplayAsymmetricCrossEntropy(ExperienceReplay):
    class ACELoss(nn.CrossEntropyLoss):
        def __init__(self, device, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen_so_far = torch.LongTensor(size=(0,)).to(device)
            self.not_first = False

        def forward(self, logits: Tensor, target: Tensor) -> Tensor:
            present = target.unique()
            self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

            mask = torch.zeros_like(logits)
            mask[:, present] = 1
            mask[:, self.seen_so_far.max():] = 1

            if self.not_first:
                logits  = logits.masked_fill(mask == 0, -1e9)

            else:
                self.not_first = True
            
            loss = super().forward(logits, target)

            return loss

    def __init__(
        self, schema: Schema, model: nn.Module, device: Literal['cpu', 'cuda'], buffer_size: int = 200, repeat: int = 1
    ):
        learner = Finetune(schema, model, device=device)
        learner.criterion = self.ACELoss(learner.device)
        super().__init__(learner, buffer_size, repeat)



class ExperienceReplayACE(ExperienceReplay):

    def __init__(
        self, learner: BatchClassifier, device: Literal['cpu', 'cuda'], 
        buffer_size: int = 200, use_augs: bool = True,
        repeat: int = 1
    ):
        super().__init__(learner, buffer_size, repeat)
        self.device = device
        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.device)
        self.use_augs = use_augs
        self.train_tf_init = False 
        
    def batch_train(self, x: Tensor, y: Tensor, train_task_id: int) -> None:
        if not self.train_tf_init:
            self.train_tf = self._train_transforms(x.shape[2])
            self.train_tf_init = True
        
        for _ in range(self.repeat):

            inc_loss = self.process_inc(x, y, train_task_id)

            re_loss = 0
            if self._buffer.count > 0:
                if train_task_id > 0:
                    replay_x, replay_y = self._buffer.sample(x.shape[0])
                    replay_x = replay_x.view(-1, *self._buffer.original_shape).to(self.device)
                    replay_y = replay_y.to(self.device)
                    re_loss = self.process_re(replay_x, replay_y)

            self.learner.update_learner(inc_loss + re_loss)

        self._buffer.update(x, y)

    def _train_transforms(self, H: Optional[int] = None) -> nn.Module:
        # num_attributes = self.learner.schema.get_num_attributes()
        # H = int(math.sqrt(num_attributes))
               
        # if self.use_augs:
        #     tfs = v2.Compose([
        #         v2.RandomCrop(size=(H, H), padding=4, fill=-1),
        #         v2.RandomHorizontalFlip(p=0.5),
        #     ])
        # else:
        #     tfs = v2.Identity()

        if self.use_augs:
            tfs = nn.Sequential(
                v2.RandomCrop(size=(H, H), padding=4, fill=-1),
                v2.RandomHorizontalFlip(p=0.5),
            )
        else:
            tfs = nn.Identity()

        return tfs

    def process_re(self, x: Tensor, y: Tensor):
        """ get a loss signal from data """

        aug_data = self.train_tf(x)

        pred     = self.learner.predict_logits(aug_data)
        loss     = self.learner.criterion(pred, y)
        
        return loss


    def process_inc(self, x: Tensor, y: Tensor, train_task_id: int):
        """ get loss from incoming data """

        aug_data = self.train_tf(x)

        present = y.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        # process data
        # x = x.view(batch_size, -1)
        # aug_data = aug_data.view(aug_data.shape[0], -1)
        logits = self.learner.predict_logits(aug_data)
        mask   = torch.zeros_like(logits)

        # unmask current classes
        mask[:, present] = 1

        # unmask unseen classes
        mask[:, self.seen_so_far.max():] = 1

        if train_task_id > 0:
            logits  = logits.masked_fill(mask == 0, -1e9)

        loss = self.learner.criterion(logits, y)

        return loss

class ER_ACE(ExperienceReplay):
    def __init__(self, learner, buffer_size = 200, repeat = 1):
        super().__init__(learner, buffer_size, repeat)

class ACELoss(nn.CrossEntropyLoss):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_so_far = torch.LongTensor(size=(0,)).to(device)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        present = target.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        mask = torch.zeros_like(logits)
        mask[:, self.seen_so_far] = 1
        
        logits  = logits.masked_fill(mask == 0, -1e9)
            
        loss = super().forward(logits, target)

        return loss 

    # def forward(self, logits: Tensor, target: Tensor) -> Tensor:
    #     present = target.unique()
    #     self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

    #     mask = torch.zeros_like(logits)
    #     mask[:, present] = 1
    #     mask[:, self.seen_so_far.max():] = 1

    #     logits  = logits.masked_fill(mask == 0, -1e9)
            
    #     loss = super().forward(logits, target)

    #     return loss