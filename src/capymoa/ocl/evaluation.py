"""Evaluate online continual learning in classification tasks."""

from dataclasses import dataclass
import os
from typing import List, Optional, Sequence, Tuple, Union

from capymoa.ocl.strategy import (
        ExperienceReplay, GDumb, NCM, SLDA
    )

from capymoa.ocl.strategy._experience_replay import ExperienceDelayReplay
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from capymoa.base import BatchClassifier, Classifier
from capymoa.evaluation.evaluation import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
    start_time_measuring,
    stop_time_measuring,
)
from capymoa.evaluation.results import PrequentialResults
from capymoa.instance import Instance, LabeledInstance
from capymoa.ocl.base import TestTaskAware, TrainTaskAware
from capymoa.type_alias import LabelIndex


@dataclass(frozen=True)
class OCLMetrics:
    r"""A collection of metrics evaluating an online continual learner.

    We define some metrics in terms of a matrix :math:`R\in\mathbb{R}^{T \times T}`
    (:attr:`accuracy_matrix`) where each element :math:`R_{i,j}` contains the
    the test accuracy on task :math:`j` after sequentially training on tasks
    :math:`1` through :math:`i`.

    Online learning make predictions continuously during training, so we also
    provide "anytime" versions of the metrics. These metrics are collected
    periodically during training. Specifically, :math:`H` times per task.
    The results of this evaluation are stored in a matrix
    :math:`A\in\mathbb{R}^{T \times H \times T}` (:attr:`anytime_accuracy_matrix`)
    where each element :math:`A_{i,h,j}` contains the test accuracy on task
    :math:`j` after sequentially training on tasks :math:`1` through :math:`i-1`
    and step :math:`h` of task :math:`i`.
    """

    anytime_accuracy_all: np.ndarray
    r"""The accuracy on all tasks after training on each step in each task.

    .. math::
    
        a_\text{any all}(t, h) = \frac{1}{T}\sum^T_{i=1} A_{t,h,i}

    We flatten the $t,h$ dimensions to a 1D array. Use
    :attr:`anytime_task_index` to get the corresponding task index for plotting.
    """
    anytime_accuracy_all_avg: float
    r"""The average of :attr:`anytime_accuracy_all` over all tasks.

    .. math::
    
        \bar{a}_\text{any all} = \frac{1}{T}\sum_{t=1}^T \frac{1}{H}\sum_{h=1}^H a_\text{any all}(t, h)

    """
    anytime_accuracy_seen: np.ndarray
    r"""The accuracy on **seen** tasks after training on each step in each task.
    
    .. math::
    
        a_\text{any seen}(t, h) = \frac{1}{t}\sum^t_{i=1} A_{t,h,i}

    We flatten the $t,h$ dimensions to a 1D array. Use
    :attr:`anytime_task_index` to get the corresponding task index for plotting.
    """
    anytime_accuracy_seen_avg: float
    r"""The average of :attr:`anytime_accuracy_seen` over all tasks.

    .. math::
    
        \bar{a}_\text{any seen} = \frac{1}{T}\sum_{t=1}^T \frac{1}{H}\sum_{h=1}^H a_\text{any seen}(t, h)
    """
    anytime_task_index: np.ndarray
    r"""The position in each task where the anytime accuracy was measured."""

    accuracy_all: np.ndarray
    r"""The accuracy on all tasks after training on each task.
    
    .. math::

        a_\text{all}(t) = \frac{1}{T} \sum_{i=1}^{T} R_{t,i}

    Use :attr:`task_index` to get the corresponding task index for plotting.
    """
    accuracy_all_avg: float
    r"""The average of :attr:`accuracy_all` over all tasks.
    
    .. math::

        \bar{a}_\text{all} = \frac{1}{T}\sum_{t=1}^T a_\text{all}(t)
    """
    accuracy_seen: np.ndarray
    r"""The accuracy on **seen** tasks after training on each task.
    
    .. math::

        a_\text{seen}(t) = \frac{1}{t}\sum^t_{i=1} R_{t,i}

    Use :attr:`task_index` to get the corresponding task index for plotting.
    """
    accuracy_seen_avg: float
    r"""The average of :attr:`accuracy_seen` over all tasks.
    
    .. math::

        \bar{a}_\text{seen} = \frac{1}{T}\sum_{t=1}^T a_\text{seen}(t)
    """
    accuracy_final: float
    r"""The accuracy on all tasks after training on the final task.

    .. math::
    
        a_\text{final} = a_\text{all}(T)
    """
    task_index: np.ndarray
    r"""The position of each task in the metrics."""

    forward_transfer: float
    r"""A scalar measuring the impact learning had on future tasks.

    .. math::

       r_\text{FWT} = \frac{2}{T(T-1)}\sum_{i=1}^{T} \sum_{j=i+1}^{T} R_{i,j}
    """
    backward_transfer: float
    r"""A scalar measuring the impact learning had on past tasks.

    .. math::

       r_\text{BWT} = \frac{2}{T(T-1)} \sum_{i=2}^{T} \sum_{j=1}^{i-1} (R_{i,j} - R_{j,j})
    """
    accuracy_matrix: np.ndarray
    r"""A matrix measuring the accuracy on each task after training on each task.
    
    ``R[i, j]`` is the accuracy on task :math:`j` after training on tasks
    :math:`1` through :math:`i`.
    """

    anytime_accuracy_matrix: np.ndarray
    r"""A matrix measuring the accuracy on each task after training on each task and step.
    
    This matrix is :math:`A` with the first two dimensions flattened to a 2D array.
    """

    ttt: PrequentialResults
    """Test-then-train/prequential results."""
    boundaries: np.ndarray
    """Instance index for the boundaries."""
    ttt_windowed_task_index: np.ndarray
    """The position of each window within each task.
    
    Useful as the ``x`` axis for
    :attr:`capymoa.evaluation.results.PrequentialResults.windowed`.
    """


def _backwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return ((R - R.diag()).tril().sum() / (n * (n - 1) / 2)).item()


def _forwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return (R.triu(1).sum() / (n * (n - 1) / 2)).item()


def _get_ttt_windowed_task_index(boundaries: np.ndarray, window_size: int):
    tasks = np.zeros(int(boundaries[-1]) // window_size)
    for task_id, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        win_start = int(start) // window_size
        win_end = int(end) // window_size
        tasks[win_start:win_end] = np.linspace(
            task_id, task_id + 1, win_end - win_start
        )
    return tasks


class _OCLEvaluator:
    """A builder used to collect statistics during online continual learning evaluation."""

    cm: torch.Tensor
    """Confusion 'Matrix' of shape: 
    ``(eval_step_id, train_task_id, test_task_id, true_class, predicted_class)``.
    """

    def __init__(self, task_count: int, eval_step_count: int, class_count: int):
        self.task_count = task_count
        self.class_count = class_count
        self.seen_tasks = 0
        self.step_count = eval_step_count
        self.cm = torch.zeros(
            task_count,
            eval_step_count,
            task_count,
            class_count,
            class_count,
            dtype=torch.int,
        )

    def holdout_update(
        self,
        train_task_id: int,
        eval_step_id: int,
        test_task_id: int,
        y_true: LabelIndex,
        y_pred: Optional[LabelIndex],
    ):
        """Record a prediction when using holdout evaluation."""
        if y_pred is not None:
            self.cm[train_task_id, eval_step_id, test_task_id, y_true, y_pred] += 1
        # TODO: handle missing predictions

    def build(
        self, ttt: PrequentialResults, boundary_instances: torch.Tensor
    ) -> OCLMetrics:
        """Creates metrics using collected statistics."""
        correct = self.cm.diagonal(dim1=3, dim2=4).sum(-1)
        total = self.cm.sum((3, 4))
        anytime_acc = correct / total  # (train task, step_id, test task)
        accuracy_matrix = anytime_acc[:, -1, :]  # (train task, test task)

        anytime_accuracy_seen = torch.zeros(self.task_count, self.step_count)
        anytime_accuracy_all = torch.zeros(self.task_count, self.step_count)
        for t_train in range(self.task_count):
            for s_step in range(self.step_count):
                anytime_accuracy_seen[t_train, s_step] = (
                    anytime_acc[t_train, s_step, : t_train + 1].mean().item()
                )
                anytime_accuracy_all[t_train, s_step] = (
                    anytime_acc[t_train, s_step, :].mean().item()
                )

        def _accuracy_seen(t: int) -> float:
            return accuracy_matrix[t, : t + 1].mean().item()

        def _accuracy_all(t: int) -> float:
            return accuracy_matrix[t, :].mean().item()

        tasks = np.arange(self.task_count, dtype=int)

        accuracy_seen = np.vectorize(_accuracy_seen)(tasks)
        accuracy_all = np.vectorize(_accuracy_all)(tasks)
        boundaries = boundary_instances.numpy()

        ttt_windowed_task_index = None
        if ttt.windowed is not None:
            ttt_windowed_task_index = _get_ttt_windowed_task_index(
                boundaries, ttt.windowed.window_size
            )
            assert len(ttt_windowed_task_index) == len(ttt.windowed.accuracy())

        return OCLMetrics(
            accuracy_seen=accuracy_seen,
            accuracy_all=accuracy_all,
            accuracy_final=_accuracy_all(self.task_count - 1),
            accuracy_all_avg=np.mean(accuracy_all),
            accuracy_seen_avg=np.mean(accuracy_seen),
            accuracy_matrix=accuracy_matrix.numpy(),
            anytime_accuracy_all=anytime_accuracy_all.flatten().numpy(),
            anytime_accuracy_seen=anytime_accuracy_seen.flatten().numpy(),
            anytime_accuracy_all_avg=anytime_accuracy_all.mean().item(),
            anytime_accuracy_seen_avg=anytime_accuracy_seen.mean().item(),
            anytime_task_index=np.linspace(
                0, self.task_count, self.step_count * self.task_count + 1
            )[1:],
            task_index=np.arange(self.task_count) + 1,
            anytime_accuracy_matrix=anytime_acc.flatten(end_dim=1).numpy(),
            backward_transfer=_backwards_transfer(accuracy_matrix),
            forward_transfer=_forwards_transfer(accuracy_matrix),
            ttt=ttt,
            boundaries=boundaries,
            ttt_windowed_task_index=ttt_windowed_task_index,
        )


_OCLClassifier = Union[TrainTaskAware, TestTaskAware, Classifier]


def _batch_test(learner: Classifier, x: Tensor) -> np.ndarray:
    """Test a batch of instances using the learner."""
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        return learner.batch_predict(x).cpu().detach().numpy(), learner.batch_predict_proba(x).cpu().detach().numpy()
    else:
        yb_pred = np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            instance = Instance.from_array(learner.schema, x[i].numpy())
            yb_pred[i] = learner.predict(instance)
        return yb_pred


def _batch_train(learner: Classifier, x: Tensor, y: Tensor, train_task_id: int):
    """Train a batch of instances using the learner."""
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    if isinstance(learner, ExperienceReplay) or isinstance(learner, ExperienceDelayReplay):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        y = y.to(dtype=learner.y_dtype, device=learner.device)
        
        learner.batch_train(x, y, train_task_id)
    elif isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        y = y.to(dtype=learner.y_dtype, device=learner.device)
        
        learner.batch_train(x, y)
    else:
        for i in range(batch_size):
            instance = LabeledInstance.from_array(
                learner.schema, x[i].numpy(), int(y[i].item())
            )
            learner.train(instance)

def _batch_train_random(learner: Classifier, batches: List[Tuple[Tensor, Tensor]], 
                        train_task_id: int):

    #join all batches
    xb_join = torch.cat([b[0] for b in batches], dim=0)
    yb_join = torch.cat([b[1] for b in batches], dim=0)

    #size of batch
    n = batches[0][0].shape[0]
    count = xb_join.shape[0]
    indices = torch.randint(0, count, (n,))
    
    xb_selected = xb_join[indices]
    yb_selected = yb_join[indices]
    print(f'Size random {len(batches)} batches')
    for batch in batches:
        print(f'Delay random {batch[3]}')

    # print(f'Train random ER {len(batches)} batches, {count} instances, selected {n} instances')
    _batch_train(learner, xb_selected, yb_selected, train_task_id )

def _batch_delayed_train(learner: Classifier, batches: List[Tuple[Tensor, Tensor]], 
                        delay: int, train_task_id: int):
    """Train a batch of instances using the learner."""
    learner.batch_train(batches, delay, train_task_id)

def _batch_mixed_delayed_train(learner: Classifier, batches: List[Tuple[Tensor, Tensor]], 
                                train_task_id: int):
    """Train a batch of instances using the learner."""
    print(f'Size mixed {len(batches)} batches')
    for batch in batches:
        print(f'Delay mixed {batch[3]}')
    learner.batch_mixed_train(batches, train_task_id)


def ocl_train_eval_loop(
    learner: _OCLClassifier,
    train_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    test_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    continual_evaluations: int = 1,
    progress_bar: bool = False,
    eval_window_size: int = 1000,
) -> OCLMetrics:
    """Train and evaluate a learner on a sequence of tasks.

    :param learner: A classifier that is possibly train task aware and/or
        test task aware.
    :param train_streams: A sequence of streams containing the training tasks.
    :param test_streams: A sequence of streams containing the testing tasks.
    :param continual_evaluations: The number of times to evaluate the learner
        during each task. If 1, the learner is only evaluated at the end of each task.
    :param progress_bar: Whether to display a progress bar. The bar displayed
        will show the progress over all training and evaluation steps including
        the continual evaluations.
    :return: A collection of metrics evaluating the learner's performance.
    """
    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise ValueError("Learner must be a classifier")
    if 1 > continual_evaluations:
        raise ValueError("Continual evaluations must be at least 1")
    if (min_stream_len := min(len(s) for s in train_streams)) < continual_evaluations:
        raise ValueError(
            "Cannot evaluate more times than the number of batches. "
            f"(min stream length (in batches): {min_stream_len}, "
            f"continual evaluations: {continual_evaluations})"
        )

    metrics = _OCLEvaluator(
        n_tasks, continual_evaluations, learner.schema.get_num_classes()
    )
    online_eval = ClassificationEvaluator(schema=learner.schema)
    windowed_eval = ClassificationWindowedEvaluator(
        schema=learner.schema, window_size=eval_window_size
    )
    boundary_instances = torch.zeros(len(train_streams) + 1)
    start_wallclock_time, start_cpu_time = start_time_measuring()

    # Setup progress bar
    train_len = sum(len(stream) for stream in train_streams)
    test_len = sum(len(stream) for stream in test_streams)
    pbar = tqdm(
        total=train_len + test_len * continual_evaluations * n_tasks,
        disable=not progress_bar,
        desc="Train & Eval",
    )

    # Iterate over each task
    for train_task_id, train_stream in enumerate(train_streams):
        # Setup stream and inform learner of the test task
        if isinstance(learner, TrainTaskAware):
            learner.on_train_task(train_task_id)

        # Train and evaluation loop for a single task
        for step, (xb, yb) in enumerate(train_stream):
            # Update the learner and collect prequential statistics
            xb: Tensor
            yb: Tensor
            pbar.update(1)
            yb_pred = _batch_test(learner, xb)
            _batch_train(learner, xb, yb)
            for y, y_pred in zip(yb, yb_pred, strict=True):
                online_eval.update(y.item(), y_pred)
                windowed_eval.update(y.item(), y_pred)

            # Evaluate the learner on evenly spaced steps during training
            evaluate_every = len(train_stream) // continual_evaluations
            if (step + 1) % evaluate_every == 0:
                eval_step = step // evaluate_every

                if eval_step >= continual_evaluations:
                    # This can occur when not dropping the last incomplete batch.
                    continue

                for test_task_id, test_stream in enumerate(test_streams):
                    # Setup stream and inform learner of the test task
                    if isinstance(learner, TestTaskAware):
                        learner.on_test_task(test_task_id)

                    # predict instances in the current task
                    for test_xb, test_yb in test_stream:
                        pbar.update(1)
                        yb_pred = _batch_test(learner, test_xb)

                        for y, y_pred in zip(test_yb, yb_pred):
                            metrics.holdout_update(
                                train_task_id,
                                eval_step,
                                test_task_id,
                                y.item(),
                                y_pred,
                            )

            boundary_instances[train_task_id + 1] = online_eval.instances_seen

    # TODO: We should measure time spent in ``learner.train`` separately from
    # time spent in evaluation.
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )
    return metrics.build(
        PrequentialResults(
            learner=str(learner),
            stream=f"{train_streams[0]}x{len(train_streams)}",
            cumulative_evaluator=online_eval,
            windowed_evaluator=windowed_eval,
            wallclock=elapsed_wallclock_time,
            cpu_time=elapsed_cpu_time,
        ),
        boundary_instances,
    )


def ocl_train_eval_delayed_loop(
    learner: _OCLClassifier,
    train_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    test_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    continual_evaluations: int = 1,
    progress_bar: bool = False,
    eval_window_size: int = 1000,
    delay_label: Optional[int] = None,
    delay_batches: Optional[bool] = None,
    select_tasks: Optional[List[int]] = None,
    no_delayed_tasks: Optional[List[int]] = None,
    start_delay_size: Optional[int] = None,
    number_delayed_batches: int = 1,
) -> OCLMetrics:
    """Train and evaluate a learner on a sequence of tasks.

    :param learner: A classifier that is possibly train task aware and/or
        test task aware.
    :param train_streams: A sequence of streams containing the training tasks.
    :param test_streams: A sequence of streams containing the testing tasks.
    :param continual_evaluations: The number of times to evaluate the learner
        during each task. If 1, the learner is only evaluated at the end of each task.
    :param progress_bar: Whether to display a progress bar. The bar displayed
        will show the progress over all training and evaluation steps including
        the continual evaluations.
    :param delay_label: Optional delay (in number of instances) to make the label instances available.
    :return: A collection of metrics evaluating the learner's performance.
    """
    if len(select_tasks) > 0:
        new_train_streams = list()
        new_test_streams = list()
        for train_task_id, train_stream in enumerate(train_streams):
            #filter on train_stream class from mnist
            if train_task_id in select_tasks:
                new_train_streams.append(train_stream)

        for test_task_id, test_stream in enumerate(test_streams):
            if test_task_id in select_tasks:
                new_test_streams.append(test_stream)

        train_streams = new_train_streams
        test_streams = new_test_streams

    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise ValueError("Learner must be a classifier")
    if 1 > continual_evaluations:
        raise ValueError("Continual evaluations must be at least 1")
    if (min_stream_len := min(len(s) for s in train_streams)) < continual_evaluations:
        raise ValueError(
            "Cannot evaluate more times than the number of batches. "
            f"(min stream length (in batches): {min_stream_len}, "
            f"continual evaluations: {continual_evaluations})"
        )

    metrics = _OCLEvaluator(
        n_tasks, continual_evaluations, learner.schema.get_num_classes()
    )
    online_eval = ClassificationEvaluator(schema=learner.schema)
    windowed_eval = ClassificationWindowedEvaluator(
        schema=learner.schema, window_size=eval_window_size
    )
    boundary_instances = torch.zeros(len(train_streams) + 1)
    start_wallclock_time, start_cpu_time = start_time_measuring()

    # Setup progress bar
    train_len = sum(len(stream) for stream in train_streams)
    test_len = sum(len(stream) for stream in test_streams)
    pbar = tqdm(
        total=train_len + test_len * continual_evaluations * n_tasks,
        disable=not progress_bar,
        desc="Train & Eval",
    )
    
    train_batches = list()
    # Iterate over each task
    #TODO: delay random batches
    for train_task_id, train_stream in enumerate(train_streams):
        # Setup stream and inform learner of the test task
        if isinstance(learner, TrainTaskAware):
            learner.on_train_task(train_task_id)
        
        batches_no_delay = 0
        # Train and evaluation loop for a single task
        for step, (xb, yb) in enumerate(train_stream):
            # Update the learner and collect prequential statistics
            xb: Tensor
            yb: Tensor
            pbar.update(1)
            yb_pred, yb_pred_proba = _batch_test(learner, xb)
                          
            if (len(no_delayed_tasks) > 0 and train_task_id in no_delayed_tasks) or (batches_no_delay < start_delay_size):
                # TODO: The case of learning is EDR
                if isinstance(learner, ExperienceReplay):
                    # print("ER learner")
                    _batch_train(learner, xb, yb, train_task_id)
                else:
                    # print("EDR learner")
                    _batch_delayed_train(learner, [(xb, yb, yb_pred_proba)], delay_label + number_delayed_batches, train_task_id)

                batches_no_delay += 1
                # clean de delayed instances when the task is stationary
                # train_batches = list()
            elif delay_label is not None:
                # Apply label delay
                if(len(train_batches) >= delay_label + number_delayed_batches):
                    # batch_instance = train_batches.pop(0)
                    
                    if isinstance(learner, ExperienceReplay):
                        # batches_instances = train_batches[:number_delayed_batches]
                        # del train_batches[:number_delayed_batches]
                        batches_instances = train_batches[:number_delayed_batches]
                        del train_batches[:number_delayed_batches]
                        # print("ER learner")
                        # print(f'Batch Delay {delay_label + number_delayed_batches}')
                        # for instance in batches_instances:
                        # _batch_train(learner, batches_instances[0][0], batches_instances[0][1], train_task_id)
                        _batch_train_random(learner, batches_instances, train_task_id)
                    else:
                        # print("EDR learner")
                        #TODO: train ER equals train ER 
                        #TODO: train random ER to select instances and EDR with importance sampling
                        batches_instances = train_batches[:number_delayed_batches]
                        del train_batches[:number_delayed_batches]
                        _batch_delayed_train(learner, batches_instances, delay_label + number_delayed_batches, train_task_id)
                        
                else:
                    train_batches.append((xb, yb, yb_pred_proba))

            for y, y_pred in zip(yb, yb_pred, strict=True):
                online_eval.update(y.item(), y_pred)
                windowed_eval.update(y.item(), y_pred)

            # Evaluate the learner on evenly spaced steps during training
            # 
            evaluate_every = len(train_stream) // continual_evaluations
            if (step + 1) % evaluate_every == 0:
                eval_step = step // evaluate_every

                if eval_step >= continual_evaluations:
                    # This can occur when not dropping the last incomplete batch.
                    continue

                for test_task_id, test_stream in enumerate(test_streams):
                    # Setup stream and inform learner of the test task
                    if isinstance(learner, TestTaskAware):
                        learner.on_test_task(test_task_id)

                    # predict instances in the current task
                    for test_xb, test_yb in test_stream:
                        pbar.update(1)
                        yb_pred = _batch_test(learner, test_xb)

                        for y, y_pred in zip(test_yb, yb_pred):
                            metrics.holdout_update(
                                train_task_id,
                                eval_step,
                                test_task_id,
                                y.item(),
                                y_pred,
                            )

            boundary_instances[train_task_id + 1] = online_eval.instances_seen

    # TODO: We should measure time spent in ``learner.train`` separately from
    # time spent in evaluation.
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )
    return metrics.build(
        PrequentialResults(
            learner=str(learner),
            stream=f"{train_streams[0]}x{len(train_streams)}",
            cumulative_evaluator=online_eval,
            windowed_evaluator=windowed_eval,
            wallclock=elapsed_wallclock_time,
            cpu_time=elapsed_cpu_time,
        ),
        boundary_instances,
    )


def ocl_train_eval_mixed_delayed_loop(
    learner: _OCLClassifier,
    train_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    test_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    continual_evaluations: int = 1,
    progress_bar: bool = False,
    eval_window_size: int = 1000,
    delayed_batches: Optional[int] = None,
    select_tasks: Optional[List[int]] = None,
    number_delayed_batches: int = 1,
    prob_no_delay_batches: float = 0.5,
    er_strategy: str = "ER"
) -> OCLMetrics:
    """Train and evaluate a learner on a sequence of tasks.

    :param learner: A classifier that is possibly train task aware and/or
        test task aware.
    :param train_streams: A sequence of streams containing the training tasks.
    :param test_streams: A sequence of streams containing the testing tasks.
    :param continual_evaluations: The number of times to evaluate the learner
        during each task. If 1, the learner is only evaluated at the end of each task.
    :param progress_bar: Whether to display a progress bar. The bar displayed
        will show the progress over all training and evaluation steps including
        the continual evaluations.
    :param delay_label: Optional delay (in number of instances) to make the label instances available.
    :return: A collection of metrics evaluating the learner's performance.
    """
    if len(select_tasks) > 0:
        new_train_streams = list()
        new_test_streams = list()
        for train_task_id, train_stream in enumerate(train_streams):
            #filter on train_stream class from mnist
            if train_task_id in select_tasks:
                new_train_streams.append(train_stream)

        for test_task_id, test_stream in enumerate(test_streams):
            if test_task_id in select_tasks:
                new_test_streams.append(test_stream)

        train_streams = new_train_streams
        test_streams = new_test_streams

    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise ValueError("Learner must be a classifier")
    if 1 > continual_evaluations:
        raise ValueError("Continual evaluations must be at least 1")
    if (min_stream_len := min(len(s) for s in train_streams)) < continual_evaluations:
        raise ValueError(
            "Cannot evaluate more times than the number of batches. "
            f"(min stream length (in batches): {min_stream_len}, "
            f"continual evaluations: {continual_evaluations})"
        )

    metrics = _OCLEvaluator(
        n_tasks, continual_evaluations, learner.schema.get_num_classes()
    )
    online_eval = ClassificationEvaluator(schema=learner.schema)
    windowed_eval = ClassificationWindowedEvaluator(
        schema=learner.schema, window_size=eval_window_size
    )
    boundary_instances = torch.zeros(len(train_streams) + 1)
    start_wallclock_time, start_cpu_time = start_time_measuring()

    # Setup progress bar
    train_len = sum(len(stream) for stream in train_streams)
    test_len = sum(len(stream) for stream in test_streams)
    pbar = tqdm(
        total=train_len + test_len * continual_evaluations * n_tasks,
        disable=not progress_bar,
        desc="Train & Eval",
    )
    
    train_batches_delayed = list()
    train_batches_no_delay = list()
    # Iterate over each task
    #TODO: delay random batches
    for train_task_id, train_stream in enumerate(train_streams):
        # Setup stream and inform learner of the test task
        if isinstance(learner, TrainTaskAware):
            learner.on_train_task(train_task_id)

        # Train and evaluation loop for a single task
        for step, (xb, yb) in enumerate(train_stream):
            # Update the learner and collect prequential statistics
            xb: Tensor
            yb: Tensor
            pbar.update(1)
            yb_pred, yb_pred_proba = _batch_test(learner, xb)

            if train_task_id != 0 and torch.rand(1).item() < prob_no_delay_batches:
                train_batches_no_delay.append((xb, yb, yb_pred_proba, 1))
            
            if delayed_batches is not None:
                # Apply label delay
                if(len(train_batches_delayed) >= delayed_batches + number_delayed_batches) or len(train_batches_no_delay) > 0:
                    if len(train_batches_no_delay) > 0:
                        batches_instances = train_batches_delayed[:number_delayed_batches]
                        del train_batches_delayed[:number_delayed_batches]
                        batches_instances.append(train_batches_no_delay.pop(0))   
                    else:
                        batches_instances = train_batches_delayed[:number_delayed_batches+1]
                        del train_batches_delayed[:number_delayed_batches+1]
                        
                    
                    for b in batches_instances:
                        _log_batches_train(learner, b[1], train_task_id, step)

                    if (isinstance(learner, ExperienceReplay) or
                        isinstance(learner, GDumb) or
                        isinstance(learner, NCM) or
                        isinstance(learner, SLDA)):
                        # batches_instances = train_batches[:number_delayed_batches]
                        # del train_batches[:number_delayed_batches]
                        
                        # print("ER learner")
                        # print(f'Batch Delay {delayed_batches + number_delayed_batches}')
                        # for instance in batches_instances:
                        # _batch_train(learner, batches_instances[0][0], batches_instances[0][1], train_task_id)
                        if er_strategy == "RER":
                            _batch_train_random(learner, batches_instances, train_task_id)
                        elif er_strategy == "ER_f":
                            # print("RER_f")
                            # sort batches_instances by delay
                            batches_instances = sorted(batches_instances, key=lambda x: x[3])
                            _batch_train(learner, batches_instances[0][0], batches_instances[0][1], train_task_id)
                        elif er_strategy == "ER_l":
                            # print("RER_l")
                            # sort batches_instances by delay
                            batches_instances = sorted(batches_instances, key=lambda x: x[3], reverse=True)
                            _batch_train(learner, batches_instances[0][0], batches_instances[0][1], train_task_id)
                        elif (
                            er_strategy == "ER_2B"
                            or er_strategy == "ER-ACE"
                            or er_strategy == "gdumb"
                            or er_strategy == "ncm"
                            or er_strategy == "slda"
                        ):
                            batches_instances = sorted(batches_instances, key=lambda x: x[3], reverse=True)
                            selected_batch = batches_instances[0]
                            
                            if len(batches_instances) > 1:
                                old_batch = batches_instances[1]
                                train_batches_delayed.insert(0, old_batch)
                                
                            _batch_train(learner, selected_batch[0], selected_batch[1], train_task_id)

                    else:
                        # print("EDR learner")
                        #TODO: train ER equals train ER 
                        #TODO: train random ER to select instances and EDR with importance sampling
                        _batch_mixed_delayed_train(learner, batches_instances, train_task_id)
                        
                else:
                    train_batches_delayed.append((xb, yb, yb_pred_proba, delayed_batches + number_delayed_batches))

            for y, y_pred in zip(yb, yb_pred, strict=True):
                online_eval.update(y.item(), y_pred)
                windowed_eval.update(y.item(), y_pred)

            # Evaluate the learner on evenly spaced steps during training
            evaluate_every = len(train_stream) // continual_evaluations
            if (step + 1) % evaluate_every == 0:
                eval_step = step // evaluate_every

                if eval_step >= continual_evaluations:
                    # This can occur when not dropping the last incomplete batch.
                    continue

                for test_task_id, test_stream in enumerate(test_streams):
                    # Setup stream and inform learner of the test task
                    if isinstance(learner, TestTaskAware):
                        learner.on_test_task(test_task_id)

                    # predict instances in the current task
                    for test_xb, test_yb in test_stream:
                        pbar.update(1)
                        yb_pred = _batch_test(learner, test_xb)

                        for y, y_pred in zip(test_yb, yb_pred):
                            metrics.holdout_update(
                                train_task_id,
                                eval_step,
                                test_task_id,
                                y.item(),
                                y_pred,
                            )

            boundary_instances[train_task_id + 1] = online_eval.instances_seen

    # TODO: We should measure time spent in ``learner.train`` separately from
    # time spent in evaluation.
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )
    return metrics.build(
        PrequentialResults(
            learner=str(learner),
            stream=f"{train_streams[0]}x{len(train_streams)}",
            cumulative_evaluator=online_eval,
            windowed_evaluator=windowed_eval,
            wallclock=elapsed_wallclock_time,
            cpu_time=elapsed_cpu_time,
        ),
        boundary_instances,
    )

def _log_batches_train(learner, train_y: Tensor, train_task_id: int, stream_id: int):
    # count number of each classes in train_y
    class_counts = train_y.bincount(minlength=learner.schema.get_num_classes())
    
    # log the class counts in a debug file
    # Transform class_counts to a more readable format
    class_counts_str = ",".join(f"{count}" for i, count in enumerate(class_counts))
    
    os.makedirs("debug", exist_ok=True)
    with open(f"debug/cl_batches_y_{learner.__class__.__name__}.log", "a") as f:
        f.write(f"{train_task_id},{stream_id},{class_counts_str}\n")   