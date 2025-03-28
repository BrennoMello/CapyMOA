from capymoa.evaluation.evaluation import (
    start_time_measuring, 
    ClassificationEvaluator, 
    ClassificationWindowedEvaluator,
    stop_time_measuring,
    _setup_progress_bar
)
from capymoa.evaluation.results import PrequentialResults
from capymoa.base import Classifier, ClassifierSSL
from typing import Union, Optional
from capymoa.stream import Stream

from tqdm import tqdm
import numpy as np
import time



def prequential_cd_dplds_evaluation(
    stream: Stream,
    learner: Union[ClassifierSSL, Classifier],
    max_instances: Optional[int] = None,
    window_size: int = 1000,
    initial_window_size: int = 0,
    delay_length: int = 0,
    label_probability: float = 0.01,
    random_seed: int = 1,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: Union[bool, tqdm] = False,
    cd_ground_truth: Optional[int] = None
):
    """Run and evaluate a learner on a partially or delay stream using prequential evaluation.

    :param stream: A data stream to evaluate the learner on. Will be restarted if
        ``restart_stream`` is True.
    :param learner: The learner to evaluate. If the learner is an SSL learner,
        it will be trained on both labeled and unlabeled instances. If the
        learner is not an SSL learner, then it will be trained only on the
        labeled instances.
    :param max_instances: The number of instances to evaluate before exiting.
        If None, the evaluation will continue until the stream is empty.
    :param window_size: The size of the window used for windowed evaluation,
        defaults to 1000
    :param initial_window_size: Not implemented yet
    :param delay_length: If greater than zero the labeled (``label_probability``%)
        instances will appear as unlabeled before reappearing as labeled after
        ``delay_length`` instances, defaults to 0
    :param label_probability: The proportion of instances that will be labeled,
        must be in the range [0, 1], defaults to 0.01
    :param random_seed: A random seed to define the random state that decides
        which instances are labeled and which are not, defaults to 1.
    :param store_predictions: Store the learner's prediction in a list, defaults
        to False
    :param store_y: Store the ground truth targets in a list, defaults to False
    :param optimise: If True and the learner is compatible, the evaluator will
        use a Java native evaluation loop, defaults to True.
    :param restart_stream: If False, evaluation will continue from the current
        position in the stream, defaults to True. Not restarting the stream is
        useful for switching between learners or evaluators, without starting
        from the beginning of the stream.
    :param progress_bar: Enable, disable, or override the progress bar. Currently
        incompatible with ``optimize=True``.
    :return: An object containing the results of the evaluation windowed metrics,
        cumulative metrics, ground truth targets, and predictions.
    """

    if restart_stream:
        stream.restart()

    # if _is_fast_mode_compilable(stream, learner, optimise):
    #     return _prequential_ssl_evaluation_fast(
    #         stream,
    #         learner,
    #         max_instances,
    #         window_size,
    #         initial_window_size,
    #         delay_length,
    #         label_probability,
    #         random_seed,
    #     )

    # IMPORTANT: delay_length and initial_window_size have not been implemented in python yet
    # In MOA it is implemented so _prequential_ssl_evaluation_fast works just fine.
    # if initial_window_size != 0:
    #     raise ValueError(
    #         "Initial window size must be 0 for this function as the feature is not implemented yet."
    #     )

    # if delay_length != 0:
    #     raise ValueError(
    #         "Delay length must be 0 for this function as the feature is not implemented yet."
    #     )

    # Reset the random state
    mt19937 = np.random.MT19937()
    mt19937._legacy_seeding(random_seed)
    rand = np.random.Generator(mt19937)

    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    evaluator_cumulative = None
    evaluator_windowed = None
    if stream.get_schema().is_classification():
        evaluator_cumulative = ClassificationEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
        # If the window_size is None, then should not initialise or produce prequential (window) results.
        if window_size is not None:
            evaluator_windowed = ClassificationWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
    else:
        raise ValueError("The learning task is not classification")

    unlabeled_counter = 0

    progress_bar = _setup_progress_bar(
        "CD Dplds Eval", progress_bar, stream, learner, max_instances
    )
    for i, instance in enumerate(stream):
        prediction = learner.predict(instance)

        if stream.get_schema().is_classification():
            y = instance.y_index
        else:
            y = instance.y_value

        evaluator_cumulative.update(instance.y_index, prediction)
        if evaluator_windowed is not None:
            evaluator_windowed.update(instance.y_index, prediction)

        if rand.random(dtype=np.float64) >= label_probability:
            # if 0.00 >= label_probability:
            # Do not label the instance
            if isinstance(learner, ClassifierSSL):
                learner.train_on_unlabeled(instance)
                # Otherwise, just ignore the unlabeled instance
            unlabeled_counter += 1
        else:
            # Labeled instance
            learner.train(instance)

        # Storing predictions if store_predictions was set to True during initialisation
        if predictions is not None:
            predictions.append(prediction)

        # Storing ground-truth if store_y was set to True during initialisation
        if ground_truth_y is not None:
            ground_truth_y.append(y)

        if progress_bar is not None:
            progress_bar.update(1)

        if max_instances is not None and i >= (max_instances - 1):
            break

    if progress_bar is not None:
        progress_bar.close()

    # # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    # Add the results corresponding to the remainder of the stream in case the number of processed instances is not
    # perfectly divisible by the window_size (if it was, then it is already in the result_windows variable).
    if (
        evaluator_windowed is not None
        and evaluator_windowed.get_instances_seen() % window_size != 0
    ):
        evaluator_windowed.result_windows.append(evaluator_windowed.metrics())

    results = PrequentialResults(
        learner=str(learner),
        stream=stream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=evaluator_cumulative,
        windowed_evaluator=evaluator_windowed,
        ground_truth_y=ground_truth_y,
        predictions=predictions,
        other_metrics={
            "unlabeled": unlabeled_counter,
            "unlabeled_ratio": unlabeled_counter / i,
        },
    )

    return results