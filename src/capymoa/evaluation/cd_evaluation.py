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
from collections import deque

from tqdm import tqdm
import numpy as np

class ConceptDriftDetectorEvaluator():

    def __init__(self):
        self.reset()

    def reset(self):
        self.weight_observed = 0.0
        self.number_detections = 0.0
        self.number_detections_occurred = 0.0
        self.has_false_alarm = False
        self.number_changes = 0.0
        self.number_warnings = 0.0
        self.delay = 0.0
        self.total_delay = 0.0
        self.total_delay_false_alarm = 0.0
        self.delay_false_alarm = 0.0
        self.is_warning_zone = False
        self.input_values = 0.0
        self.has_change_occurred = False

    def add_result(self, example, ground_truth, class_votes):
        example = example.java_instance
        inst = example.getData()  # assuming method like Java's getData()
        self.input_values = int(inst.classValue())
        inst_weight = inst.weight()
        # classVotes[0] -> is Change
        # classVotes[1] -> is in Warning Zone
        # classVotes[2] -> delay
        # classVotes[3] -> estimation
        print(f"Ground Truth {ground_truth}")
        print(f"Drift Detector {class_votes[0]}")
        print(f"Delay Detector {self.total_delay}")
        if inst_weight > 0.0 and class_votes.length == 4:
            self.delay += 1
            self.weight_observed += inst.weight()

            if class_votes[0] == 1.0:
                self.number_detections += inst.weight()
                if self.has_change_occurred:
                    self.total_delay += self.delay - class_votes[2]
                    self.number_detections_occurred += inst.weight()
                    self.has_change_occurred = False
                    self.has_false_alarm = False
                    self.total_delay_false_alarm += self.delay_false_alarm
                    self.delay_false_alarm = 0.0
                else:
                    self.has_false_alarm = True
                    self.total_delay_false_alarm += self.delay_false_alarm
                    self.delay_false_alarm = 0.0

            if self.has_false_alarm:
                self.delay_false_alarm += 1

            if self.has_change_occurred and class_votes[1] == 1.0:
                if not self.is_warning_zone:
                    self.number_warnings += inst.weight()
                    self.is_warning_zone = True
            else:
                self.is_warning_zone = False

            if ground_truth == 1:
                self.number_changes += inst.weight()
                self.delay = 0
                self.has_change_occurred = True


    def get_performance_measurements(self):
        def safe_divide(a, b):
            return a / b if b != 0 else 0.0

        return [
            ("drift test instances", self.get_total_weight_observed()),
            ("detected changes", self.get_number_detections()),
            ("detected warnings", self.get_number_warnings()),
            ("true changes", self.get_number_changes()),
            ("delay detection (average)", safe_divide(self.get_total_delay(), self.get_number_changes())),
            ("delay true detection (average)", safe_divide(self.get_total_delay(), self.get_number_detections())),
            ("MTFA (average)", safe_divide(self.get_total_delay_false_alarm(),
                                           self.get_number_detections() - self.get_number_changes_occurred())),
            ("MDR", safe_divide(self.get_number_changes() - self.get_number_changes_occurred(),
                                self.get_number_changes())),
            ("true changes detected", self.get_number_changes_occurred()),
            ("input values", self.get_input_values())
        ]

    # Getters
    def get_total_weight_observed(self):
        return self.weight_observed

    def get_number_detections(self):
        return self.number_detections

    def get_number_warnings(self):
        return self.number_warnings

    def get_number_changes(self):
        return self.number_changes

    def get_total_delay(self):
        return self.total_delay

    def get_total_delay_false_alarm(self):
        return self.total_delay_false_alarm

    def get_number_changes_occurred(self):
        return self.number_detections_occurred

    def get_input_values(self):
        return self.input_values


class DelayBuffer():

    def __init__(self):
        self.delay_buffer = deque()
         
    def add(self, instace, timestamp, delay, label_available):
        self.delay_buffer.append((instace, timestamp+delay, label_available))
        #TODO: Use heapsort to sort the buffer
        self.delay_buffer = deque(sorted(self.delay_buffer, key=lambda x: x[1]))

    def sample(self, timestamp):
        #(instance, timestamp, label_available)
        #[(x_1, 3, True), (x_2, 5, True), (x_3, 11, False)]
        instance_tuples = list()
        for instance in list(self.delay_buffer):
            if instance[1] <= timestamp:
                instance_tuples.append(instance)
                self.delay_buffer.popleft()  

        return instance_tuples
        

def prequential_cd_partially_evaluation(
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
    cd_ground_truth_list: Optional[int] = None,
    cd_evaluator: Optional[ConceptDriftDetectorEvaluator] = None,
):
    """Run and evaluate a concept drift detector on a partially or delay stream using prequential evaluation.

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
        "CD Eval", progress_bar, stream, learner, max_instances
    )
    for i, instance in enumerate(stream):
        print(f"Instances Processed {i}")

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

        # TODO: Is it correct?
        cd_ground_truth = 0
        if i in cd_ground_truth_list:
            cd_ground_truth = 1
       
        drift_votes = learner.get_cd_votes_for_instance(instance)
        cd_evaluator.add_result(instance, cd_ground_truth, drift_votes)

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

    # Stop measuring time
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

    drift_measurements = cd_evaluator.get_performance_measurements()
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
        other_metrics=drift_measurements 
        # {
        #     "unlabeled": unlabeled_counter,
        #     "unlabeled_ratio": unlabeled_counter / i,
        # },
    )

    return results

def prequential_cd_delay_partially_evaluation(
    stream: Stream,
    learner: Union[ClassifierSSL, Classifier],
    max_instances: Optional[int] = None,
    window_size: int = 1000,
    initial_window_size: int = 0,
    delay_probability: int = 0.01,
    label_probability: float = 0.01,
    min_delay: int = 0,
    max_delay: int = 0,
    random_seed: int = 1,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: Union[bool, tqdm] = False,
    cd_ground_truth_list: Optional[int] = None,
    cd_evaluator: Optional[ConceptDriftDetectorEvaluator] = None,
):
    """Run and evaluate a concept drift detector on a partially or delay stream using prequential evaluation.

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

    progress_bar = _setup_progress_bar(
        "CD Eval", progress_bar, stream, learner, max_instances
    )
    delay_buffer = DelayBuffer()
    delay_counter = 0
    partially_counter = 0
    labeled_counter = 0
    for i, instance in enumerate(stream):
        print(f"Instances Processed {i}")

        # prediction = learner.predict(instance)

        # if stream.get_schema().is_classification():
        #     y = instance.y_index
        # else:
        #     y = instance.y_value

        # TODO: Update the evaluator with the delayed or unlabeled instance
        # evaluator_cumulative.update(instance.y_index, prediction)
        # if evaluator_windowed is not None:
        #     evaluator_windowed.update(instance.y_index, prediction)

        if i <= initial_window_size:
           learner.train(instance)
        elif rand.random(dtype=np.float64) >= delay_probability:
            #TODO: add delay on the instance
            delay = rand.uniform(min_delay, max_delay)
            delay_buffer.add(instance, i, round(delay), True)
            delay_counter += 1
        elif rand.random(dtype=np.float64) >= label_probability:
            #Instance unlabeled
            #TODO: case when the instance is delayed and unlabeled
            delay_buffer.add(instance, i, 0, False)   
            partially_counter += 1   
        else:
            # Labeled instance
            delay_buffer.add(instance, i, 0, True)
            labeled_counter += 1

        if i > initial_window_size:
            instances_current = delay_buffer.sample(i)
            for instance, delay, label_available in instances_current:
                prediction = learner.predict(instance)

                if stream.get_schema().is_classification():
                    y = instance.y_index
                else:
                    y = instance.y_value

                evaluator_cumulative.update(instance.y_index, prediction)
                if evaluator_windowed is not None:
                    evaluator_windowed.update(instance.y_index, prediction)

                #TODO: When the instance is unlabeled?    
                if isinstance(learner, ClassifierSSL):
                    learner.train_on_unlabeled(instance)
                elif label_available == True:    
                    learner.train(instance, delay) 
                
        # TODO: Is it correct?
        cd_ground_truth = 0
        if i in cd_ground_truth_list:
            cd_ground_truth = 1
       
        drift_votes = learner.get_cd_votes(instance)
        cd_evaluator.add_result(instance, cd_ground_truth, drift_votes)

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

    # Stop measuring time
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

    drift_measurements = cd_evaluator.get_performance_measurements()
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
        other_metrics=drift_measurements 
        # {
        #     "delayled": delay_counter,
        #     "delayled_ratio": delay_counter / i,
        # },
    )

    return results