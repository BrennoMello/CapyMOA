from __future__ import annotations

import inspect

from capymoa.learner import MOAClassifier
import moa.classifiers.trees as moa_trees
from capymoa.stream import Schema


class HoeffdingTree(MOAClassifier):
    """Hoeffding Tree classifier.

    Parameters
    ----------
    schema
        The schema of the stream
    random_seed
        The random seed passed to the moa learner
    grace_period
        Number of instances a leaf should observe between split attempts.
    split_criterion
        Split criterion to use. Defaults to `InfoGainSplitCriterion`
    confidence
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 0 - Majority Class</br>
        - 1 - Naive Bayes</br>
        - 2 - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    numeric_attribute_observer
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits.
    binary_split
        If True, only allow binary splits.
    max_byte_size
        The max size of the tree, in bytes.
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    disable_prepruning
        If True, disable merit-based tree pre-pruning.
    """

    MAJORITY_CLASS = 0
    NAIVE_BAYES = 1
    NAIVE_BAYES_ADAPTIVE = 2

    def __init__(
            self,
            schema: Schema | None = None,
            random_seed: int = 0,
            grace_period: int = 200,
            split_criterion: str = "InfoGainSplitCriterion",
            confidence: float = 1e-3,
            tie_threshold: float = 0.05,
            leaf_prediction: int = MAJORITY_CLASS,
            nb_threshold: int = 0,
            numeric_attribute_observer: str = "GaussianNumericAttributeClassObserver",
            binary_split: bool = False,
            max_byte_size: float = 33554433,
            memory_estimate_period: int = 1000000,
            stop_mem_management: bool = True,
            remove_poor_attrs: bool = False,
            disable_prepruning: bool = True,
    ):
        mappings = {
            "grace_period": "-g",
            "max_byte_size": "-m",
            "numeric_attribute_observer": "-n",
            "memory_estimate_period": "-e",
            "split_criterion": "-s",
            "confidence": "-c",
            "tie_threshold": "-t",
            "binary_split": "-b",
            "stop_mem_management": "-z",
            "remove_poor_attrs": "-r",
            "disable_prepruning": "-p",
            "leaf_prediction": "-l",
            "nb_threshold": "-q"
        }

        config_str = ""
        parameters = inspect.signature(self.__init__).parameters
        for key in mappings:
            if key not in parameters:
                continue
            this_parameter = parameters[key]
            set_value = locals()[key]
            is_bool = type(set_value) == bool
            if is_bool:
                if set_value:
                    str_extension = mappings[key] + " "
                else:
                    str_extension = ""
            else:
                str_extension = f"{mappings[key]} {set_value} "
            config_str += str_extension

        super(HoeffdingTree, self).__init__(moa_learner=moa_trees.HoeffdingTree,
                                            schema=schema,
                                            CLI=config_str,
                                            random_seed=random_seed)
