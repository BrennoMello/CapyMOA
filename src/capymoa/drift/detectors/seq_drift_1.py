from typing import Optional

from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import SeqDrift1ChangeDetector as _SeqDrift1ChangeDetector


class SeqDrift1ChangeDetector(MOADriftDetector):
    """SeqDrift1 Detector

    Example usages:

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> from capymoa.drift.detectors import 
    >>>
    >>> detector = 
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 6 - at index: 1011
    Change detected in data: 7 - at index: 1556

    """

    def __init__(
        self,
        delta: float = 0.01,
        deltaWarning: float = 0.1,
        block: int = 200,
        CLI: Optional[str] = None,
    ):
        if CLI is None:
            CLI = f"-d {delta} -w {deltaWarning} -b {block}"

        super().__init__(moa_detector=_SeqDrift1ChangeDetector(), CLI=CLI)
       
        self.delta = delta
        self.deltaWarning = deltaWarning
        self.block = block
        self.get_params()
