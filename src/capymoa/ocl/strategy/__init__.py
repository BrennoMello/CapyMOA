from ._experience_replay import (ExperienceReplay, ExperienceDelayReplay, 
                                 ExperienceReplayACE, ExperienceReplayAsymmetricCrossEntropy,
                                 ACELoss)
from ._slda import SLDA
from ._ncm import NCM
from ._gdumb import GDumb
from ._rar import RAR
from . import l2p
from ._ewc import EWC

__all__ = [
    "ExperienceReplay",
    "ExperienceDelayReplay",
    "ExperienceReplayACE",
    "ExperienceReplayAsymmetricCrossEntropy",
    "ACELoss",
    "SLDA",
    "NCM",
    "GDumb",
    "RAR",
    "l2p", 
    "EWC"
]
