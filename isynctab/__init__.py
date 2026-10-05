"""iSyncTab package.

Exports the original image-tabular iSyncTab model and the general
audio-video extension.
"""

from .iSyncTab import iSyncTab
from .iSyncTab_AV import iSyncTab_AV, iSyncTabAV
from .estimator import iSyncTabClassifier

__all__ = [
    "iSyncTab",
    "iSyncTab_AV",
    "iSyncTabAV",
    "iSyncTabClassifier",
]
