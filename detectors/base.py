"""
base.py — interface contract for all fraud detectors.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """Minimal interface all detectors must satisfy."""
    name: str           # class-level display name
    train_time: float   # set by fit()

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseDetector":
        """Train the detector. Must set self.train_time. Returns self."""
        ...

    @abstractmethod
    def score(self, X_test=None) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous) for the test set."""
        ...
