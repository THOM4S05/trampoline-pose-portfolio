"""Offline trampoline tracking. Importing the package does not load models."""
from .types import AthleteSelection, TrackerConfig

__version__ = "0.1.0"
__all__ = ["AthleteSelection", "TrackerConfig", "PoseTracker"]

def __getattr__(name):
    if name == "PoseTracker":
        from .api import PoseTracker
        return PoseTracker
    raise AttributeError(name)
