"""Oskar Cephalo - Automated Cephalometric Landmark Detection Package."""

from oskar_cephalo.inference import OskarCephaloDetector
from oskar_cephalo.postprocessor import LANDMARKS_29, CVM_STAGES_INFO

__version__ = "2.0.0"
__all__ = ["OskarCephaloDetector", "LANDMARKS_29", "CVM_STAGES_INFO"]
