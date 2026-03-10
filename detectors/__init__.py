"""
detectors — fraud detection algorithm package.

Re-exports all public names to preserve the existing import interface:
    from detectors import XGBoostDetector, GNNDetector, ...
"""

from detectors.base import BaseDetector
from detectors.tree_based import XGBoostDetector, LightGBMDetector, CatBoostDetector, AutoGluonDetector
from detectors.neural import AutoencoderDetector, TabNetDetector
from detectors.graph import (
    GATDetector,
    GATXGBDetector,
    GNNDetector,
    GraphSAGEXGBDetector,
    build_cc_graph,
    build_cc_graph_cosine,
    build_ieee_graph,
    build_ieee_graph_edges,
    normalize_adj,
)
from detectors.ensemble import IsolationForestDetector, StackingDetector, SuperEnsembleDetector

__all__ = [
    "BaseDetector",
    "XGBoostDetector",
    "LightGBMDetector",
    "CatBoostDetector",
    "AutoGluonDetector",
    "AutoencoderDetector",
    "TabNetDetector",
    "GATDetector",
    "GATXGBDetector",
    "GNNDetector",
    "GraphSAGEXGBDetector",
    "build_cc_graph",
    "build_cc_graph_cosine",
    "build_ieee_graph",
    "build_ieee_graph_edges",
    "normalize_adj",
    "IsolationForestDetector",
    "StackingDetector",
    "SuperEnsembleDetector",
]
