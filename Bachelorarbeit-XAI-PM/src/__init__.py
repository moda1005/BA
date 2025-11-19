"""
XAI für Predictive Maintenance
Bachelorarbeit – Entwicklung erklärbarer KI-Verfahren für Predictive Maintenance
"""

__version__ = "1.0.0"
__author__ = "Mohamed Darguech"

from . import data_prep
from . import train_model
from . import evaluate
from . import shap_analysis
from . import utils

__all__ = [
    'data_prep',
    'train_model',
    'evaluate',
    'shap_analysis',
    'utils'
]
