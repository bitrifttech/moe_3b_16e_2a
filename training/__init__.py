"""
Modular training package for modular transformer training.
"""

# Make core modules available at package level
from .core import config, logging_setup, model_setup, trainer_setup, training_loop
from .callbacks import progress, checkpoint, optimization
from .data import openassistant, knowledge_datasets, conversational
from .utils import dataset_utils

__version__ = "1.0.0"
