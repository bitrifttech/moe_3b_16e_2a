"""
Modular Dataset System for LLM Training

This package provides a flexible, reusable system for loading and processing
various datasets for large language model training.

Key Features:
- Modular dataset loaders
- Configurable preprocessing
- Smart dataset mixing
- Context length management
- Quality validation
"""

# Use lazy imports to prevent circular imports
import importlib
from typing import Any, Type, Callable, Dict, TypeVar, Optional

# Type variable for dataset loaders
T = TypeVar('T')

def lazy_import(module_path: str, class_name: str) -> Type[T]:
    """Lazily import a class to prevent circular imports."""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# Base classes
class LazyLoader:
    def __init__(self, module_path: str, class_name: str):
        self.module_path = module_path
        self.class_name = class_name
        self._class: Optional[Type] = None
    
    def __call__(self, *args, **kwargs) -> Any:
        if self._class is None:
            self._class = lazy_import(self.module_path, self.class_name)
        return self._class(*args, **kwargs)
    
    def __instancecheck__(self, instance):
        if self._class is None:
            self._class = lazy_import(self.module_path, self.class_name)
        return isinstance(instance, self._class)

# Lazy load base classes
BaseDatasetLoader = LazyLoader('.base.dataset_loader', 'BaseDatasetLoader')
DatasetConfig = LazyLoader('.base.dataset_loader', 'DatasetConfig')

# Lazy load utilities
create_conversational_mix = LazyLoader('data_modules.utils.mixing', 'create_conversational_mix')
DatasetMixer = LazyLoader('data_modules.utils.mixing', 'DatasetMixer')
SmartSampler = LazyLoader('data_modules.utils.sampling', 'SmartSampler')

# Lazy load dataset loaders
def create_openassistant_loader(*args, **kwargs):
    from .conversational.openassistant import create_openassistant_loader as _create
    return _create(*args, **kwargs)

def create_anthropic_hh_loader(*args, **kwargs):
    from .conversational.anthropic_hh import create_anthropic_hh_loader as _create
    return _create(*args, **kwargs)

def create_sharegpt_loader(*args, **kwargs):
    from .conversational.sharegpt import create_sharegpt_loader as _create
    return _create(*args, **kwargs)

def create_alpaca_loader(*args, **kwargs):
    from .conversational.alpaca import create_alpaca_loader as _create
    return _create(*args, **kwargs)

def create_ultrachat_loader(*args, **kwargs):
    from .conversational.ultrachat import create_ultrachat_loader as _create
    return _create(*args, **kwargs)

def create_openorca_loader(*args, **kwargs):
    from .conversational.openorca import create_openorca_loader as _create
    return _create(*args, **kwargs)

def create_lmsys_chat_loader(*args, **kwargs):
    from .conversational.lmsys_chat import create_lmsys_chat_loader as _create
    return _create(*args, **kwargs)

def create_chatbot_arena_loader(*args, **kwargs):
    from .conversational.chatbot_arena import create_chatbot_arena_loader as _create
    return _create(*args, **kwargs)

def create_wikipedia_loader(*args, **kwargs):
    from .knowledge.wikipedia import create_wikipedia_loader as _create
    return _create(*args, **kwargs)

def create_the_pile_loader(*args, **kwargs):
    from .knowledge.the_pile import create_the_pile_loader as _create
    return _create(*args, **kwargs)

__version__ = "1.0.0"
__all__ = [
    # Base classes
    'BaseDatasetLoader',
    'DatasetConfig',
    
    # Utilities
    'create_conversational_mix',
    'DatasetMixer',
    'SmartSampler',
    
    # Dataset loaders
    'create_openassistant_loader',
    'create_anthropic_hh_loader',
    'create_sharegpt_loader',
    'create_alpaca_loader',
    'create_ultrachat_loader',
    'create_openorca_loader',
    'create_lmsys_chat_loader',
    'create_chatbot_arena_loader',
    'create_wikipedia_loader',
    'create_the_pile_loader',
]
