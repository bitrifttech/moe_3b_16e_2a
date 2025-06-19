"""
Knowledge Datasets

Loaders for factual and encyclopedic knowledge datasets.
"""

from .wikipedia import WikipediaLoader, create_wikipedia_loader
from .the_pile import ThePileLoader, create_the_pile_loader

__all__ = [
    "WikipediaLoader",
    "ThePileLoader",
    "create_wikipedia_loader",
    "create_the_pile_loader"
]
