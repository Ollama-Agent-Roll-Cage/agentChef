"""Chef modules for managing research and dataset pipelines."""

from .base_chef import BaseChef
from .ragchef import ResearchManager

__all__ = ['BaseChef', 'ResearchManager']
