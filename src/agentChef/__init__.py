"""
AgentChef - AI Agent Development Framework
=========================================

A comprehensive toolkit for building, training, and deploying AI agents with natural language capabilities.

Core Components:
- PandasRAG: Script-friendly interface for data analysis with conversation history
- ResearchManager: Complete research pipeline (RAGChef example)
- BaseChef: Framework for building custom agent pipelines
- Storage & Generation tools: For building sophisticated agent workflows
"""

__version__ = '0.2.9'

# Core Chef Framework
from .core.chefs.base_chef import BaseChef
from .core.chefs.pandas_rag import PandasRAG
from .core.chefs.ragchef import ResearchManager

# Data Processing & Generation
from .core.generation.conversation_generator import OllamaConversationGenerator
from .core.augmentation.dataset_expander import DatasetExpander
from .core.classification.dataset_cleaner import DatasetCleaner
from .core.classification.classification import Classifier

# Interfaces & Storage
from .core.ollama.ollama_interface import OllamaInterface
from .core.llamaindex.pandas_query import PandasQueryIntegration

# Crawlers & Research Tools
from .core.crawlers.crawlers_module import (
    WebCrawlerWrapper,
    ArxivSearcher,
    DuckDuckGoSearcher,
    GitHubCrawler,
    ParquetStorageWrapper
)

# Agent Management (Optional - graceful fallback)
try:
    from .core.storage.conversation_storage import ConversationStorage, KnowledgeEntry
    from .core.prompts.agent_prompt_manager import AgentPromptManager
    HAS_AGENT_STORAGE = True
except ImportError:
    HAS_AGENT_STORAGE = False

# UI Components (Optional)
try:
    from .core.ui_components.RagchefUI.ui_module import RagchefUI
    from .core.ui_components.menu_module import AgentChefMenu
    HAS_UI = True
except ImportError:
    HAS_UI = False

# Check for Ollama availability
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    import logging
    logging.warning("Ollama not installed. Install with: pip install ollama")

# Main exports for users
__all__ = [
    # Core Framework
    'BaseChef',
    'PandasRAG', 
    'ResearchManager',
    
    # Data Processing
    'OllamaConversationGenerator',
    'DatasetExpander',
    'DatasetCleaner',
    'Classifier',
    
    # Interfaces
    'OllamaInterface',
    'PandasQueryIntegration',
    
    # Research Tools
    'WebCrawlerWrapper',
    'ArxivSearcher', 
    'DuckDuckGoSearcher',
    'GitHubCrawler',
    'ParquetStorageWrapper',
    
    # Optional exports
    'ConversationStorage',
    'KnowledgeEntry', 
    'AgentPromptManager',
    'RagchefUI',
    'AgentChefMenu'
]

# Clean up optional exports based on availability
if not HAS_AGENT_STORAGE:
    __all__ = [x for x in __all__ if x not in ['ConversationStorage', 'KnowledgeEntry', 'AgentPromptManager']]

if not HAS_UI:
    __all__ = [x for x in __all__ if x not in ['RagchefUI', 'AgentChefMenu']]