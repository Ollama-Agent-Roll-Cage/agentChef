"""
UDRAGS - Unified Dataset Research, Augmentation, & Generation System
=====================================================================

A comprehensive suite of tools for researching, generating, expanding, and cleaning 
conversation datasets powered by local Ollama models.

Main Components:
---------------
- OllamaConversationGenerator: Generate conversations from text content
- DatasetExpander: Create variations of existing conversation datasets
- DatasetCleaner: Identify and fix quality issues in datasets
- OllamaPandasQuery: Natural language querying of pandas DataFrames
- ResearchManager: Main interface for the research workflow

All components use local Ollama models, with no external API dependencies.
"""

__version__ = '0.2.6'

# Import main components
try:
    from .conversation_generator import OllamaConversationGenerator
    from .dataset_expander import DatasetExpander
    from .dataset_cleaner import DatasetCleaner
    from .pandas_query import PandasQueryIntegration, OllamaLlamaIndexIntegration
    from .udrags import ResearchManager
    from .ollama_interface import OllamaInterface
except ImportError as e:
    import logging
    logging.warning(f"Error importing UDRAGS components: {e}")

# Check for required dependencies
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    import logging
    logging.warning("Ollama not installed. Most functionality will be limited. Install with 'pip install ollama'")

# Optional UI components
try:
    from .ui_module import UdragsUI, run_ui
    HAS_UI = True
except ImportError:
    HAS_UI = False

__all__ = [
    'OllamaConversationGenerator',
    'DatasetExpander',
    'DatasetCleaner',
    'PandasQueryIntegration',
    'OllamaLlamaIndexIntegration',
    'ResearchManager',
    'OllamaInterface'
]