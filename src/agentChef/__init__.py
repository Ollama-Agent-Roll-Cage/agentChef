"""
ragchef - Unified Dataset Research, Augmentation, & Generation Chef
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

__version__ = '0.2.8'

# Import main components
try:
    import logging
    from .core.chefs.ragchef import ResearchManager
    from .core.chefs.base_chef import BaseChef
    from .core.ui_components.RagchefUI.ui_module import RagchefUI
    from .core.ui_components.menu_module import AgentChefMenu
    from .core.augmentation.dataset_expander import DatasetExpander
    from .core.classification.dataset_cleaner import DatasetCleaner
    from .core.classification.classification import Classifier
    from .core.crawlers.crawlers_module import (
        WebCrawlerWrapper,
        ArxivSearcher,
        DuckDuckGoSearcher,
        GitHubCrawler
    )
    from .core.generation.conversation_generator import OllamaConversationGenerator
    from .core.llamaindex.pandas_query import PandasQueryIntegration, OllamaLlamaIndexIntegration
    from .core.ollama.ollama_interface import OllamaInterface
    from .pandas_rag import PandasRAG
except ImportError as e:
    import logging
    logging.warning(f"Error importing ragchef components: {e}")

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
    from .core.ui_components.RagchefUI.ui_module import RagchefUI
    HAS_UI = True
except ImportError:
    HAS_UI = False

# Update resource paths
import os
from pathlib import Path

# Define package paths
PACKAGE_DIR = Path(__file__).parent
MENU_HTML_PATH = PACKAGE_DIR / "core" / "ui_components" / "menu" / "agentChefMenu.html"
DOCS_DIR = PACKAGE_DIR / "docs"

# Create docs directory if needed
DOCS_DIR.mkdir(exist_ok=True)

# Create default menu HTML if it doesn't exist
if not MENU_HTML_PATH.exists():
    import shutil
    default_html = PACKAGE_DIR / "agentChefMenu.html"
    if default_html.exists():
        shutil.copy2(default_html, MENU_HTML_PATH)
    else:
        logging.error(f"Default menu HTML not found at {default_html}")

__all__ = [
    'PandasRAG',
    'OllamaConversationGenerator',
    'DatasetExpander',
    'DatasetCleaner',
    'PandasQueryIntegration',
    'OllamaLlamaIndexIntegration',
    'ResearchManager',
    'OllamaInterface',
    'WebCrawlerWrapper', 
    'ArxivSearcher', 
    'DuckDuckGoSearcher', 
    'GitHubCrawler',
    'RagchefUI',
    'BaseChef',
    'AgentChefMenu',
    'Classifier'
]