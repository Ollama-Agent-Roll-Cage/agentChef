"""Core functionality for agentChef package."""

from .chefs.ragchef import ResearchManager
from .classification.dataset_cleaner import DatasetCleaner
from .crawlers.crawlers_module import WebCrawlerWrapper, ArxivSearcher, DuckDuckGoSearcher, GitHubCrawler
from .generation.conversation_generator import OllamaConversationGenerator
from .llamaindex.pandas_query import PandasQueryIntegration, OllamaLlamaIndexIntegration
from .ollama.ollama_interface import OllamaInterface
from .ui_components.RagchefUI.ui_module import RagchefUI

__all__ = [
    'ResearchManager',
    'DatasetCleaner',
    'WebCrawlerWrapper', 
    'ArxivSearcher',
    'DuckDuckGoSearcher',
    'GitHubCrawler',
    'OllamaConversationGenerator',
    'PandasQueryIntegration',
    'OllamaLlamaIndexIntegration',
    'OllamaInterface',
    'RagchefUI'
]
