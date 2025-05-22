"""Core functionality for agentChef package."""

from .chefs.ragchef import ResearchManager
from .chefs.base_chef import BaseChef
from .ui_components.RagchefUI.ui_module import RagchefUI
from .ui_components.menu_module import AgentChefMenu
from .llamaindex import HAS_QUERY_ENGINE
from .crawlers import WebCrawlerWrapper, ArxivSearcher, DuckDuckGoSearcher, GitHubCrawler
from .ollama import OllamaInterface

__all__ = [
    'ResearchManager',
    'BaseChef', 
    'RagchefUI',
    'AgentChefMenu',
    'HAS_LLAMA_INDEX',
    'WebCrawlerWrapper',  # Changed from WebCrawler to WebCrawlerWrapper
    'ArxivSearcher',
    'DuckDuckGoSearcher', 
    'GitHubCrawler',
    'OllamaInterface'
]
