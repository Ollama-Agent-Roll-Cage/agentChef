"""Web crawling and data collection utilities."""

from .crawlers_module import (
    WebCrawlerWrapper,
    ArxivSearcher,
    DuckDuckGoSearcher,
    GitHubCrawler
)

__all__ = [
    'WebCrawlerWrapper',
    'ArxivSearcher', 
    'DuckDuckGoSearcher',
    'GitHubCrawler'
]
