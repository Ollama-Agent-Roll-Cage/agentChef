"""crawlers_module.py
This module provides wrappers around the oarc-crawlers package for:
- WebCrawler: General web page crawling
- ArxivSearcher: ArXiv paper lookup and parsing
- DuckDuckGoSearcher: DuckDuckGo search API integration
- GitHubCrawler: GitHub repository cloning and extraction

This version replaces the previous custom implementation with calls to the 
oarc-crawlers package which provides more comprehensive functionality.

Written By: @Borcherdingl
Date: 4/13/2025
"""

import os
import re
import logging
import json
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, UTC
import asyncio

# Import oarc-crawlers components
from oarc_crawlers import (
    WebCrawler,
    ArxivCrawler, 
    DDGCrawler,
    GHCrawler,
    ParquetStorage
)

# Configuration
DATA_DIR = os.getenv('DATA_DIR', 'data')

# Initialize logging
logger = logging.getLogger(__name__)

class WebCrawlerWrapper:
    """Class for crawling web pages and extracting content.
    
    This is a wrapper around the oarc-crawlers WebCrawler class.
    """
    
    def __init__(self):
        """Initialize the web crawler with the data directory."""
        try:
            self.crawler = WebCrawler(data_dir=DATA_DIR)
            logger.info("WebCrawlerWrapper initialized successfully")
        except ImportError:
            logger.error("oarc_crawlers.WebCrawler not available")
            self.crawler = None
        except Exception as e:
            logger.error(f"Failed to initialize WebCrawler: {e}")
            self.crawler = None
        self.rate_limit_delay = 3
        
    async def fetch_url_content(self, url):
        """Fetch content from a URL."""
        if not self.crawler:
            return None
        return await self.crawler.fetch_url_content(url)

    async def extract_text_from_html(self, html):
        """Extract main text content from HTML using BeautifulSoup."""
        if not html:
            return "Failed to extract text from the webpage."
        if not self.crawler:
            return "WebCrawler not available"
        return await self.crawler.extract_text_from_html(html)

    async def extract_pypi_content(self, html, package_name):
        """Specifically extract PyPI package documentation from HTML."""
        if not self.crawler:
            return None
        return await self.crawler.extract_pypi_content(html, package_name)
    
    async def format_pypi_info(self, package_data):
        """Format PyPI package data into a readable markdown format."""
        if not self.crawler:
            return "WebCrawler not available"
        return await self.crawler.format_pypi_info(package_data)


class ArxivSearcher:
    """Class for searching and retrieving ArXiv papers."""
    
    def __init__(self):
        try:
            self.fetcher = ArxivCrawler()
            logger.info("ArxivSearcher initialized successfully")
        except ImportError:
            logger.error("oarc_crawlers.ArxivCrawler not available")
            self.fetcher = None
        except Exception as e:
            logger.error(f"Failed to initialize ArxivCrawler: {e}")
            self.fetcher = None
        self.rate_limit_delay = 3
    
    async def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ArXiv for papers using OARC-Crawlers API."""
        if not self.fetcher:
            logger.error("ArxivSearcher not initialized with fetcher")
            return []
            
        try:
            logger.info(f"Searching ArXiv using OARC for: '{query}' (max_results: {max_results})")
            
            # Use OARC ArXiv search method
            search_results = await self.fetcher.search(query, limit=max_results)
            
            if 'error' in search_results:
                logger.error(f"OARC ArXiv search error: {search_results['error']}")
                return []
            
            papers = []
            results = search_results.get('results', [])
            
            for paper_data in results:
                formatted = {
                    'title': paper_data.get('title', ''),
                    'authors': paper_data.get('authors', []),
                    'abstract': paper_data.get('abstract', ''),
                    'categories': paper_data.get('categories', []),
                    'arxiv_url': paper_data.get('arxiv_url', ''),
                    'pdf_link': paper_data.get('pdf_link', ''),
                    'published': paper_data.get('published', ''),
                    'updated': paper_data.get('updated', ''),
                    'arxiv_id': paper_data.get('id', '')
                }
                
                if formatted['title']:  # Only add if we have a title
                    papers.append(formatted)
            
            logger.info(f"OARC ArXiv search successful: {len(papers)} papers found")
            return papers
            
        except Exception as e:
            logger.error(f"Error in OARC ArXiv search: {e}")
            return []

    def _guess_arxiv_category(self, query: str) -> Optional[str]:
        """Guess ArXiv category from search query."""
        query_lower = query.lower()
        
        category_mapping = {
            'neural network': 'cs.LG',
            'machine learning': 'cs.LG', 
            'deep learning': 'cs.LG',
            'artificial intelligence': 'cs.AI',
            'computer vision': 'cs.CV',
            'natural language': 'cs.CL',
            'nlp': 'cs.CL',
            'robotics': 'cs.RO',
            'quantum': 'quant-ph',
            'physics': 'physics',
            'mathematics': 'math',
            'statistics': 'stat',
            'biology': 'q-bio',
            'economics': 'econ'
        }
        
        for keyword, category in category_mapping.items():
            if keyword in query_lower:
                return category
                
        return 'cs.LG'  # Default to machine learning

    def _format_paper_data(self, paper):
        """Format paper data to consistent structure."""
        if isinstance(paper, dict):
            return {
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'abstract': paper.get('abstract', ''),
                'categories': paper.get('categories', []),
                'arxiv_url': paper.get('arxiv_url', ''),
                'pdf_link': paper.get('pdf_link', ''),
                'published': paper.get('published', ''),
                'updated': paper.get('updated', ''),
                'arxiv_id': paper.get('arxiv_id', '')
            }
        else:
            # Handle other paper object types
            return {
                'title': getattr(paper, 'title', ''),
                'authors': getattr(paper, 'authors', []),
                'abstract': getattr(paper, 'abstract', ''),
                'categories': getattr(paper, 'categories', []),
                'arxiv_url': getattr(paper, 'arxiv_url', ''),
                'pdf_link': getattr(paper, 'pdf_link', ''),
                'published': getattr(paper, 'published', ''),
                'updated': getattr(paper, 'updated', ''),
                'arxiv_id': getattr(paper, 'arxiv_id', '')
            }

    async def format_paper_for_learning(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Format paper for learning."""
        try:
            formatted = {
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', []),
                'content': paper.get('abstract', ''),  # Default to abstract if no content
                'metadata': {
                    'arxiv_id': paper.get('arxiv_id', ''),
                    'categories': paper.get('categories', []),
                    'published': paper.get('published', ''),
                    'updated': paper.get('updated', ''),
                    'arxiv_url': paper.get('arxiv_url', ''),
                    'pdf_link': paper.get('pdf_link', '')
                }
            }
            
            # Try to fetch PDF content if available
            if self.fetcher and paper.get('pdf_link'):
                try:
                    if hasattr(self.fetcher, 'fetch_pdf'):
                        pdf_content = await self.fetcher.fetch_pdf(paper['pdf_link'])
                        if pdf_content:
                            formatted['content'] = pdf_content
                except Exception as e:
                    logger.warning(f"Could not fetch PDF: {str(e)}")
                    
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting paper: {str(e)}")
            return {
                'title': paper.get('title', ''),
                'content': paper.get('abstract', ''),
                'metadata': {}
            }


class DuckDuckGoSearcher:
    """Class for performing searches using DuckDuckGo API via OARC-Crawlers DDGCrawler."""
    
    def __init__(self):
        """Initialize the DuckDuckGo searcher using OARC-Crawlers."""
        self.searcher = None
        self.use_fallback = False
        
        try:
            self.searcher = DDGCrawler(data_dir=DATA_DIR)
            logger.info("DuckDuckGoSearcher initialized successfully")
        except ImportError:
            logger.error("oarc_crawlers.DDGCrawler not available")
            self._init_fallback()
        except Exception as e:
            logger.warning(f"Failed to initialize DDGCrawler: {e}")
            self._init_fallback()
        
        self.rate_limit_delay = 3
    
    def _init_fallback(self):
        """Initialize fallback DuckDuckGo search."""
        try:
            # Use direct DDGS if OARC-Crawlers DDG fails
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.use_fallback = True
            logger.info("Using fallback DDGS for DuckDuckGo search")
        except ImportError:
            logger.error("Neither OARC DDGCrawler nor duckduckgo_search available")
            self.ddgs = None
    
    async def text_search(self, search_query, max_results=5):
        """Perform an async text search using available DDG implementation."""
        # First try OARC-Crawlers
        if self.searcher and not self.use_fallback:
            try:
                # Use the correct API method based on detection
                results = await self.searcher.text_search(search_query, max_results=max_results)
                return results
            except Exception as e:
                logger.warning(f"OARC DDG search failed, trying fallback: {e}")
                self.use_fallback = True
                self._init_fallback()
        
        # Try fallback method
        if self.use_fallback and hasattr(self, 'ddgs') and self.ddgs:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, 
                    self._sync_ddg_search, 
                    search_query, 
                    max_results
                )
                return results
            except Exception as e:
                logger.error(f"Fallback DDG search failed: {e}")
        
        logger.error("No DuckDuckGo searcher available")
        return []
    
    def _sync_ddg_search(self, query, max_results):
        """Synchronous DDG search for executor."""
        try:
            results = []
            for result in self.ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                })
            return results
        except Exception as e:
            logger.error(f"Sync DDG search error: {e}")
            return []
    
    async def image_search(self, search_query, max_results=5):
        """Perform an async image search."""
        if not self.searcher:
            logger.error("DuckDuckGo searcher not available")
            return []
            
        try:
            results = await self.searcher.image_search(search_query, max_results=max_results)
            return results
        except Exception as e:
            logger.error(f"Error in DuckDuckGo image search: {e}")
            return []
    
    async def news_search(self, search_query, max_results=5):
        """Perform an async news search."""
        if not self.searcher:
            logger.error("DuckDuckGo searcher not available")
            return []
            
        try:
            results = await self.searcher.news_search(search_query, max_results=max_results)
            return results
        except Exception as e:
            logger.error(f"Error in DuckDuckGo news search: {e}")
            return []


class GitHubCrawler:
    """Class for crawling and extracting content from GitHub repositories.
    
    This is a wrapper around the oarc-crawlers GHCrawler class.
    """
    
    def __init__(self, data_dir=None):
        """Initialize the GitHub Crawler."""
        self.data_dir = data_dir or DATA_DIR
        try:
            self.crawler = GHCrawler(data_dir=self.data_dir)
            logger.info("GitHubCrawler initialized successfully")
        except ImportError:
            logger.error("oarc_crawlers.GHCrawler not available")
            self.crawler = None
        except Exception as e:
            logger.error(f"Failed to initialize GHCrawler: {e}")
            self.crawler = None
        
        self.github_data_dir = Path(f"{self.data_dir}/github_repos")
        self.github_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_repo_info_from_url(url: str) -> Tuple[str, str, str]:
        """Extract repository owner and name from GitHub URL."""
        # Simple regex parsing if OARC method not available
        import re
        pattern = r'github\.com/([^/]+)/([^/]+)'
        match = re.search(pattern, url)
        if match:
            owner, repo = match.groups()
            repo = repo.replace('.git', '')  # Remove .git suffix if present
            return owner, repo, 'main'  # Default branch
        raise ValueError(f"Invalid GitHub URL: {url}")

    def get_repo_dir_path(self, owner: str, repo_name: str) -> Path:
        """Get the directory path for storing repository data."""
        return self.github_data_dir / f"{owner}_{repo_name}"

    async def clone_repo(self, repo_url: str, temp_dir: Optional[str] = None) -> Path:
        """Clone a GitHub repository to a temporary directory."""
        if not self.crawler:
            raise Exception("GitHubCrawler not available")
        return await self.crawler.clone_repo(repo_url, temp_dir)

    async def get_repo_summary(self, repo_url: str) -> str:
        """Get a summary of the repository."""
        if not self.crawler:
            return "GitHubCrawler not available"
        return await self.crawler.get_repo_summary(repo_url)

    async def find_similar_code(self, repo_url: str, code_snippet: str) -> str:
        """Find similar code in the repository."""
        if not self.crawler:
            return "GitHubCrawler not available"
        return await self.crawler.find_similar_code(repo_url, code_snippet)


class ParquetStorageWrapper:
    """Class for handling data storage in Parquet format.
    
    This is a wrapper around the oarc-crawlers ParquetStorage class.
    """
    
    def __init__(self, data_dir=DATA_DIR):
        """Initialize the ParquetStorage wrapper."""
        self.data_dir = Path(data_dir)
        try:
            self.storage = ParquetStorage()
            logger.info("ParquetStorageWrapper initialized successfully")
        except ImportError:
            logger.error("oarc_crawlers.ParquetStorage not available")
            self.storage = None
        except Exception as e:
            logger.error(f"Failed to initialize ParquetStorage: {e}")
            self.storage = None

    def save_to_parquet(self, data: Union[Dict, List, pd.DataFrame], file_path: Union[str, Path]) -> bool:
        """Save data to a Parquet file."""
        if not self.storage:
            logger.error("ParquetStorage not available")
            return False
        try:
            return self.storage.save_to_parquet(data, file_path)
        except Exception as e:
            logger.error(f"Error saving to parquet: {str(e)}")
            return False

    def load_from_parquet(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Load data from a Parquet file."""
        if not self.storage:
            logger.error("ParquetStorage not available")
            return None
        try:
            return self.storage.load_from_parquet(file_path)
        except Exception as e:
            logger.error(f"Error loading from parquet: {str(e)}")
            return None

    def append_to_parquet(self, data: Union[Dict, List, pd.DataFrame], file_path: Union[str, Path]) -> bool:
        """Append data to an existing Parquet file or create a new one."""
        if not self.storage:
            logger.error("ParquetStorage not available")
            return False
        try:
            return self.storage.append_to_parquet(data, file_path)
        except Exception as e:
            logger.error(f"Error appending to parquet: {str(e)}")
            return False


# Add API detection function
def detect_oarc_api():
    """Detect available OARC-Crawlers API methods."""
    api_info = {
        'arxiv_methods': [],
        'ddg_methods': [],
        'github_methods': [],
        'web_methods': [],
        'has_oarc': False
    }
    
    try:
        from oarc_crawlers import ArxivCrawler, DDGCrawler, GHCrawler, WebCrawler
        api_info['has_oarc'] = True
        
        # Check ArxivCrawler methods
        try:
            arxiv = ArxivCrawler()
            api_info['arxiv_methods'] = [method for method in dir(arxiv) if not method.startswith('_') and callable(getattr(arxiv, method))]
        except Exception as e:
            logger.warning(f"Could not inspect ArxivCrawler: {e}")
        
        # Check DDGCrawler methods  
        try:
            ddg = DDGCrawler()
            api_info['ddg_methods'] = [method for method in dir(ddg) if not method.startswith('_') and callable(getattr(ddg, method))]
        except Exception as e:
            logger.warning(f"Could not inspect DDGCrawler: {e}")
        
        # Check GHCrawler methods
        try:
            gh = GHCrawler()
            api_info['github_methods'] = [method for method in dir(gh) if not method.startswith('_') and callable(getattr(gh, method))]
        except Exception as e:
            logger.warning(f"Could not inspect GHCrawler: {e}")
        
        # Check WebCrawler methods
        try:
            web = WebCrawler()
            api_info['web_methods'] = [method for method in dir(web) if not method.startswith('_') and callable(getattr(web, method))]
        except Exception as e:
            logger.warning(f"Could not inspect WebCrawler: {e}")
        
        logger.info(f"OARC API Detection - ArXiv methods: {api_info['arxiv_methods']}")
        logger.info(f"OARC API Detection - DDG methods: {api_info['ddg_methods']}")
        logger.info(f"OARC API Detection - GitHub methods: {api_info['github_methods']}")
        logger.info(f"OARC API Detection - Web methods: {api_info['web_methods']}")
        
    except ImportError:
        logger.warning("OARC-Crawlers not available")
    except Exception as e:
        logger.error(f"Error detecting OARC API: {e}")
    
    return api_info

# Call this during module initialization
_API_INFO = detect_oarc_api()