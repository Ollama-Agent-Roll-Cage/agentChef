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
import random  # Add this import
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
        self.crawler = WebCrawler(data_dir=DATA_DIR)
        
    async def fetch_url_content(self, url):
        """Fetch content from a URL.
        
        Args:
            url (str): The URL to fetch content from
            
        Returns:
            str: HTML content of the page or None if failed
        """
        return await self.crawler.fetch_url_content(url)

    async def extract_text_from_html(self, html):
        """Extract main text content from HTML using BeautifulSoup.
        
        Args:
            html (str): HTML content
            
        Returns:
            str: Extracted text content
        """
        if not html:
            return "Failed to extract text from the webpage."
        
        return await self.crawler.extract_text_from_html(html)

    async def extract_pypi_content(self, html, package_name):
        """Specifically extract PyPI package documentation from HTML.
        
        Args:
            html (str): HTML content from PyPI page
            package_name (str): Name of the package
            
        Returns:
            dict: Structured package data or None if failed
        """
        return await self.crawler.extract_pypi_content(html, package_name)
    
    async def format_pypi_info(self, package_data):
        """Format PyPI package data into a readable markdown format.
        
        Args:
            package_data (dict): Package data from PyPI API
            
        Returns:
            str: Formatted markdown text
        """
        return await self.crawler.format_pypi_info(package_data)


class ArxivSearcher:
    """Class for searching and retrieving ArXiv papers.
    
    This is a wrapper around the oarc-crawlers ArxivCrawler class.
    """
    
    def __init__(self):
        """Initialize the ArXiv searcher with the data directory."""
        self.fetcher = ArxivCrawler(data_dir=DATA_DIR)
        self.rate_limit_delay = 3  # seconds between requests
    
    @staticmethod
    def extract_arxiv_id(url_or_id):
        """Extract arXiv ID from a URL or direct ID string.
        
        Args:
            url_or_id (str): ArXiv URL or direct ID
            
        Returns:
            str: Extracted ArXiv ID
            
        Raises:
            ValueError: If ID cannot be extracted
        """
        return ArxivCrawler.extract_arxiv_id(url_or_id)

    async def fetch_paper_info(self, arxiv_id):
        """Fetch paper metadata from arXiv API.
        
        Args:
            arxiv_id (str): ArXiv paper ID
            
        Returns:
            dict: Paper metadata
            
        Raises:
            ValueError: If paper cannot be found
            ConnectionError: If connection to ArXiv fails
        """
        return await self.fetcher.fetch_paper_info(arxiv_id)

    async def format_paper_for_learning(self, paper_info):
        """Format paper information for learning.
        
        Args:
            paper_info (dict): Paper metadata
            
        Returns:
            str: Formatted markdown text
        """
        # Use the same format as in the original ArxivSearcher
        formatted_text = f"""# {paper_info['title']}

**Authors:** {', '.join(paper_info['authors'])}

**Published:** {paper_info['published'][:10]}

**Categories:** {', '.join(paper_info['categories'])}

## Abstract
{paper_info['abstract']}

**Links:**
- [ArXiv Page]({paper_info['arxiv_url']})
- [PDF Download]({paper_info['pdf_link']})
"""
        if 'comment' in paper_info and paper_info['comment']:
            formatted_text += f"\n**Comments:** {paper_info['comment']}\n"
            
        if 'journal_ref' in paper_info and paper_info['journal_ref']:
            formatted_text += f"\n**Journal Reference:** {paper_info['journal_ref']}\n"
            
        if 'doi' in paper_info and paper_info['doi']:
            formatted_text += f"\n**DOI:** {paper_info['doi']}\n"
            
        return formatted_text

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Old search method for backward compatibility."""
        return await self.search_papers(query, max_results=max_results)
        
    async def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        try:
            papers = []
            # The search method returns a dict not an async iterator
            search_results = await self.fetcher.search(query, limit=max_results)
            
            # Process the results which should be in a list or similar structure
            for paper in search_results.get('results', []):
                papers.append({
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', []),
                    'abstract': paper.get('abstract', ''),
                    'categories': paper.get('categories', []),
                    'arxiv_url': paper.get('arxiv_url', ''),
                    'pdf_link': paper.get('pdf_link', ''),
                    'published': paper.get('published', ''),
                    'updated': paper.get('updated', ''),
                    'arxiv_id': paper.get('arxiv_id', '')
                })
                
                if len(papers) >= max_results:
                    break
                
                # Add rate limiting delay
                await asyncio.sleep(self.rate_limit_delay)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {str(e)}")
            return []


# Update DuckDuckGo imports
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
    logging.warning("DuckDuckGo search not available. Install with 'pip install -U duckduckgo-search'")

class DuckDuckGoSearcher:
    """Class for performing searches using DuckDuckGo API."""
    
    def __init__(self):
        """Initialize the DuckDuckGo searcher with the data directory."""
        self.searcher = DDGCrawler(data_dir=DATA_DIR)
        self.rate_limit_delay = 20  # Increase initial delay to 20 seconds
        self.max_retries = 3  # Reduce max retries to avoid excessive attempts
        self.retry_backoff = 3  # More aggressive backoff multiplier
        self.session = None  # Will store DDGS session
    
    async def text_search(self, search_query, max_results=5):
        """Perform an async text search using DuckDuckGo."""
        if not HAS_DDG:
            return [{
                "title": "Search Unavailable",
                "link": "",
                "snippet": "DuckDuckGo search package not installed. Install with: pip install -U duckduckgo-search"
            }]
        
        # Initialize session if needed
        if self.session is None:
            self.session = DDGS()
            # Add random user agent rotation
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
        
        current_delay = self.rate_limit_delay
        results = []
        
        for attempt in range(self.max_retries):
            try:
                # Add random delay component
                random_delay = current_delay * (0.5 + random.random())
                await asyncio.sleep(random_delay)
                
                # Use a with statement for proper session handling
                with self.session as ddgs:
                    ddgs.timeout = 60  # Increase timeout
                    
                    # Try text search first
                    try:
                        for r in ddgs.text(search_query, max_results=max_results):
                            results.append({
                                "title": r.get("title", ""),
                                "link": r.get("link", ""),
                                "snippet": r.get("body", "")
                            })
                            # Add random delay between results
                            await asyncio.sleep(2 + random.random() * 3)
                            
                            if len(results) >= max_results:
                                return results
                                
                    except Exception as e:
                        if "202 Ratelimit" in str(e):
                            # Close and recreate session on rate limit
                            self.session = DDGS()
                            current_delay *= self.retry_backoff
                            continue
                        raise
                        
                # If we got here with results, return them
                if results:
                    return results
                    
                current_delay *= self.retry_backoff
                    
            except Exception as e:
                logger.error(f"Error performing DuckDuckGo search (attempt {attempt + 1}/{self.max_retries}): {e}")
                current_delay *= self.retry_backoff
                
        # If all retries failed, return error result
        return [{
            "title": "Search Error",
            "link": "",
            "snippet": f"DuckDuckGo search failed after {self.max_retries} attempts due to rate limiting. Please try again later."
        }]

class GitHubCrawler:
    """Class for crawling and extracting content from GitHub repositories.
    
    This is a wrapper around the oarc-crawlers GitHubCrawler class.
    """
    
    def __init__(self, data_dir=None):
        """Initialize the GitHub Crawler.
        
        Args:
            data_dir (str, optional): Directory to store data. Defaults to DATA_DIR.
        """
        self.data_dir = data_dir or DATA_DIR
        self.crawler = GHCrawler(data_dir=self.data_dir)  # Fixed class name
        self.github_data_dir = Path(f"{self.data_dir}/github_repos")
        self.github_data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def extract_repo_info_from_url(url: str) -> Tuple[str, str, str]:
        """Extract repository owner and name from GitHub URL.
        
        Args:
            url (str): GitHub repository URL
            
        Returns:
            Tuple[str, str, str]: Repository owner, name, and branch (if available)
            
        Raises:
            ValueError: If URL is not a valid GitHub repository URL
        """
        return GHCrawler.extract_repo_info_from_url(url)

    def get_repo_dir_path(self, owner: str, repo_name: str) -> Path:
        """Get the directory path for storing repository data.
        
        Args:
            owner (str): Repository owner
            repo_name (str): Repository name
            
        Returns:
            Path: Directory path
        """
        return self.github_data_dir / f"{owner}_{repo_name}"

    async def clone_repo(self, repo_url: str, temp_dir: Optional[str] = None) -> Path:
        """Clone a GitHub repository to a temporary directory.
        
        Args:
            repo_url (str): GitHub repository URL
            temp_dir (str, optional): Temporary directory path. If None, creates one.
            
        Returns:
            Path: Path to the cloned repository
            
        Raises:
            Exception: If cloning fails
        """
        return await self.crawler.clone_repo(repo_url, temp_dir)

    def is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if file is binary, False otherwise
        """
        return self.crawler.is_binary_file(file_path)

    async def process_repo_to_dataframe(self, repo_path: Path, max_file_size_kb: int = 500):
        """Process repository files and convert to DataFrame.
        
        Args:
            repo_path (Path): Path to cloned repository
            max_file_size_kb (int): Maximum file size in KB to process
            
        Returns:
            pd.DataFrame: DataFrame containing file information
        """
        return await self.crawler.process_repo_to_dataframe(repo_path, max_file_size_kb)

    @staticmethod
    def get_language_from_extension(extension: str) -> str:
        """Get programming language name from file extension.
        
        Args:
            extension (str): File extension with leading dot
            
        Returns:
            str: Language name or 'Unknown'
        """
        return OARCGitHubCrawler.get_language_from_extension(extension)

    async def clone_and_store_repo(self, repo_url: str) -> str:
        """Clone a GitHub repository and store its data in Parquet format.
        
        Args:
            repo_url (str): GitHub repository URL
            
        Returns:
            str: Path to the Parquet file containing repository data
            
        Raises:
            Exception: If cloning or processing fails
        """
        return await self.crawler.clone_and_store_repo(repo_url)

    async def query_repo_content(self, repo_url: str, query: str) -> str:
        """Query repository content using natural language.
        
        Args:
            repo_url (str): GitHub repository URL
            query (str): Natural language query about the repository
            
        Returns:
            str: Query result formatted as markdown
            
        Raises:
            Exception: If querying fails
        """
        return await self.crawler.query_repo_content(repo_url, query)

    async def get_repo_summary(self, repo_url: str) -> str:
        """Get a summary of the repository.
        
        Args:
            repo_url (str): GitHub repository URL
            
        Returns:
            str: Repository summary formatted as markdown
        """
        return await self.crawler.get_repo_summary(repo_url)

    async def find_similar_code(self, repo_url: str, code_snippet: str) -> str:
        """Find similar code in the repository.
        
        Args:
            repo_url (str): GitHub repository URL
            code_snippet (str): Code snippet to find similar code for
            
        Returns:
            str: Similar code findings formatted as markdown
        """
        return await self.crawler.find_similar_code(repo_url, code_snippet)
