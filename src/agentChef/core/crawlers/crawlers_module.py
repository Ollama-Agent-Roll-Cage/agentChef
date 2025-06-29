"""crawlers_module.py
This module provides agent-focused wrappers around crawling functionality for:
- WebCrawler: General web page crawling
- ArxivSearcher: ArXiv paper lookup and parsing
- DuckDuckGoSearcher: DuckDuckGo search API integration
- GitHubCrawler: GitHub repository cloning and extraction
- AgentDataManager: Unified interface for agent data management

This version integrates with the new agent-focused storage and prompt system
to better support agent training and knowledge management.

Written By: @Borcherdingl
Date: 6/29/2025
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
try:
    from oarc_crawlers import (
        WebCrawler,
        ArxivCrawler, 
        DDGCrawler,
        GHCrawler,
        ParquetStorage as OARCParquetStorage
    )
    HAS_OARC_CRAWLERS = True
except ImportError:
    HAS_OARC_CRAWLERS = False
    OARCParquetStorage = None
    logging.warning("oarc-crawlers not available. Some features will be limited.")

# Import our new agent-focused components
try:
    from agentChef.core.storage.conversation_storage import ConversationStorage, KnowledgeEntry
    from agentChef.core.prompts.agent_prompt_manager import AgentPromptManager
    HAS_AGENT_COMPONENTS = True
except ImportError:
    HAS_AGENT_COMPONENTS = False
    logging.warning("Agent components not available. Some features will be limited.")

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
                'arxiv_id': paper.get('id', '')
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
            if OARCParquetStorage:
                self.storage = OARCParquetStorage()
                logger.info("ParquetStorageWrapper initialized with OARC storage")
            else:
                self.storage = None
                logger.warning("OARC ParquetStorage not available")
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


class AgentDataManager:
    """
    Unified interface for managing agent data from various sources.
    Integrates crawling, storage, and agent-specific processing.
    """
    
    def __init__(self, agent_name: str, data_dir: Optional[str] = None):
        """
        Initialize the AgentDataManager.
        
        Args:
            agent_name: Name of the agent this manager serves
            data_dir: Directory for storing agent data
        """
        self.agent_name = agent_name
        self.data_dir = Path(data_dir) if data_dir else Path(DATA_DIR)
        
        # Initialize storage and prompt management
        if HAS_AGENT_COMPONENTS:
            self.storage = ConversationStorage(self.data_dir)
            self.prompt_manager = AgentPromptManager(self.data_dir / "prompts")
        else:
            self.storage = None
            self.prompt_manager = None
            
        # Initialize crawlers
        self.web_crawler = WebCrawlerWrapper()
        self.arxiv_searcher = ArxivSearcher()
        self.ddg_searcher = DuckDuckGoSearcher()
        self.github_crawler = GitHubCrawler(data_dir)
        
        logger.info(f"Initialized AgentDataManager for agent: {agent_name}")
    
    def crawl_and_store_for_agent(self, source_type: str, source_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crawl data from a source and store it in agent-specific format.
        
        Args:
            source_type: Type of source (web, arxiv, ddg, github)
            source_params: Parameters for the specific crawler
            
        Returns:
            Dict containing crawl results and storage info
        """
        try:
            # Crawl data based on source type
            if source_type == "web":
                crawler_result = self._crawl_web_for_agent(source_params)
            elif source_type == "arxiv":
                crawler_result = self._crawl_arxiv_for_agent(source_params)
            elif source_type == "ddg":
                crawler_result = self._crawl_ddg_for_agent(source_params)
            elif source_type == "github":
                crawler_result = self._crawl_github_for_agent(source_params)
            else:
                return {"error": f"Unknown source type: {source_type}"}
            
            # Store results in agent-specific format
            if crawler_result and not crawler_result.get("error"):
                storage_result = self._store_crawled_data_for_agent(
                    crawler_result, 
                    source_type, 
                    source_params
                )
                return {
                    "crawl_result": crawler_result,
                    "storage_result": storage_result,
                    "agent_name": self.agent_name
                }
            else:
                return {"error": "Failed to crawl data", "details": crawler_result}
                
        except Exception as e:
            logger.error(f"Error in crawl_and_store_for_agent: {e}")
            return {"error": str(e)}
    
    def _store_crawled_data_for_agent(self, crawled_data: Dict[str, Any], 
                                    source_type: str, source_params: Dict[str, Any]) -> Dict[str, Any]:
        """Store crawled data in agent-specific format."""
        if not self.storage:
            return {"error": "Storage not available"}
        
        try:
            # Convert crawled data to knowledge entries
            knowledge_entries = self._convert_to_knowledge_entries(crawled_data, source_type)
            
            # Save knowledge entries
            success = self.storage.save_knowledge(self.agent_name, knowledge_entries)
            
            # Also create a conversation about the crawled data
            conversation = self._create_conversation_from_crawled_data(crawled_data, source_type)
            conv_id = None
            if conversation:
                from agentChef.core.storage.conversation_storage import ConversationMetadata
                metadata = ConversationMetadata(
                    agent_name=self.agent_name,
                    conversation_id="",
                    created_at="",
                    updated_at="",
                    context=f"crawled_{source_type}",
                    source="crawler",
                    tags=[source_type, "crawled_data"]
                )
                conv_id = self.storage.save_conversation(self.agent_name, conversation, metadata)
            
            return {
                "knowledge_saved": success,
                "knowledge_count": len(knowledge_entries),
                "conversation_id": conv_id,
                "agent_name": self.agent_name
            }
            
        except Exception as e:
            logger.error(f"Error storing crawled data: {e}")
            return {"error": str(e)}
    
    def _convert_to_knowledge_entries(self, crawled_data: Dict[str, Any], 
                                    source_type: str) -> List['KnowledgeEntry']:
        """Convert crawled data to knowledge entries."""
        entries = []
        source = crawled_data.get("source", source_type)
        timestamp = crawled_data.get("timestamp", datetime.now().isoformat())
        
        if source_type == "web":
            entry = KnowledgeEntry(
                agent_name=self.agent_name,
                entry_id=f"web_{hash(crawled_data.get('url', ''))}",
                topic=f"web_content_{crawled_data.get('url', 'unknown')}",
                content=crawled_data.get("content", ""),
                knowledge_type="web_content",
                source=crawled_data.get("url", "web"),
                created_at=timestamp,
                tags=["web", "crawled"]
            )
            entries.append(entry)
            
        return entries
    
    def _create_conversation_from_crawled_data(self, crawled_data: Dict[str, Any], 
                                             source_type: str) -> Optional[List[Dict[str, Any]]]:
        """Create a conversation about the crawled data."""
        try:
            if source_type == "web":
                url = crawled_data.get("url", "")
                content_preview = crawled_data.get("content", "")[:200] + "..."
                
                conversation = [
                    {"from": "human", "value": f"What did you learn from crawling {url}?"},
                    {"from": "gpt", "value": f"I crawled the webpage at {url} and extracted content: {content_preview}"}
                ]
                return conversation
                
        except Exception as e:
            logger.error(f"Error creating conversation from crawled data: {e}")
            return None
    
    def get_agent_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of all knowledge stored for this agent."""
        if not self.storage:
            return {"error": "Storage not available"}
        
        try:
            # Get all knowledge entries
            all_knowledge = self.storage.load_knowledge(self.agent_name)
            
            # Get statistics
            stats = self.storage.get_agent_stats(self.agent_name)
            
            return {
                "agent_name": self.agent_name,
                "total_knowledge_entries": len(all_knowledge),
                "agent_stats": stats,
                "recent_entries": [entry.topic for entry in all_knowledge[-5:]] if all_knowledge else []
            }
            
        except Exception as e:
            logger.error(f"Error getting agent knowledge summary: {e}")
            return {"error": str(e)}


# Add API detection function
def detect_oarc_crawlers_api():
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
_API_INFO = detect_oarc_crawlers_api()