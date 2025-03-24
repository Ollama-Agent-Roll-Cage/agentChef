import os
import re
import gzip
import tarfile
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from datetime import datetime
import time

# For DuckDuckGo search
from duckduckgo_search import DDGS

# Configure logging
def setup_logging(log_dir):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'research_assistant.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ResearchAssistant")

def perform_web_search(queries, update_callback=None):
    """
    Perform web search using DuckDuckGo.
    
    Args:
        queries: List of search queries
        update_callback: Optional callback function to report progress
        
    Returns:
        List of search results
    """
    all_results = []
    
    for query in queries[:3]:  # Limit to 3 queries
        if update_callback:
            update_callback(f"Searching web for: {query}")
            
        try:
            # Search for text
            text_results = DDGS().text(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                max_results=5
            )
            all_results.extend(text_results or [])
            
            # Add a delay before the next search to avoid rate limiting
            time.sleep(2)
            
            # Search for news
            news_results = DDGS().news(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                max_results=3
            )
            all_results.extend(news_results or [])
            
            # Add another delay before the next query
            time.sleep(2)
            
        except Exception as e:
            if update_callback:
                update_callback(f"Search error for query '{query}': {str(e)}")
    
    # Remove duplicates by URL
    unique_results = []
    seen_urls = set()
    for result in all_results:
        url = result.get('href') or result.get('url')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    if update_callback:
        update_callback(f"Found {len(unique_results)} unique web results")
        
    return unique_results

def search_arxiv_papers(queries, update_callback=None):
    """
    Search for papers on ArXiv using the provided queries.
    
    Args:
        queries: List of search queries
        update_callback: Optional callback function to report progress
        
    Returns:
        List of paper information dictionaries
    """
    all_papers = []
    
    for query in queries:
        if update_callback:
            update_callback(f"Searching ArXiv for: {query}")
            
        try:
            # URL encode the query
            encoded_query = urllib.parse.quote(query)
            base_url = 'http://export.arxiv.org/api/query'
            search_url = f"{base_url}?search_query=all:{encoded_query}&start=0&max_results=5"
            
            with urllib.request.urlopen(search_url) as response:
                response_data = response.read().decode('utf-8')
            
            # Parse the XML response
            root = ET.fromstring(response_data)
            
            # Define namespace mapping
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Extract papers
            entries = root.findall('.//atom:entry', namespaces)
            
            for entry in entries:
                paper_id_element = entry.find('.//arxiv:id', namespaces)
                paper_id = paper_id_element.text.split('/')[-1] if paper_id_element is not None else None
                
                title = entry.find('atom:title', namespaces).text.strip().replace('\n', ' ')
                authors = [author.find('atom:name', namespaces).text for author in entry.findall('atom:author', namespaces)]
                abstract = entry.find('atom:summary', namespaces).text.strip().replace('\n', ' ')
                published = entry.find('atom:published', namespaces).text
                
                paper_info = {
                    'arxiv_id': paper_id,
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'published': published,
                    'query': query
                }
                
                all_papers.append(paper_info)
                
        except Exception as e:
            if update_callback:
                update_callback(f"ArXiv search error for query '{query}': {str(e)}")
    
    # Remove duplicates by paper ID
    unique_papers = []
    seen_ids = set()
    for paper in all_papers:
        if paper['arxiv_id'] not in seen_ids:
            seen_ids.add(paper['arxiv_id'])
            unique_papers.append(paper)
    
    if update_callback:
        update_callback(f"Found {len(unique_papers)} unique ArXiv papers")
    
    return unique_papers

def download_paper_source(arxiv_id, temp_dir, update_callback=None):
    """
    Download source files for a paper from ArXiv.
    
    Args:
        arxiv_id: ArXiv ID of the paper
        temp_dir: Temporary directory to store downloaded files
        update_callback: Optional callback function to report progress
        
    Returns:
        Path to the directory containing the paper source files
    """
    try:
        # Create directory for paper
        paper_dir = temp_dir / arxiv_id
        paper_dir.mkdir(exist_ok=True)
        
        # Download source file
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        temp_file = paper_dir / "source.tar.gz"
        
        with urllib.request.urlopen(source_url) as response:
            with open(temp_file, 'wb') as f:
                f.write(response.read())
        
        # Try to extract as tar.gz
        try:
            with tarfile.open(temp_file, 'r:gz') as tar:
                tar.extractall(path=paper_dir)
                if update_callback:
                    update_callback(f"Extracted tar.gz source for {arxiv_id}")
        except tarfile.ReadError:
            # If not tar.gz, try as gzip
            try:
                with gzip.open(temp_file, 'rb') as gz:
                    with open(paper_dir / 'main.tex', 'wb') as f:
                        f.write(gz.read())
                if update_callback:
                    update_callback(f"Extracted gzip source for {arxiv_id}")
            except Exception:
                if update_callback:
                    update_callback(f"Source for {arxiv_id} is not in standard format")
        
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()
            
        return paper_dir
        
    except Exception as e:
        if update_callback:
            update_callback(f"Error downloading source for {arxiv_id}: {str(e)}")
        return None

def extract_latex_content(paper_dir, update_callback=None):
    """
    Extract and concatenate LaTeX content from source files.
    
    Args:
        paper_dir: Path to the directory containing the paper source files
        update_callback: Optional callback function to report progress
        
    Returns:
        String containing the extracted LaTeX content
    """
    try:
        latex_content = []
        
        # Find all .tex files
        tex_files = list(paper_dir.glob('**/*.tex'))
        
        if not tex_files:
            if update_callback:
                update_callback(f"No .tex files found in {paper_dir}")
            return ""
        
        # Try to identify the main .tex file
        main_candidates = [f for f in tex_files if f.name.lower() in ('main.tex', 'paper.tex', 'manuscript.tex')]
        
        if main_candidates:
            main_file = main_candidates[0]
        else:
            # Find the largest .tex file or just use the first one
            main_file = max(tex_files, key=lambda f: f.stat().st_size)
        
        # Read the main file
        with open(main_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            latex_content.append(f"% Main file: {main_file.name}\n{content}")
        
        # Read other important files like abstract, intro, etc.
        important_sections = ['abstract', 'intro', 'introduction', 'method', 'approach', 
                            'result', 'conclusion', 'discussion']
        
        for tex_file in tex_files:
            if tex_file != main_file:
                if any(section in tex_file.name.lower() for section in important_sections):
                    try:
                        with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            latex_content.append(f"% From file: {tex_file.name}\n{content}")
                    except Exception as e:
                        if update_callback:
                            update_callback(f"Error reading {tex_file}: {str(e)}")
        
        return "\n\n".join(latex_content)
        
    except Exception as e:
        if update_callback:
            update_callback(f"Error extracting LaTeX content: {str(e)}")
        return ""
