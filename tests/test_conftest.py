"""
test_conftest.py - Common fixtures and utilities for agentChef tests

This file contains:
1. Fixtures for mocking external dependencies
2. Common test data for reuse across test modules
3. Configuration for pytest
"""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from agentChef.ollama_interface import OllamaInterface
from agentChef.conversation_generator import OllamaConversationGenerator
from agentChef.dataset_expander import DatasetExpander
from agentChef.dataset_cleaner import DatasetCleaner
from agentChef.crawlers_module import WebCrawler, ArxivSearcher, DuckDuckGoSearcher, GitHubCrawler
from agentChef.udrags import ResearchManager

# Sample test data
SAMPLE_PAPER_CONTENT = """
Attention mechanisms have become an integral part of compelling sequence modeling
and transduction models in various tasks, allowing modeling of dependencies without
regard to their distance in the input or output sequences. In this paper we present the
Transformer, a model architecture eschewing recurrence and instead relying entirely
on an attention mechanism to draw global dependencies between input and output.
"""

SAMPLE_CONVERSATIONS = [
    [
        {"from": "human", "value": "What are attention mechanisms?"},
        {"from": "gpt", "value": "Attention mechanisms allow models to focus on specific parts of input."}
    ],
    [
        {"from": "human", "value": "Can you explain transformers?"},
        {"from": "gpt", "value": "Transformers are neural network architectures that use attention mechanisms."}
    ]
]

SAMPLE_HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <style>
        body { font-family: Arial; }
    </style>
    <script>
        console.log('Hello, world!');
    </script>
</head>
<body>
    <h1>Test Heading</h1>
    <p>This is a test paragraph.</p>
    <p>This is another paragraph with <a href="https://example.com">a link</a>.</p>
</body>
</html>"""

ARXIV_PAPER_INFO = {
    'arxiv_id': '1706.03762',
    'title': 'Attention Is All You Need',
    'authors': ['Ashish Vaswani', 'Noam Shazeer'],
    'abstract': 'We propose a new simple network architecture, the Transformer...',
    'published': '2017-06-12',
    'pdf_link': 'https://arxiv.org/pdf/1706.03762.pdf',
    'arxiv_url': 'https://arxiv.org/abs/1706.03762',
    'categories': ['cs.CL', 'cs.LG']
}

@pytest.fixture
def temp_data_dir():
    """Fixture to create a temporary data directory for tests."""
    data_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    Path(f"{data_dir}/papers").mkdir(parents=True, exist_ok=True)
    Path(f"{data_dir}/datasets").mkdir(parents=True, exist_ok=True)
    Path(f"{data_dir}/crawls").mkdir(parents=True, exist_ok=True)
    Path(f"{data_dir}/github_repos").mkdir(parents=True, exist_ok=True)
    
    # Set environment variable
    old_data_dir = os.environ.get('DATA_DIR')
    os.environ['DATA_DIR'] = data_dir
    
    yield data_dir
    
    # Clean up the directory
    import shutil
    shutil.rmtree(data_dir)
    
    # Restore old environment variable
    if old_data_dir:
        os.environ['DATA_DIR'] = old_data_dir
    else:
        del os.environ['DATA_DIR']

@pytest.fixture
def mock_ollama_interface():
    """Fixture to provide a mocked OllamaInterface."""
    mock_interface = MagicMock(spec=OllamaInterface)
    
    # Configure the mock response for chat
    mock_interface.chat.return_value = {
        "message": {
            "content": "This is a test response."
        }
    }
    
    # Configure the mock response for embeddings
    mock_interface.embeddings.return_value = [0.1, 0.2, 0.3]
    
    # Configure the mock is_available
    mock_interface.is_available.return_value = True
    
    return mock_interface

@pytest.fixture
def mock_conversation_generator(mock_ollama_interface):
    """Fixture to provide a mocked OllamaConversationGenerator."""
    mock_generator = MagicMock(spec=OllamaConversationGenerator)
    
    # Configure the mock response for generate_conversation
    mock_generator.generate_conversation.return_value = SAMPLE_CONVERSATIONS[0]
    
    # Configure chunking
    mock_generator.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
    
    # Set the ollama interface
    mock_generator.ollama_interface = mock_ollama_interface
    
    return mock_generator

@pytest.fixture
def mock_dataset_expander(mock_ollama_interface):
    """Fixture to provide a mocked DatasetExpander."""
    mock_expander = MagicMock(spec=DatasetExpander)
    
    # Configure the mock response for expand_conversation_dataset
    expanded_conversation = [
        {"from": "human", "value": "What's an attention mechanism?"},
        {"from": "gpt", "value": "An attention mechanism allows models to focus on relevant parts of input data."}
    ]
    mock_expander.expand_conversation_dataset.return_value = [expanded_conversation]
    
    # Configure the mock response for generate_conversations_from_paper
    mock_expander.generate_conversations_from_paper.return_value = (
        SAMPLE_CONVERSATIONS, 
        [expanded_conversation, expanded_conversation]
    )
    
    # Configure the mock paraphrase_text
    mock_expander.paraphrase_text.return_value = "This is a paraphrased text."
    
    # Set the ollama interface
    mock_expander.ollama_interface = mock_ollama_interface
    
    return mock_expander

@pytest.fixture
def mock_dataset_cleaner(mock_ollama_interface):
    """Fixture to provide a mocked DatasetCleaner."""
    mock_cleaner = MagicMock(spec=DatasetCleaner)
    
    # Configure the mock response for clean_dataset
    cleaned_conversation = [
        {"from": "human", "value": "What's an attention mechanism?"},
        {"from": "gpt", "value": "An attention mechanism allows models to focus on relevant parts of input data."}
    ]
    mock_cleaner.clean_dataset.return_value = [cleaned_conversation]
    
    # Configure the mock response for analyze_dataset
    mock_cleaner.analyze_dataset.return_value = {
        "total_original": 2,
        "total_expanded": 4,
        "issues_by_type": {"grammar": 2, "coherence": 1},
        "detailed_issues": [{"conversation_idx": 1, "issue_type": "grammar"}],
        "length_analysis": {"average_diff": 5, "max_diff": 10}
    }
    
    # Set the ollama interface
    mock_cleaner.ollama_interface = mock_ollama_interface
    
    return mock_cleaner

@pytest.fixture
def mock_arxiv_searcher():
    """Fixture to provide a mocked ArxivSearcher."""
    mock_arxiv = MagicMock(spec=ArxivSearcher)
    
    # Configure the fetch_paper_info mock
    mock_arxiv.fetch_paper_info = AsyncMock(return_value=ARXIV_PAPER_INFO)
    
    # Configure the format_paper_for_learning mock
    mock_arxiv.format_paper_for_learning = AsyncMock(return_value=SAMPLE_PAPER_CONTENT)
    
    return mock_arxiv

@pytest.fixture
def mock_ddg_searcher():
    """Fixture to provide a mocked DuckDuckGoSearcher."""
    mock_ddg = MagicMock(spec=DuckDuckGoSearcher)
    
    # Configure the text_search mock
    mock_ddg.text_search = AsyncMock(return_value="# Search Results\n- Result 1\n- Result 2")
    
    return mock_ddg

@pytest.fixture
def mock_github_crawler():
    """Fixture to provide a mocked GitHubCrawler."""
    mock_github = MagicMock(spec=GitHubCrawler)
    
    # Configure the get_repo_summary mock
    mock_github.get_repo_summary = AsyncMock(return_value="# Repo Summary\nStars: 100\nForks: 20")
    
    # Configure the query_repo_content mock
    mock_github.query_repo_content = AsyncMock(
        return_value="# Query Results\nFound 5 Python files related to attention mechanisms."
    )
    
    # Configure the clone_and_store_repo mock
    mock_github.clone_and_store_repo = AsyncMock(return_value="/path/to/repo.parquet")
    
    return mock_github

@pytest.fixture
def mock_web_crawler():
    """Fixture to provide a mocked WebCrawler."""
    # For WebCrawler's static methods, we patch them in the test itself
    # since they're not instance methods, but this fixture serves as a reminder
    
    # Example patching (to be used in tests):
    # with patch('agentChef.crawlers_module.WebCrawler.fetch_url_content', new_callable=AsyncMock) as mock_fetch:
    #     mock_fetch.return_value = SAMPLE_HTML_CONTENT
    
    pass

@pytest.fixture
def mock_research_manager(
    mock_ollama_interface,
    mock_conversation_generator,
    mock_dataset_expander,
    mock_dataset_cleaner,
    mock_arxiv_searcher,
    mock_ddg_searcher,
    mock_github_crawler,
    temp_data_dir
):
    """Fixture to provide a mocked ResearchManager with all dependencies."""
    manager = ResearchManager(data_dir=temp_data_dir, model_name="llama3")
    # Add configuration for the research state
    manager.research_state = {
        "topic": "Transformer neural networks",
        "arxiv_papers": [ARXIV_PAPER_INFO],
        "processed_papers": [{"paper_info": ARXIV_PAPER_INFO, "formatted_info": SAMPLE_PAPER_CONTENT}],
        "search_results": "# Search Results\n- Result 1\n- Result 2",
        "github_repos": [{"repo_url": "https://github.com/example/transformer", "summary": "# Repo Summary"}],
        "conversations": SAMPLE_CONVERSATIONS,
        "expanded_data": [],
        "cleaned_data": []
    }

    # Add methods to test interactions
    manager.research_topic = AsyncMock(return_value={"status": "success", "papers_found": 2})
    manager.generate_conversation_dataset = AsyncMock(return_value={"status": "success", "conversations": 10})
    manager.process_paper_files = AsyncMock(return_value={"status": "success", "papers_processed": 2})
    manager.cleanup = MagicMock(return_value=True)

    return manager

# Configure pytest
def pytest_configure(config):
    """Configure pytest."""
    # Register markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    
@pytest.fixture(autouse=True)
def disable_logging():
    """Disable logging during tests."""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)