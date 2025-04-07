import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import os
import json
import tempfile
from pathlib import Path
import pytest

from agentChef.udrags import ResearchManager, OllamaInterface
from agentChef.conversation_generator import OllamaConversationGenerator
from agentChef.dataset_expander import DatasetExpander
from agentChef.dataset_cleaner import DatasetCleaner
from agentChef.crawlers_module import WebCrawler, ArxivSearcher, DuckDuckGoSearcher, GitHubCrawler

class TestResearchManager(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        
        # Mock the Ollama interface
        self.mock_ollama_interface = MagicMock(spec=OllamaInterface)
        
        # Configure the mock response for chat
        self.mock_ollama_interface.chat.return_value = {
            "message": {
                "content": "This is a test response."
            }
        }
        
        # Create the research manager with our mock interface
        self.manager = ResearchManager(data_dir=self.test_data_dir, model_name="llama3")
        
        # Replace the Ollama interface with our mock
        self.manager.ollama_interface = self.mock_ollama_interface
        
        # Sample paper content
        self.sample_paper_content = """
        Attention mechanisms have become an integral part of compelling sequence modeling
        and transduction models in various tasks, allowing modeling of dependencies without
        regard to their distance in the input or output sequences. In this paper we present the
        Transformer, a model architecture eschewing recurrence and instead relying entirely
        on an attention mechanism to draw global dependencies between input and output.
        """
        
        # Create a mock for ArxivSearcher
        self.mock_arxiv = MagicMock(spec=ArxivSearcher)
        self.mock_arxiv.fetch_paper_info = AsyncMock()
        self.mock_arxiv.format_paper_for_learning = AsyncMock(return_value=self.sample_paper_content)
        self.manager.arxiv_searcher = self.mock_arxiv
        
        # Create a mock for DuckDuckGoSearcher
        self.mock_ddg = MagicMock(spec=DuckDuckGoSearcher)
        self.mock_ddg.text_search = AsyncMock(return_value="# Search Results\n- Result 1\n- Result 2")
        self.manager.ddg_searcher = self.mock_ddg
        
        # Create a mock for GitHubCrawler
        self.mock_github = MagicMock(spec=GitHubCrawler)
        self.mock_github.get_repo_summary = AsyncMock(return_value="# Repo Summary\nStars: 100\nForks: 20")
        self.manager.github_crawler = self.mock_github
        
        # Create a mock for OllamaConversationGenerator
        self.mock_generator = MagicMock(spec=OllamaConversationGenerator)
        mock_conversation = [
            {"from": "human", "value": "What is a transformer model?"},
            {"from": "gpt", "value": "A transformer is a deep learning model that adopts the mechanism of attention."}
        ]
        self.mock_generator.generate_conversation.return_value = mock_conversation
        self.mock_generator.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        self.manager.conversation_generator = self.mock_generator
        
        # Create a mock for DatasetExpander
        self.mock_expander = MagicMock(spec=DatasetExpander)
        expanded_conversation = [
            {"from": "human", "value": "What's a transformer architecture?"},
            {"from": "gpt", "value": "The transformer architecture utilizes attention mechanisms for processing sequences."}
        ]
        self.mock_expander.expand_conversation_dataset.return_value = [expanded_conversation]
        self.mock_expander.save_conversations_to_jsonl.return_value = f"{self.test_data_dir}/expanded.jsonl"
        self.manager.dataset_expander = self.mock_expander
        
        # Create a mock for DatasetCleaner
        self.mock_cleaner = MagicMock(spec=DatasetCleaner)
        cleaned_conversation = [
            {"from": "human", "value": "What's a transformer architecture?"},
            {"from": "gpt", "value": "The transformer architecture utilizes attention mechanisms for processing sequential data."}
        ]
        self.mock_cleaner.clean_dataset.return_value = [cleaned_conversation]
        self.manager.dataset_cleaner = self.mock_cleaner

    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_data_dir)
    
    async def test_research_topic(self):
        """Test researching a topic."""
        # Mock ArXiv paper information
        mock_paper_info = {
            'arxiv_id': '1706.03762',
            'title': 'Attention Is All You Need',
            'authors': ['Ashish Vaswani', 'Noam Shazeer'],
            'abstract': 'We propose a new simple network architecture, the Transformer...',
            'published': '2017-06-12',
            'pdf_link': 'https://arxiv.org/pdf/1706.03762.pdf',
            'arxiv_url': 'https://arxiv.org/abs/1706.03762',
            'categories': ['cs.CL', 'cs.LG']
        }
        self.mock_arxiv.fetch_paper_info.return_value = mock_paper_info
        
        # Mock the arxiv query generation
        with patch('agentChef.udrags.ollama.chat') as mock_ollama_chat:
            mock_ollama_chat.return_value = {
                'message': {'content': '1. "transformer neural networks"\n2. "attention mechanism"\n3. "self-attention models"'}
            }
            
            # Research a topic
            topic = "Transformer neural networks"
            results = await self.manager.research_topic(
                topic=topic,
                max_papers=3,
                max_search_results=5,
                include_github=True,
                github_repos=["https://github.com/example/transformer"],
                callback=lambda msg: None  # Dummy callback
            )
            
            # Verify ArxivSearcher was called
            self.mock_arxiv.fetch_paper_info.assert_called()
            self.mock_arxiv.format_paper_for_learning.assert_called_with(mock_paper_info)
            
            # Verify DuckDuckGoSearcher was called
            self.mock_ddg.text_search.assert_called_with(topic, max_results=5)
            
            # Verify GitHubCrawler was called for the provided repo
            self.mock_github.get_repo_summary.assert_called_with("https://github.com/example/transformer")
            
            # Verify the structure of the research results
            self.assertEqual(results["topic"], topic)
            self.assertIn("arxiv_papers", results)
            self.assertIn("search_results", results)
            self.assertIn("github_repos", results)
            self.assertIn("processed_papers", results)
            self.assertIn("summary", results)
    
    async def test_generate_conversation_dataset(self):
        """Test generating a conversation dataset."""
        # Mock the papers from research
        self.manager.research_state = {
            "processed_papers": [
                {
                    "paper_info": {
                        "title": "Attention Is All You Need",
                        "authors": ["Ashish Vaswani", "Noam Shazeer"]
                    },
                    "formatted_info": self.sample_paper_content
                }
            ]
        }
        
        # Generate a conversation dataset
        results = await self.manager.generate_conversation_dataset(
            num_turns=3,
            expansion_factor=2,
            clean=True,
            callback=lambda msg: None  # Dummy callback
        )
        
        # Verify conversation generator was called
        self.mock_generator.chunk_text.assert_called_with(
            self.sample_paper_content, chunk_size=2000, overlap=200
        )
        self.mock_generator.generate_conversation.assert_called()
        
        # Verify dataset expander was called
        self.mock_expander.expand_conversation_dataset.assert_called_once()
        
        # Verify dataset cleaner was called
        self.mock_cleaner.clean_dataset.assert_called_once()
        
        # Verify the structure of the results
        self.assertIn("conversations", results)
        self.assertIn("expanded_conversations", results)
        self.assertIn("cleaned_conversations", results)
        self.assertIn("output_path", results)
    
    async def test_generate_conversation_dataset_no_papers(self):
        """Test generating a conversation dataset with no papers."""
        # Set empty research state
        self.manager.research_state = {
            "processed_papers": []
        }
        
        # Generate a conversation dataset with no papers
        results = await self.manager.generate_conversation_dataset(
            num_turns=3,
            expansion_factor=2,
            clean=True
        )
        
        # Verify error handling
        self.assertIn("error", results)
        self.assertEqual(results["conversations"], [])
        self.assertEqual(results["expanded_conversations"], [])
        self.assertEqual(results["cleaned_conversations"], [])
    
    async def test_process_paper_files(self):
        """Test processing paper files."""
        # Create temporary paper files
        paper1_path = os.path.join(self.test_data_dir, "paper1.txt")
        paper2_path = os.path.join(self.test_data_dir, "paper2.txt")
        
        with open(paper1_path, 'w') as f:
            f.write("Paper 1 content")
        with open(paper2_path, 'w') as f:
            f.write("Paper 2 content")
        
        # Mock the dataset expander's generate_conversations_from_paper method
        original_convs = [
            [
                {"from": "human", "value": "Question 1?"},
                {"from": "gpt", "value": "Answer 1."}
            ]
        ]
        expanded_convs = [
            [
                {"from": "human", "value": "Question 1 paraphrased?"},
                {"from": "gpt", "value": "Answer 1 paraphrased."}
            ]
        ]
        self.mock_expander.generate_conversations_from_paper.return_value = (original_convs, expanded_convs)
        
        # Configure the mock convert_to_multi_format
        output_files = {
            'jsonl': f"{self.test_data_dir}/output.jsonl",
            'parquet': f"{self.test_data_dir}/output.parquet",
            'csv': f"{self.test_data_dir}/output.csv",
            'df': MagicMock()  # Mock DataFrame
        }
        self.mock_expander.convert_to_multi_format.return_value = output_files
        
        # Process paper files
        paper_files = [paper1_path, paper2_path]
        results = await self.manager.process_paper_files(
            paper_files=paper_files,
            output_format='all',
            num_turns=3,
            expansion_factor=2,
            clean=True,
            callback=lambda msg: None  # Dummy callback
        )
        
        # Verify that generate_conversations_from_paper was called for each paper
        self.assertEqual(self.mock_expander.generate_conversations_from_paper.call_count, 2)
        
        # Verify convert_to_multi_format was called
        self.mock_expander.convert_to_multi_format.assert_called_once()
        
        # Verify the structure of the results
        self.assertIn("conversations_count", results)
        self.assertIn("output_paths", results)
        self.assertEqual(results["output_paths"], output_files)
    
    async def test_process_paper_files_error(self):
        """Test processing paper files with errors."""
        # Create an invalid paper path
        invalid_path = os.path.join(self.test_data_dir, "nonexistent.txt")
        
        # Process non-existent paper files
        results = await self.manager.process_paper_files(
            paper_files=[invalid_path],
            output_format='jsonl',
            num_turns=3,
            expansion_factor=2,
            clean=True
        )
        
        # Verify error handling
        self.assertIn("error", results)
        self.assertEqual(results["output_paths"], [])
    
    def test_cleanup(self):
        """Test cleanup method."""
        # Create a temporary directory to be cleaned up
        temp_dir = Path(tempfile.mkdtemp())
        self.manager.temp_dir = temp_dir
        
        # Create a file in the temporary directory
        test_file = temp_dir / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Verify the directory and file exist
        self.assertTrue(temp_dir.exists())
        self.assertTrue(test_file.exists())
        
        # Call cleanup
        self.manager.cleanup()
        
        # Verify the directory was removed
        self.assertFalse(temp_dir.exists())

class TestOllamaInterface(unittest.TestCase):
    """Test the simplified OllamaInterface from udrags.py."""
    
    @patch('agentChef.udrags.ollama')
    def test_chat(self, mock_ollama):
        """Test the chat method."""
        # Configure the mock response
        mock_response = {
            "message": {
                "content": "This is a test response."
            }
        }
        mock_ollama.chat.return_value = mock_response
        
        # Create interface and call chat
        interface = OllamaInterface(model_name="llama3")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        
        response = interface.chat(messages=messages)
        
        # Verify ollama.chat was called correctly
        mock_ollama.chat.assert_called_once_with(model="llama3", messages=messages)
        
        # Verify the response
        self.assertEqual(response, mock_response)

class TestEndToEnd(unittest.TestCase):
    """End-to-end test for the UDRAGS workflow."""
    
    @patch('agentChef.udrags.ResearchManager.research_topic')
    @patch('agentChef.udrags.ResearchManager.generate_conversation_dataset')
    async def test_udrags_workflow(self, mock_generate, mock_research):
        """Test the full UDRAGS workflow."""
        # Configure mock responses
        mock_research.return_value = {
            "topic": "Transformer neural networks",
            "arxiv_papers": [{"title": "Attention Is All You Need"}],
            "processed_papers": [{"formatted_info": "Paper content"}],
            "search_results": "Search results",
            "github_repos": [{"repo_url": "url", "summary": "summary"}],
            "summary": "Research summary"
        }
        
        mock_generate.return_value = {
            "conversations": [{"from": "human", "value": "Question?"}],
            "expanded_conversations": [{"from": "human", "value": "Question?"}],
            "cleaned_conversations": [{"from": "human", "value": "Question?"}],
            "output_path": "/path/to/output.jsonl"
        }
        
        # Create the research manager
        manager = ResearchManager(data_dir=tempfile.mkdtemp(), model_name="llama3")
        
        # Research a topic
        research_results = await manager.research_topic(
            topic="Transformer neural networks",
            max_papers=3,
            callback=lambda msg: None  # Dummy callback
        )
        
        # Generate a dataset from the research
        generate_results = await manager.generate_conversation_dataset(
            num_turns=3,
            expansion_factor=2,
            clean=True,
            callback=lambda msg: None  # Dummy callback
        )
        
        # Verify the research function was called
        mock_research.assert_called_once()
        
        # Verify the generate function was called
        mock_generate.assert_called_once()
        
        # Clean up
        manager.cleanup()

if __name__ == "__main__":
    unittest.main()