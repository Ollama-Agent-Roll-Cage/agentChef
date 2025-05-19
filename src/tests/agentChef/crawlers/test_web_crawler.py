import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp
import asyncio
import os
import json
from pathlib import Path
import pytest
import tempfile

from agentChef.crawlers.crawlers_module import WebCrawler

class TestWebCrawler(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        os.environ['DATA_DIR'] = self.test_data_dir
        
        # Create crawls directory
        Path(f"{self.test_data_dir}/crawls").mkdir(parents=True, exist_ok=True)
        
        # Sample HTML content for testing
        self.sample_html = """<!DOCTYPE html>
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
        
        # Sample PyPI HTML content
        self.sample_pypi_html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Test Package</title>
        </head>
        <body>
            <div class="sidebar">
                <div class="sidebar-section">
                    <h3>Meta</h3>
                    <p>Version: 1.0.0</p>
                    <p>License: MIT</p>
                </div>
            </div>
            <div class="project-description">
                <h1>Test Package</h1>
                <p>This is a test package description.</p>
                <pre><code class="python">import test_package</code></pre>
                <ul>
                    <li>Feature 1</li>
                    <li>Feature 2</li>
                </ul>
            </div>
        </body>
        </html>"""
    
    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_data_dir)
    
    @patch('aiohttp.ClientSession.get')
    @patch('agentChef.crawlers_module.ParquetStorage.save_to_parquet')  # Mock the parquet saving
    async def test_fetch_url_content(self, mock_save_parquet, mock_get):
        """Test fetching content from a URL."""
        # Configure the mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=self.sample_html)
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Configure mock for parquet saving
        mock_save_parquet.return_value = "dummy_path.parquet"
        
        # Fetch URL content
        url = "https://example.com"
        html = await WebCrawler.fetch_url_content(url)
        
        # Verify aiohttp session was used correctly
        mock_get.assert_called_once_with(url)
        mock_response.text.assert_called_once()
        
        # Verify the returned HTML
        self.assertEqual(html, self.sample_html)
        
        # Verify the save_to_parquet was called instead of checking for the file
        mock_save_parquet.assert_called_once()
    
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_url_content_error(self, mock_get):
        """Test error handling when fetching URL content."""
        # Configure the mock response for error
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Fetch URL content with error
        url = "https://example.com/not-found"
        html = await WebCrawler.fetch_url_content(url)
        
        # Verify aiohttp session was used correctly
        mock_get.assert_called_once_with(url)
        
        # Verify the result is None for non-200 status
        self.assertIsNone(html)
    
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_url_content_exception(self, mock_get):
        """Test exception handling when fetching URL content."""
        # Configure the mock to raise an exception
        mock_get.side_effect = Exception("Connection error")
        
        # Fetch URL content with exception
        url = "https://example.com"
        html = await WebCrawler.fetch_url_content(url)
        
        # Verify aiohttp session was used correctly
        mock_get.assert_called_once_with(url)
        
        # Verify the result is None for exceptions
        self.assertIsNone(html)
    
    async def test_extract_text_from_html(self):
        """Test extracting text from HTML content."""
        # Extract text from sample HTML
        text = await WebCrawler.extract_text_from_html(self.sample_html)
        
        # Verify the extracted text
        self.assertIn("Test Heading", text)
        self.assertIn("This is a test paragraph", text)
        self.assertIn("This is another paragraph with a link", text)
        
        # Verify script and style content was removed
        self.assertNotIn("font-family", text)
        self.assertNotIn("console.log", text)
        
        # Test with None input
        result = await WebCrawler.extract_text_from_html(None)
        self.assertEqual(result, "Failed to extract text from the webpage.")
        
        # Test with empty input
        result = await WebCrawler.extract_text_from_html("")
        self.assertEqual(result, "Failed to extract text from the webpage.")
    
    async def test_extract_text_from_html_with_beautifulsoup_error(self):
        """Test fallback extraction when BeautifulSoup fails."""
        with patch('bs4.BeautifulSoup', side_effect=Exception("BeautifulSoup error")):
            # Extract text with BeautifulSoup error
            text = await WebCrawler.extract_text_from_html(self.sample_html)
            
            # Verify text was extracted using regex fallback
            self.assertIn("Test Heading", text)
            self.assertIn("This is a test paragraph", text)
    
    async def test_extract_pypi_content(self):
        """Test extracting PyPI package content."""
        # Extract PyPI content
        package_name = "test-package"
        package_data = await WebCrawler.extract_pypi_content(self.sample_pypi_html, package_name)
        
        # Verify the extracted data structure
        self.assertEqual(package_data['name'], package_name)
        self.assertIn('metadata', package_data)
        self.assertIn('documentation', package_data)
        
        # Check metadata
        self.assertIn('Meta', package_data['metadata'])
        self.assertEqual(package_data['metadata']['Meta'][0], "Version: 1.0.0")
        self.assertEqual(package_data['metadata']['Meta'][1], "License: MIT")
        
        # Check documentation
        self.assertIn("# Test Package", package_data['documentation'])
        self.assertIn("This is a test package description", package_data['documentation'])
        self.assertIn("```python", package_data['documentation'])
        self.assertIn("- Feature 1", package_data['documentation'])
    
    async def test_extract_pypi_content_missing_elements(self):
        """Test extracting PyPI content with missing elements."""
        # Create HTML with missing elements
        html_missing_description = """<!DOCTYPE html>
        <html>
        <head>
            <title>Test Package</title>
        </head>
        <body>
            <div class="sidebar">
                <div class="sidebar-section">
                    <h3>Meta</h3>
                    <p>Version: 1.0.0</p>
                </div>
            </div>
        </body>
        </html>"""
        
        # Extract PyPI content with missing description
        package_name = "test-package"
        package_data = await WebCrawler.extract_pypi_content(html_missing_description, package_name)
        
        # Verify that None is returned when description is missing
        self.assertIsNone(package_data)
    
    async def test_extract_pypi_content_with_exception(self):
        """Test exception handling when extracting PyPI content."""
        # Patch where the BeautifulSoup is actually being used in the crawlers_module
        with patch('agentChef.crawlers_module.BeautifulSoup', side_effect=Exception("BeautifulSoup error")):
            # Extract PyPI content with exception
            package_name = "test-package"
            package_data = await WebCrawler.extract_pypi_content(self.sample_pypi_html, package_name)
            
            # Verify that None is returned on exception
            self.assertIsNone(package_data)
    
    async def test_format_pypi_info(self):
        """Test formatting PyPI package information."""
        # Sample PyPI API data
        pypi_data = {
            "info": {
                "name": "test-package",
                "version": "1.0.0",
                "summary": "A test package for testing",
                "description": "This is a detailed description of the test package.",
                "author": "Test Author",
                "author_email": "test@example.com",
                "home_page": "https://example.com/test-package",
                "license": "MIT",
                "project_urls": {
                    "Documentation": "https://docs.example.com/test-package",
                    "Source": "https://github.com/example/test-package"
                },
                "requires_dist": ["requests>=2.0.0", "beautifulsoup4"]
            }
        }
        
        # Format PyPI info
        formatted = await WebCrawler.format_pypi_info(pypi_data)
        
        # Verify the formatted markdown
        self.assertIn("# test-package v1.0.0", formatted)
        self.assertIn("## Summary", formatted)
        self.assertIn("A test package for testing", formatted)
        self.assertIn("**Author**: Test Author (test@example.com)", formatted)
        self.assertIn("**License**: MIT", formatted)
        self.assertIn("**Documentation**: https://docs.example.com/test-package", formatted)
        self.assertIn("- requests>=2.0.0", formatted)
        self.assertIn("pip install test-package", formatted)
        self.assertIn("## Description", formatted)
        
        # Test with None input
        result = await WebCrawler.format_pypi_info(None)
        self.assertEqual(result, "Could not retrieve package information.")
        
        # Test with missing information
        minimal_data = {"info": {"name": "minimal"}}
        minimal_result = await WebCrawler.format_pypi_info(minimal_data)
        self.assertIn("# minimal v", minimal_result)
        self.assertIn("No summary available", minimal_result)
        self.assertIn("No dependencies listed", minimal_result)
    
    async def test_format_pypi_info_with_long_description(self):
        """Test formatting PyPI info with a long description."""
        # Create PyPI data with a long description
        long_description = "x" * 2000
        pypi_data = {
            "info": {
                "name": "test-package",
                "version": "1.0.0",
                "summary": "A test package",
                "description": long_description,
                "author": "Test Author",
                "author_email": "test@example.com"
            }
        }
        
        # Format PyPI info with long description
        formatted = await WebCrawler.format_pypi_info(pypi_data)
        
        # Verify the description was truncated
        self.assertIn("## Description Preview", formatted)
        self.assertIn("(Description truncated for brevity)", formatted)
        
        # Verify the truncated description is reasonably sized
        description_start = formatted.find("## Description Preview\n") + len("## Description Preview\n")
        description_end = formatted.find("(Description truncated for brevity)")
        truncated_desc = formatted[description_start:description_end].strip()
        self.assertTrue(len(truncated_desc) < 2000)

if __name__ == "__main__":
    import unittest.async_case
    unittest.main()