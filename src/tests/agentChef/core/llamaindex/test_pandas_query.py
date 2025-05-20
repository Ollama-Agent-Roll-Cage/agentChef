import unittest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
import pytest
import os

# Import modules with conditional imports to handle potential ImportError
try:
    from src.agentChef.llamaindex.pandas_query import PandasQueryIntegration, OllamaLlamaIndexIntegration
    HAS_QUERY_INTEGRATION = True
except ImportError:
    HAS_QUERY_INTEGRATION = False
    
# Skip tests if integration is not available
pytestmark = pytest.mark.skipif(not HAS_QUERY_INTEGRATION, reason="LlamaIndex integration not available")

class TestPandasQueryIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test case with sample data."""
        # Skip test if LlamaIndex is not available
        if not HAS_QUERY_INTEGRATION:
            self.skipTest("LlamaIndex not available")
        
        # Create a sample dataframe for testing
        self.df = pd.DataFrame({
            "city": ["Toronto", "Tokyo", "Berlin", "Sydney", "New York"],
            "population": [2930000, 13960000, 3645000, 5312000, 8419000],
            "country": ["Canada", "Japan", "Germany", "Australia", "USA"],
            "continent": ["North America", "Asia", "Europe", "Oceania", "North America"]
        })
        
        # Mock the PandasQueryEngine and its dependencies
        self.query_engine_patcher = patch('llama_index.experimental.query_engine.PandasQueryEngine', create=True)
        self.mock_query_engine_class = self.query_engine_patcher.start()
        self.mock_query_engine = self.mock_query_engine_class.return_value
        
        # Configure the mock query engine
        self.mock_query_engine.query.return_value = MagicMock(
            __str__=lambda self: "Tokyo has the highest population.",
            metadata={"pandas_instruction_str": "df[df['population'].idxmax()]['city']"}
        )
        
        # Initialize the PandasQueryIntegration
        self.pandas_query = PandasQueryIntegration(verbose=True)
    
    def tearDown(self):
        """Clean up after test case."""
        # Stop the patcher
        self.query_engine_patcher.stop()
    
    def test_create_query_engine(self):
        """Test creating a PandasQueryEngine."""
        # Create a query engine
        query_engine = self.pandas_query.create_query_engine(self.df)
        
        # Verify PandasQueryEngine was initialized correctly
        self.mock_query_engine_class.assert_called_once()
        self.assertEqual(query_engine, self.mock_query_engine)
        
        # Test with custom instructions
        custom_instructions = "Focus on population statistics and use median for averages."
        query_engine = self.pandas_query.create_query_engine(self.df, custom_instructions)
        
        # Verify update_prompts was called
        self.mock_query_engine.update_prompts.assert_called_once()
    
    def test_query_dataframe(self):
        """Test querying a DataFrame."""
        # Query the dataframe
        result = self.pandas_query.query_dataframe(
            self.df, 
            "What is the city with the highest population?"
        )
        
        # Verify query was called
        self.mock_query_engine.query.assert_called_once_with("What is the city with the highest population?")
        
        # Verify result structure
        self.assertIn("response", result)
        self.assertIn("pandas_instructions", result)
        self.assertIn("raw_response", result)
        
        # Verify values
        self.assertEqual(result["response"], "Tokyo has the highest population.")
        self.assertEqual(result["pandas_instructions"], "df[df['population'].idxmax()]['city']")
    
    def test_query_dataframe_error(self):
        """Test error handling in query_dataframe."""
        # Configure the mock to raise an exception
        self.mock_query_engine.query.side_effect = Exception("Query error")
        
        # Query the dataframe with error
        result = self.pandas_query.query_dataframe(
            self.df, 
            "What is the city with the highest population?"
        )
        
        # Verify error handling
        self.assertIn("error", result)
        self.assertIn("response", result)
        self.assertIn("Error querying DataFrame", result["response"])
    
    def test_generate_dataset_insights(self):
        """Test generating insights from a DataFrame."""
        # Configure the mock for multiple calls
        self.mock_query_engine.query.side_effect = [
            MagicMock(
                __str__=lambda self: "The dataframe has 5 rows and 4 columns.",
                metadata={"pandas_instruction_str": "df.shape"}
            ),
            MagicMock(
                __str__=lambda self: "There are no missing values in the dataframe.",
                metadata={"pandas_instruction_str": "df.isnull().sum()"}
            ),
            MagicMock(
                __str__=lambda self: "Tokyo has the highest population with 13,960,000 people.",
                metadata={"pandas_instruction_str": "df[df['population'].idxmax()]"}
            )
        ]
        
        # Generate insights
        insights = self.pandas_query.generate_dataset_insights(self.df, num_insights=3)
        
        # Verify query was called multiple times
        self.assertEqual(self.mock_query_engine.query.call_count, 3)
        
        # Verify insights structure
        self.assertEqual(len(insights), 3)
        self.assertIn("query", insights[0])
        self.assertIn("insight", insights[0])
        self.assertIn("pandas_code", insights[0])
    
    def test_compare_datasets(self):
        """Test comparing two DataFrames."""
        # Create a second dataframe with some differences
        df2 = pd.DataFrame({
            "city": ["Toronto", "Tokyo", "Berlin", "Sydney", "New York", "Paris"],
            "population": [2930000, 13960000, 3645000, 5312000, 8419000, 2161000],
            "country": ["Canada", "Japan", "Germany", "Australia", "USA", "France"],
            "continent": ["North America", "Asia", "Europe", "Oceania", "North America", "Europe"]
        })
        
        # Configure the mock for multiple calls
        self.mock_query_engine.query.side_effect = [
            MagicMock(
                __str__=lambda self: "Original has 5 rows, Modified has 6 rows.",
                metadata={"pandas_instruction_str": "print(f'Original: {sum(_dataset==\"Original\")} rows, Modified: {sum(_dataset==\"Modified\")} rows')"}
            ),
            MagicMock(
                __str__=lambda self: "The Modified dataset has one additional city: Paris.",
                metadata={"pandas_instruction_str": "df[df['_dataset'] == 'Modified']['city'].value_counts()"}
            ),
            MagicMock(
                __str__=lambda self: "Summary of differences between datasets.",
                metadata={"pandas_instruction_str": "# Summary code"}
            )
        ]
        
        # Compare datasets
        comparison = self.pandas_query.compare_datasets(
            self.df, df2, 
            df1_name="Original", df2_name="Modified",
            aspects=["shape", "schema", "statistics"]
        )
        
        # Verify query was called multiple times
        self.assertTrue(self.mock_query_engine.query.call_count >= 3)
        
        # Verify comparison structure
        self.assertIn("comparison_details", comparison)
        self.assertIn("overall_summary", comparison)
        self.assertIn("common_columns", comparison)
        self.assertIn("unique_columns_df1", comparison)
        self.assertIn("unique_columns_df2", comparison)
        
        # Verify common columns
        self.assertEqual(set(comparison["common_columns"]), {"city", "population", "country", "continent"})

class TestOllamaLlamaIndexIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test case with sample data."""
        # Skip test if LlamaIndex is not available
        if not HAS_QUERY_INTEGRATION:
            self.skipTest("LlamaIndex not available")
        
        # Create a sample dataframe for testing
        self.df = pd.DataFrame({
            "city": ["Toronto", "Tokyo", "Berlin", "Sydney", "New York"],
            "population": [2930000, 13960000, 3645000, 5312000, 8419000],
            "country": ["Canada", "Japan", "Germany", "Australia", "USA"],
            "continent": ["North America", "Asia", "Europe", "Oceania", "North America"]
        })
        
        # Mock Ollama
        self.ollama_patcher = patch('agentChef.pandas_query.OllamaLlamaIndexIntegration.ollama', create=True)
        self.mock_ollama = self.ollama_patcher.start()
        
        # Configure the mock Ollama chat response
        self.mock_ollama.chat.return_value = {
            'message': {
                'content': '```python\ndf[df[\'population\'].idxmax()][\'city\']\n```'
            }
        }
        
        # Initialize the OllamaLlamaIndexIntegration
        self.ollama_query = OllamaLlamaIndexIntegration(ollama_model="llama3")
    
    def tearDown(self):
        """Clean up after test case."""
        # Stop the patcher
        self.ollama_patcher.stop()
    
    def test_query_dataframe_with_ollama(self):
        """Test querying a DataFrame with Ollama."""
        # Query the dataframe
        result = self.ollama_query.query_dataframe_with_ollama(
            self.df, 
            "What is the city with the highest population?"
        )
        
        # Verify Ollama chat was called
        self.mock_ollama.chat.assert_called_once()
        
        # Verify the system and user prompt structure
        messages = self.mock_ollama.chat.call_args[1]['messages']
        system_prompt = messages[0]['content']
        user_prompt = messages[1]['content']
        
        # Check system prompt contains expected instructions
        self.assertIn("data analysis assistant", system_prompt.lower())
        
        # Check user prompt contains dataframe info and query
        self.assertIn("DataFrame Information", user_prompt)
        self.assertIn("Query: What is the city with the highest population?", user_prompt)
        
        # Verify result structure
        self.assertIn("response", result)
        self.assertIn("pandas_code", result)
        self.assertEqual(result["pandas_code"], "df[df['population'].idxmax()]['city']")
    
    def test_query_dataframe_with_markdown_code_blocks(self):
        """Test handling of markdown code blocks in Ollama responses."""
        # Configure the mock Ollama chat response with markdown code blocks
        self.mock_ollama.chat.return_value = {
            'message': {
                'content': '```python\ndf[df[\'population\'].idxmax()][\'city\']\n```\n\nThis would give you Tokyo.'
            }
        }
        
        # Query the dataframe
        result = self.ollama_query.query_dataframe_with_ollama(
            self.df, 
            "What is the city with the highest population?"
        )
        
        # Verify code extraction from markdown blocks
        self.assertEqual(result["pandas_code"], "df[df['population'].idxmax()]['city']")
    
    @patch('builtins.eval')
    def test_query_dataframe_code_execution(self, mock_eval):
        """Test code execution in query_dataframe_with_ollama."""
        # Configure the mock eval to return "Tokyo"
        mock_eval.return_value = "Tokyo"
        
        # Query the dataframe
        result = self.ollama_query.query_dataframe_with_ollama(
            self.df, 
            "What is the city with the highest population?"
        )
        
        # Verify eval was called with the correct code and namespace
        mock_eval.assert_called_once()
        
        # The first argument should be the code to evaluate
        code_arg = mock_eval.call_args[0][0]
        self.assertIn("df[df['population'].idxmax()]['city']", code_arg)
        
        # The second argument should be a dict with 'df' and 'pd'
        namespace_arg = mock_eval.call_args[0][1]
        self.assertIn('df', namespace_arg)
        self.assertIn('pd', namespace_arg)
        
        # Verify result
        self.assertEqual(result["response"], "Tokyo")
    
    def test_query_dataframe_execution_error(self):
        """Test error handling in code execution."""
        # Configure the mock Ollama chat response with invalid code
        self.mock_ollama.chat.return_value = {
            'message': {
                'content': 'df.invalid_function()'
            }
        }
        
        # Query the dataframe with code that will raise an exception
        result = self.ollama_query.query_dataframe_with_ollama(
            self.df, 
            "Call an invalid function"
        )
        
        # Verify error handling
        self.assertIn("error", result)
        self.assertIn("Error executing pandas code", result["error"])
        self.assertIn("pandas_code", result)
        self.assertEqual(result["pandas_code"], "df.invalid_function()")
    
    def test_query_dataframe_ollama_error(self):
        """Test error handling when Ollama returns an error."""
        # Configure the mock to raise an exception
        self.mock_ollama.chat.side_effect = Exception("Ollama API error")
        
        # Query the dataframe
        result = self.ollama_query.query_dataframe_with_ollama(
            self.df, 
            "What is the city with the highest population?"
        )
        
        # Verify error handling
        self.assertIn("error", result)
        self.assertIn("Error querying DataFrame with Ollama", result["error"])

if __name__ == "__main__":
    unittest.main()