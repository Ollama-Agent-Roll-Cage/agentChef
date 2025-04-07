import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import os
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from agentChef.dataset_cleaner import DatasetCleaner
from agentChef.ollama_interface import OllamaInterface

class TestDatasetCleaner(unittest.TestCase):
    
    def setUp(self):
        # Create a mock Ollama interface
        self.mock_ollama_interface = MagicMock(spec=OllamaInterface)
        
        # Set up the mock response for cleaning
        self.mock_clean_response = {
            "message": {
                "content": "This is a cleaned response."
            }
        }
        self.mock_ollama_interface.chat.return_value = self.mock_clean_response
        
        # Create the cleaner with our mock interface
        self.cleaner = DatasetCleaner(
            ollama_interface=self.mock_ollama_interface,
            output_dir=tempfile.mkdtemp()  # Use a temporary directory for testing
        )
        
        # Sample conversations for testing
        self.original_conversations = [
            [
                {"from": "human", "value": "What are attention mechanisms?"},
                {"from": "gpt", "value": "Attention mechanisms allow models to focus on specific parts of input."}
            ],
            [
                {"from": "human", "value": "Can you explain transformers?"},
                {"from": "gpt", "value": "Transformers are neural network architectures that use attention mechanisms."}
            ]
        ]
        
        # Sample expanded conversations with some issues
        self.expanded_conversations = [
            [
                {"from": "human", "value": "What are attention mechanisms?"},
                {"from": "gpt", "value": "Attention mechanisms allow models to focusing on specific parts of input."}
            ],
            [
                {"from": "human", "value": "Can you explaining transformers?"},
                {"from": "gpt", "value": "Transformers is neural network architectures that use attention mechanisms."}
            ]
        ]
    
    def tearDown(self):
        # Clean up the temporary directory
        if hasattr(self, 'cleaner') and hasattr(self.cleaner, 'output_dir'):
            import shutil
            try:
                shutil.rmtree(self.cleaner.output_dir)
            except:
                pass
    
    def test_analyze_dataset(self):
        """Test analyzing a dataset for quality issues."""
        # Mock DataFrame operations to avoid relying on actual implementations
        with patch('agentChef.dataset_cleaner.DatasetCleaner._convert_conversations_to_df') as mock_convert, \
             patch('agentChef.dataset_cleaner.DatasetCleaner._analyze_length_differences') as mock_analyze_length, \
             patch('agentChef.dataset_cleaner.DatasetCleaner._analyze_semantic_quality') as mock_analyze_semantic:
            
            # Configure the mocks
            mock_convert.return_value = pd.DataFrame({'from': ['human', 'gpt'], 'value': ['question', 'answer']})
            mock_analyze_length.return_value = {"average_diff": 5, "max_diff": 10}
            mock_analyze_semantic.return_value = {
                "issues_by_type": {"grammar": 2, "coherence": 1},
                "detailed_issues": [{"conversation_idx": 1, "issue_type": "grammar"}]
            }
            
            # Analyze the dataset
            analysis = self.cleaner.analyze_dataset(
                self.original_conversations,
                self.expanded_conversations
            )
            
            # Verify the analysis structure
            self.assertIn('total_original', analysis)
            self.assertIn('total_expanded', analysis)
            self.assertIn('issues_by_type', analysis)
            self.assertIn('detailed_issues', analysis)
            self.assertIn('length_analysis', analysis)
            
            # Check the counts
            self.assertEqual(analysis['total_original'], 2)
            self.assertEqual(analysis['total_expanded'], 2)
            
            # Verify the mocks were called
            mock_convert.assert_called()
            mock_analyze_length.assert_called_once()
            mock_analyze_semantic.assert_called_once()
    
    @patch('agentChef.pandas_query.OllamaLlamaIndexIntegration')
    def test_perform_advanced_analysis(self, mock_ollama_query):
        """Test performing advanced analysis with query integration."""
        # Mock the query engine
        mock_query_engine = mock_ollama_query.return_value
        mock_query_engine.query_dataframe_with_ollama.return_value = {
            "response": "Analysis result"
        }
        
        # Set the mock query engine
        self.cleaner.ollama_query = mock_query_engine
        
        # Create sample dataframes
        orig_df = pd.DataFrame({
            'from': ['human', 'gpt'],
            'value': ['What are attention mechanisms?', 'Attention mechanisms allow focusing.']
        })
        expanded_df = pd.DataFrame({
            'from': ['human', 'gpt'],
            'value': ['What are attention mechanisms?', 'Attention mechanisms allow models to focus.']
        })
        
        # Perform advanced analysis
        analysis = self.cleaner._perform_advanced_analysis(orig_df, expanded_df)
        
        # Verify the analysis structure
        self.assertIn('ollama_analysis', analysis)
        
        # Verify the query engine was called
        self.assertTrue(mock_query_engine.query_dataframe_with_ollama.called)
        
        # Test handling when no query engine is available
        self.cleaner.ollama_query = None
        self.cleaner.pandas_query = None
        
        analysis = self.cleaner._perform_advanced_analysis(orig_df, expanded_df)
        self.assertEqual(analysis, {})
    
    def test_clean_dataset(self):
        """Test cleaning a dataset."""
        # Mock the conversation cleaning
        with patch('agentChef.dataset_cleaner.DatasetCleaner._clean_conversation') as mock_clean:
            # Configure the mock to return cleaned conversations
            def side_effect(orig, expanded, criteria):
                # Simple cleaning: copy the conversation but fix grammar in responses
                result = []
                for turn in expanded:
                    if turn['from'] == 'gpt':
                        # Fix simple grammar issue in the mocked response
                        value = turn['value'].replace('focusing', 'focus').replace('is', 'are')
                        result.append({'from': turn['from'], 'value': value})
                    else:
                        result.append(turn.copy())
                return result
            
            mock_clean.side_effect = side_effect
            
            # Mock the analysis to indicate issues in both conversations
            with patch('agentChef.dataset_cleaner.DatasetCleaner.analyze_dataset') as mock_analyze:
                mock_analyze.return_value = {
                    "detailed_issues": [
                        {"conversation_idx": 0, "issue_type": "grammar"},
                        {"conversation_idx": 1, "issue_type": "grammar"}
                    ]
                }
                
                # Clean the dataset
                cleaned = self.cleaner.clean_dataset(
                    self.original_conversations,
                    self.expanded_conversations,
                    cleaning_criteria={
                        "fix_hallucinations": True,
                        "normalize_style": True,
                        "correct_grammar": True,
                        "ensure_coherence": True
                    }
                )
                
                # Verify the cleaning process
                self.assertEqual(len(cleaned), 2)  # Should have 2 conversations
                
                # Verify each conversation was cleaned
                self.assertEqual(mock_clean.call_count, 2)
                
                # Verify the cleaning criteria were passed correctly
                criteria = mock_clean.call_args[0][2]
                self.assertTrue(criteria["fix_hallucinations"])
                self.assertTrue(criteria["normalize_style"])
                self.assertTrue(criteria["correct_grammar"])
                self.assertTrue(criteria["ensure_coherence"])

    def test_clean_conversation(self):
        """Test cleaning a single conversation."""
        # Define original and expanded conversations with issues
        original_conv = [
            {"from": "human", "value": "What are attention mechanisms?"},
            {"from": "gpt", "value": "Attention mechanisms allow models to focus on specific parts of input."}
        ]
        
        expanded_conv = [
            {"from": "human", "value": "What are attention mechanisms?"},
            {"from": "gpt", "value": "Attention mechanisms allow models to focusing on specific parts of input."}
        ]
        
        # Mock the turn content cleaning
        with patch('agentChef.dataset_cleaner.DatasetCleaner._clean_turn_content') as mock_clean_turn, \
             patch('agentChef.dataset_cleaner.DatasetCleaner._needs_cleaning') as mock_needs_cleaning:
            
            # Configure the mocks
            mock_needs_cleaning.return_value = True
            mock_clean_turn.return_value = "Attention mechanisms allow models to focus on specific parts of input."
            
            # Set up cleaning criteria
            criteria = {
                "fix_hallucinations": True,
                "normalize_style": True,
                "correct_grammar": True,
                "ensure_coherence": True
            }
            
            # Clean the conversation
            cleaned = self.cleaner._clean_conversation(original_conv, expanded_conv, criteria)
            
            # Verify the structure of the cleaned conversation
            self.assertEqual(len(cleaned), 2)
            self.assertEqual(cleaned[0]['from'], 'human')
            self.assertEqual(cleaned[1]['from'], 'gpt')
            
            # Verify turn cleaning was called for the gpt turn
            mock_clean_turn.assert_called_once()
            
            # Test handling turns that don't need cleaning
            mock_needs_cleaning.return_value = False
            mock_clean_turn.reset_mock()
            
            cleaned = self.cleaner._clean_conversation(original_conv, expanded_conv, criteria)
            
            # Verify turn cleaning was not called
            mock_clean_turn.assert_not_called()
    
    def test_needs_cleaning(self):
        """Test determining if a turn needs cleaning."""
        # Test obvious grammar issues
        expanded_turn = {"from": "gpt", "value": "They is going to the store."}
        original_turn = {"from": "gpt", "value": "They are going to the store."}
        
        criteria = {"correct_grammar": True}
        
        self.assertTrue(self.cleaner._needs_cleaning(expanded_turn, original_turn, criteria))
        
        # Test coherence issues (length difference)
        expanded_turn = {"from": "gpt", "value": "Yes."}
        original_turn = {"from": "gpt", "value": "Yes, that's correct. Transformers are a type of neural network architecture."}
        
        criteria = {"ensure_coherence": True}
        
        self.assertTrue(self.cleaner._needs_cleaning(expanded_turn, original_turn, criteria))
        
        # Test case where cleaning is not needed
        expanded_turn = {"from": "gpt", "value": "This is a good answer."}
        original_turn = {"from": "gpt", "value": "This is also a good answer."}
        
        criteria = {"correct_grammar": True, "ensure_coherence": True}
        
        self.assertFalse(self.cleaner._needs_cleaning(expanded_turn, original_turn, criteria))
        
        # Test with no criteria
        self.assertFalse(self.cleaner._needs_cleaning(expanded_turn, original_turn, {}))
    
    def test_clean_turn_content(self):
        """Test cleaning the content of a conversation turn."""
        # Mock the Ollama interface for chat
        self.mock_ollama_interface.chat.return_value = {
            "message": {"content": "Attention mechanisms allow models to focus on specific parts of input."}
        }
        
        # Test cleaning a turn with a grammar issue
        original_content = "Attention mechanisms allow models to focus on specific parts of input."
        expanded_content = "Attention mechanisms allow models to focusing on specific parts of input."
        
        cleaned = self.cleaner._clean_turn_content(
            original_content,
            expanded_content,
            "gpt",
            1,
            {"correct_grammar": True}
        )
        
        # Verify the Ollama interface was called correctly
        self.mock_ollama_interface.chat.assert_called_once()
        
        # Check that the system prompt contains cleaning instructions
        system_prompt = self.mock_ollama_interface.chat.call_args[1]['messages'][0]['content']
        self.assertIn("cleaning", system_prompt.lower())
        
        # Check that the user prompt contains both original and expanded content
        user_prompt = self.mock_ollama_interface.chat.call_args[1]['messages'][1]['content']
        self.assertIn(original_content, user_prompt)
        self.assertIn(expanded_content, user_prompt)
        
        # Verify the result
        self.assertEqual(cleaned, "Attention mechanisms allow models to focus on specific parts of input.")
        
        # Test handling of API errors
        self.mock_ollama_interface.chat.side_effect = Exception("API error")
        
        # Should return the original expanded content on error
        result = self.cleaner._clean_turn_content(
            original_content,
            expanded_content,
            "gpt",
            1,
            {"correct_grammar": True}
        )
        
        self.assertEqual(result, expanded_content)
    
    def test_convert_conversations_to_df(self):
        """Test converting conversations to DataFrame."""
        # Test basic conversion
        df = self.cleaner._convert_conversations_to_df(self.original_conversations)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)  # 2 conversations × 2 turns
        
        # Verify columns
        expected_columns = ['conversation_idx', 'turn_idx', 'from', 'value', 'content_length']
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Verify values
        self.assertEqual(df.iloc[0]['from'], 'human')
        self.assertEqual(df.iloc[0]['value'], 'What are attention mechanisms?')
        
        # Test with empty conversations
        empty_df = self.cleaner._convert_conversations_to_df([])
        self.assertIsInstance(empty_df, pd.DataFrame)
        self.assertEqual(len(empty_df), 0)
    
    def test_analyze_length_differences(self):
        """Test analyzing length differences between original and expanded conversations."""
        # Create sample dataframes with different content lengths
        orig_df = pd.DataFrame({
            'conversation_idx': [0, 0, 1, 1],
            'turn_idx': [0, 1, 0, 1],
            'from': ['human', 'gpt', 'human', 'gpt'],
            'value': ['Q1', 'A1 is long', 'Q2', 'A2 is very long'],
            'content_length': [2, 10, 2, 14]
        })
        
        expanded_df = pd.DataFrame({
            'conversation_idx': [0, 0, 1, 1],
            'turn_idx': [0, 1, 0, 1],
            'from': ['human', 'gpt', 'human', 'gpt'],
            'value': ['Q1', 'A1', 'Q2', 'A2'],
            'content_length': [2, 2, 2, 2]
        })
        
        # Analyze length differences
        analysis = self.cleaner._analyze_length_differences(orig_df, expanded_df)
        
        # Verify analysis structure
        self.assertIn('average_diff', analysis)
        self.assertIn('max_diff', analysis)
        self.assertIn('by_turn_type', analysis)
        
        # Check calculations
        # Original: [2, 10, 2, 14] - Expanded: [2, 2, 2, 2] = Diffs: [0, 8, 0, 12]
        self.assertEqual(analysis['average_diff'], 5.0)  # (0 + 8 + 0 + 12) / 4
        self.assertEqual(analysis['max_diff'], 12)  # max([0, 8, 0, 12])
        
        # Check turn type analysis
        self.assertIn('human', analysis['by_turn_type'])
        self.assertIn('gpt', analysis['by_turn_type'])
        self.assertEqual(analysis['by_turn_type']['human']['avg_diff'], 0.0)  # No difference in human turns
        self.assertEqual(analysis['by_turn_type']['gpt']['avg_diff'], 10.0)  # (8 + 12) / 2
    
    def test_analyze_semantic_quality(self):
        """Test analyzing semantic quality between original and expanded conversations."""
        # Mock the LLM call for semantic analysis
        with patch('agentChef.dataset_cleaner.DatasetCleaner._check_semantic_quality') as mock_check:
            # Configure the mock to return some quality issues
            mock_check.side_effect = [
                {"has_issues": False},
                {"has_issues": True, "issue_type": "grammar", "details": "Grammar error detected"}
            ]
            
            # Analyze semantic quality
            analysis = self.cleaner._analyze_semantic_quality(
                self.original_conversations,
                self.expanded_conversations
            )
            
            # Verify analysis structure
            self.assertIn('issues_by_type', analysis)
            self.assertIn('detailed_issues', analysis)
            
            # Check issue counts
            self.assertEqual(analysis['issues_by_type'].get('grammar', 0), 1)
            
            # Check detailed issues
            self.assertEqual(len(analysis['detailed_issues']), 1)
            self.assertEqual(analysis['detailed_issues'][0]['issue_type'], 'grammar')
            
            # Verify the semantic check was called for each conversation
            self.assertEqual(mock_check.call_count, 2)
    
    def test_check_semantic_quality(self):
        """Test checking semantic quality of a single conversation pair."""
        # Mock the Ollama interface for chat
        self.mock_ollama_interface.chat.return_value = {
            "message": {"content": '{"has_issues": true, "issue_type": "grammar", "details": "Grammar error found"}'}
        }
        
        # Check semantic quality
        result = self.cleaner._check_semantic_quality(
            self.original_conversations[0],
            self.expanded_conversations[0]
        )
        
        # Verify the Ollama interface was called correctly
        self.mock_ollama_interface.chat.assert_called_once()
        
        # Check that the system prompt contains analysis instructions
        system_prompt = self.mock_ollama_interface.chat.call_args[1]['messages'][0]['content']
        self.assertIn("quality analysis", system_prompt.lower())
        
        # Verify the result structure
        self.assertTrue(result['has_issues'])
        self.assertEqual(result['issue_type'], 'grammar')
        self.assertEqual(result['details'], 'Grammar error found')
        
        # Test handling of malformed responses
        self.mock_ollama_interface.chat.return_value = {
            "message": {"content": "This is not valid JSON"}
        }
        
        # Should handle the error and return a default response
        result = self.cleaner._check_semantic_quality(
            self.original_conversations[0],
            self.expanded_conversations[0]
        )
        
        self.assertIn('has_issues', result)
        self.assertIn('error', result)

if __name__ == "__main__":
    unittest.main()