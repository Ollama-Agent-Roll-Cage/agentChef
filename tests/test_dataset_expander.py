import unittest
from unittest.mock import patch, MagicMock, mock_open, ANY
import json
import os
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from agentChef.dataset_expander import DatasetExpander
from agentChef.ollama_interface import OllamaInterface
from agentChef.conversation_generator import OllamaConversationGenerator

class TestDatasetExpander(unittest.TestCase):
    
    def setUp(self):
        # Create a mock Ollama interface
        self.mock_ollama_interface = MagicMock(spec=OllamaInterface)
        
        # Set up the mock response for paraphrase
        self.mock_paraphrase_response = {
            "message": {
                "content": "This is a paraphrased response."
            }
        }
        self.mock_ollama_interface.chat.return_value = self.mock_paraphrase_response
        
        # Create the expander with our mock interface
        self.expander = DatasetExpander(
            ollama_interface=self.mock_ollama_interface,
            output_dir=tempfile.mkdtemp()  # Use a temporary directory for testing
        )
        
        # Sample conversations for testing
        self.sample_conversations = [
            [
                {"from": "human", "value": "What are attention mechanisms?"},
                {"from": "gpt", "value": "Attention mechanisms allow models to focus on specific parts of input."}
            ],
            [
                {"from": "human", "value": "Can you explain transformers?"},
                {"from": "gpt", "value": "Transformers are neural network architectures that use attention mechanisms."}
            ]
        ]
    
    def tearDown(self):
        # Clean up the temporary directory
        if hasattr(self, 'expander') and hasattr(self.expander, 'output_dir'):
            import shutil
            try:
                shutil.rmtree(self.expander.output_dir)
            except:
                pass
    
    def test_expand_conversation_dataset(self):
        """Test expanding a conversation dataset."""
        # Set up a sequence of paraphrase responses
        paraphrase_responses = [
            {"message": {"content": "What are the attention mechanisms used for?"}},
            {"message": {"content": "Attention mechanisms help models focus on relevant parts of the input data."}},
            {"message": {"content": "Could you describe transformers?"}},
            {"message": {"content": "Transformers are neural networks that utilize attention mechanisms for processing data."}}
        ]
        
        # Mock verify_paraphrase and clean_generated_content to avoid additional API calls
        with patch.object(self.expander, 'verify_paraphrase', side_effect=lambda orig, para, ref, is_q: para), \
             patch.object(self.expander, 'clean_generated_content', side_effect=lambda text, is_q: text):
            
            # Configure the mock to return different responses in sequence
            self.mock_ollama_interface.chat.side_effect = paraphrase_responses
            
            # Expand the conversations
            expanded = self.expander.expand_conversation_dataset(
                conversations=self.sample_conversations,
                expansion_factor=1,
                static_fields={'human': False, 'gpt': False}
            )
            
            # Verify the ollama interface was called for each turn
            self.assertEqual(self.mock_ollama_interface.chat.call_count, 4)  # 2 conversations x 2 turns
        
        # Verify the expanded conversations structure
        self.assertEqual(len(expanded), 2)  # 2 original x 1 expansion factor
        self.assertEqual(len(expanded[0]), 2)  # 2 turns in first conversation
        self.assertEqual(len(expanded[1]), 2)  # 2 turns in second conversation
        
        # Check that the content was paraphrased
        self.assertEqual(expanded[0][0]["value"], "What are the attention mechanisms used for?")
        self.assertEqual(expanded[0][1]["value"], "Attention mechanisms help models focus on relevant parts of the input data.")
    
    def test_paraphrase_text(self):
        """Test paraphrasing text."""
        # Set up mock response
        self.mock_ollama_interface.chat.return_value = {
            "message": {"content": "This is the paraphrased version."}
        }
        
        # Mock clean_generated_content to avoid modifying the mock response
        with patch.object(self.expander, 'clean_generated_content', return_value="This is the cleaned version."):
            # Paraphrase some text
            original = "This is the original text."
            paraphrased = self.expander.paraphrase_text(original)
            
            # Verify the ollama interface was called correctly
            self.mock_ollama_interface.chat.assert_called_once()
            
            # Verify the result
            self.assertEqual(paraphrased, "This is the cleaned version.")
    
    def test_verify_paraphrase(self):
        """Test verifying paraphrased text."""
        # Set up mock response
        self.mock_ollama_interface.chat.return_value = {
            "message": {"content": "Verified: This is the corrected paraphrased version."}
        }
        
        # Verify a paraphrase
        original = "What is attention in neural networks?"
        paraphrased = "What's attention mechanism in neural nets?"
        reference = {"field": "machine learning"}
        is_question = True
        
        verified = self.expander.verify_paraphrase(original, paraphrased, reference, is_question)
        
        # Verify the ollama interface was called correctly
        self.mock_ollama_interface.chat.assert_called_once()
        
        # Check that the system prompt contains verification instructions
        system_prompt = self.mock_ollama_interface.chat.call_args[1]['messages'][0]['content']
        self.assertIn("verif", system_prompt.lower())
        
        # Check that the user prompt contains both original and paraphrased texts
        user_prompt = self.mock_ollama_interface.chat.call_args[1]['messages'][1]['content']
        self.assertIn(original, user_prompt)
        self.assertIn(paraphrased, user_prompt)
        self.assertIn(str(reference), user_prompt)
        
        # Verify the result
        self.assertEqual(verified, "Verified: This is the corrected paraphrased version.")
        
        # Test handling of questions without question marks
        self.mock_ollama_interface.chat.return_value = {
            "message": {"content": "What is attention"}
        }
        
        verified = self.expander.verify_paraphrase(original, "What is attention", {}, True)
        
        # Should add a question mark
        self.assertEqual(verified, "What is attention?")
    
    def test_clean_generated_content(self):
        """Test cleaning generated content."""
        # Test removing explanatory phrases
        text = "Generated content: This is the actual content."
        cleaned = self.expander.clean_generated_content(text, False)
        self.assertEqual(cleaned, "This is the actual content.")
        
        # Test removing placeholders
        text = "This contains a ___PLACEHOLDER___ in the middle."
        cleaned = self.expander.clean_generated_content(text, False)
        self.assertEqual(cleaned, "This contains a  in the middle.")
        
        # Test fixing capitalization
        text = "this should start with a capital letter."
        cleaned = self.expander.clean_generated_content(text, False)
        self.assertEqual(cleaned, "This should start with a capital letter.")
        
        # Test ensuring proper punctuation for statements
        text = "This is a statement"
        cleaned = self.expander.clean_generated_content(text, False)
        self.assertEqual(cleaned, "This is a statement.")
        
        # Test ensuring proper punctuation for questions
        text = "Is this a question"
        cleaned = self.expander.clean_generated_content(text, True)
        self.assertEqual(cleaned, "Is this a question?")
    
    @patch('agentChef.dataset_expander.DatasetExpander.expand_conversation_dataset')
    @patch('agentChef.conversation_generator.OllamaConversationGenerator.generate_conversation')
    def test_generate_conversations_from_paper(self, mock_generate, mock_expand):
        """Test generating conversations from a paper."""
        # Mock the conversation generator
        mock_conversation_generator = MagicMock(spec=OllamaConversationGenerator)
        
        # Configure mock_generate to return sample conversations
        sample_conversation = [
            {"from": "human", "value": "What is this paper about?"},
            {"from": "gpt", "value": "This paper is about attention mechanisms."}
        ]
        mock_generate.return_value = sample_conversation
        mock_conversation_generator.generate_conversation.return_value = sample_conversation
        
        # Set up chunking to return test chunks
        mock_conversation_generator.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        
        # Configure mock_expand to return expanded conversations
        expanded_conversation = [
            {"from": "human", "value": "What's the paper discussing?"},
            {"from": "gpt", "value": "The paper discusses attention mechanisms in detail."}
        ]
        mock_expand.return_value = [expanded_conversation]
        
        # Generate conversations from paper
        paper_content = "This is a sample paper content."
        original, expanded = self.expander.generate_conversations_from_paper(
            paper_content=paper_content,
            conversation_generator=mock_conversation_generator,
            num_chunks=2,
            num_turns=3,
            expansion_factor=1,
            static_fields={'human': True, 'gpt': False},
            reference_fields=['human']
        )
        
        # Verify the chunking was called correctly
        mock_conversation_generator.chunk_text.assert_called_once_with(
            paper_content, chunk_size=2000, overlap=200
        )
        
        # Verify the conversation generator was called for each chunk
        self.assertEqual(mock_conversation_generator.generate_conversation.call_count, 2)
        
        # Verify the expansion was called with correct parameters
        mock_expand.assert_called_once()
        self.assertIn('static_fields', mock_expand.call_args[1])
        self.assertEqual(mock_expand.call_args[1]['static_fields'], {'human': True, 'gpt': False})
    
    def test_save_conversations_to_jsonl(self):
        """Test saving conversations to JSONL format."""
        with patch("builtins.open", mock_open()) as mock_file:
            # Save conversations to JSONL
            output_path = self.expander.save_conversations_to_jsonl(
                self.sample_conversations, "test_conversations"
            )
            
            # Verify file was opened correctly
            expected_path = os.path.join(self.expander.output_dir, "test_conversations.jsonl")
            mock_file.assert_called_once_with(expected_path, 'w', encoding='utf-8')
            
            # Verify each conversation was written
            handle = mock_file()
            self.assertEqual(handle.write.call_count, 2)  # Once for each conversation
            
            # Verify the output path is correct
            self.assertEqual(output_path, expected_path)
    
    def test_save_conversations_to_parquet(self):
        """Test saving conversations to Parquet format."""
        with patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            # Save conversations to Parquet
            output_path = self.expander.save_conversations_to_parquet(
                self.sample_conversations, "test_conversations"
            )
            
            # Verify DataFrame.to_parquet was called correctly
            expected_path = os.path.join(self.expander.output_dir, "test_conversations.parquet")
            mock_to_parquet.assert_called_once_with(expected_path, engine='pyarrow')
            
            # Verify the output path is correct
            self.assertEqual(output_path, expected_path)
    
    def test_load_conversations_from_jsonl(self):
        """Test loading conversations from JSONL format."""
        # Create a sample JSONL content
        jsonl_content = '\n'.join([json.dumps(conv) for conv in self.sample_conversations])
        
        # Mock the open function to return our sample content
        with patch("builtins.open", mock_open(read_data=jsonl_content)) as mock_file:
            # Load conversations from JSONL
            conversations = self.expander.load_conversations_from_jsonl("dummy_path.jsonl")
            
            # Verify file was opened correctly
            mock_file.assert_called_once_with("dummy_path.jsonl", 'r', encoding='utf-8')
            
            # Verify the loaded conversations match our sample
            self.assertEqual(len(conversations), 2)
            self.assertEqual(conversations[0][0]["from"], "human")
            self.assertEqual(conversations[0][0]["value"], "What are attention mechanisms?")
    
    def test_convert_conversations_to_dataframe(self):
        """Test converting conversations to DataFrame."""
        # Convert conversations to DataFrame
        df = self.expander.convert_conversations_to_dataframe(self.sample_conversations)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)  # 2 conversations x 2 turns
        
        # Verify columns
        expected_columns = [
            'conversation_id', 'turn_idx', 'source', 'content', 
            'content_length', 'word_count', 'is_question'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Verify values
        self.assertEqual(df.iloc[0]['source'], 'human')
        self.assertEqual(df.iloc[0]['content'], 'What are attention mechanisms?')
        self.assertTrue(df.iloc[0]['is_question'])
        
        # Test with empty conversations
        empty_df = self.expander.convert_conversations_to_dataframe([])
        self.assertIsInstance(empty_df, pd.DataFrame)
        self.assertEqual(len(empty_df), 0)
        self.assertTrue(all(col in empty_df.columns for col in expected_columns))
    
    def test_convert_to_multi_format(self):
        """Test converting conversations to multiple formats."""
        with patch('agentChef.dataset_expander.DatasetExpander.save_conversations_to_jsonl') as mock_jsonl, \
             patch('agentChef.dataset_expander.DatasetExpander.save_conversations_to_parquet') as mock_parquet, \
             patch('pandas.DataFrame.to_csv') as mock_csv:
            
            # Mock the return values
            mock_jsonl.return_value = os.path.join(self.expander.output_dir, "test.jsonl")
            mock_parquet.return_value = os.path.join(self.expander.output_dir, "test.parquet")
            
            # Convert to multiple formats
            output_files = self.expander.convert_to_multi_format(
                self.sample_conversations,
                "test",
                formats=['jsonl', 'parquet', 'csv', 'df']
            )
            
            # Verify each format was called correctly
            mock_jsonl.assert_called_once_with(self.sample_conversations, "test")
            mock_parquet.assert_called_once_with(self.sample_conversations, "test")
            mock_csv.assert_called_once()
            
            # Verify the output dictionary
            self.assertIn('jsonl', output_files)
            self.assertIn('parquet', output_files)
            self.assertIn('csv', output_files)
            self.assertIn('df', output_files)
            self.assertIsInstance(output_files['df'], pd.DataFrame)
    
    @patch('agentChef.pandas_query.OllamaLlamaIndexIntegration')
    def test_analyze_expanded_dataset(self, mock_ollama_query):
        """Test analyzing the expanded dataset."""
        # Mock the query engine
        mock_query_engine = mock_ollama_query.return_value
        mock_query_engine.query_dataframe_with_ollama.return_value = {
            "response": "Analysis result"
        }
        
        # Set the mock query engine
        self.expander.pandas_query = None  # Ensure pandas_query is not used
        self.expander.ollama_query = mock_query_engine
        
        # Analyze the dataset
        analysis = self.expander.analyze_expanded_dataset(
            self.sample_conversations,
            self.sample_conversations * 2  # Double the size to simulate expansion
        )
        
        # Verify the analysis structure
        self.assertIn('original_count', analysis)
        self.assertIn('expanded_count', analysis)
        self.assertIn('expansion_ratio', analysis)
        self.assertIn('basic_statistics', analysis)
        self.assertIn('advanced_analysis', analysis)
        self.assertIn('ollama_analysis', analysis['advanced_analysis'])
    
    def test_is_question(self):
        """Test the question detection function."""
        # Test obvious questions with question marks
        self.assertTrue(self.expander._is_question("What is this?"))
        self.assertTrue(self.expander._is_question("How does it work?"))
        
        # Test questions without question marks but starting with question words
        self.assertTrue(self.expander._is_question("What is this"))
        self.assertTrue(self.expander._is_question("how does it work"))
        
        # Test questions starting with modal verbs
        self.assertTrue(self.expander._is_question("Can you explain this"))
        self.assertTrue(self.expander._is_question("Would it be possible"))
        
        # Test statements (not questions)
        self.assertFalse(self.expander._is_question("This is a statement."))
        self.assertFalse(self.expander._is_question("The model works well."))
        
        # Test edge cases
        self.assertFalse(self.expander._is_question(""))
        self.assertTrue(self.expander._is_question("question?"))

if __name__ == "__main__":
    unittest.main()