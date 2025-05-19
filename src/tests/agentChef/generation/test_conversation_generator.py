import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import pytest

from src.agentChef.generation.conversation_generator import OllamaConversationGenerator
from src.agentChef.ollama.ollama_interface import OllamaInterface

class TestOllamaConversationGenerator(unittest.TestCase):
    
    def setUp(self):
        # Create a mock Ollama interface
        self.mock_ollama_interface = MagicMock(spec=OllamaInterface)
        
        # Set up the mock response
        self.mock_conversation = [
            {"from": "human", "value": "What are attention mechanisms in neural networks?"},
            {"from": "gpt", "value": "Attention mechanisms allow neural networks to focus on specific parts of the input."}
        ]
        
        # Configure the mock ollama interface to return our test conversation
        self.mock_chat_response = {
            "message": {
                "content": json.dumps(self.mock_conversation)
            }
        }
        self.mock_ollama_interface.chat.return_value = self.mock_chat_response
        
        # Create the generator with our mock interface
        self.generator = OllamaConversationGenerator(
            model_name="llama3", 
            ollama_interface=self.mock_ollama_interface
        )
    
    def test_chunk_text(self):
        """Test the static text chunking method with various inputs."""
        # Test with a short text that doesn't need chunking
        text = "This is a short text."
        chunks = self.generator.chunk_text(text, chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
        
        # Test with text that needs to be split
        long_text = "This is a longer text. It has multiple sentences. We want to make sure it gets split correctly. This should be enough text to create multiple chunks."
        chunks = self.generator.chunk_text(long_text, chunk_size=50, overlap=10)
        self.assertTrue(len(chunks) > 1)
        
        # Verify that chunks have overlap
        if len(chunks) > 1:
            # Check if the end of chunk 1 is at the start of chunk 2
            overlap_text = chunks[0][-10:]
            self.assertTrue(overlap_text in chunks[1])
        
        # Test with empty text
        empty_text = ""
        chunks = self.generator.chunk_text(empty_text, chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], empty_text)
        
        # Test with None input
        with self.assertRaises(AttributeError):
            self.generator.chunk_text(None, chunk_size=50, overlap=10)
    
    def test_generate_conversation(self):
        """Test generating a conversation."""
        content = "Attention mechanisms have become an integral part of sequence modeling."
        conversation = self.generator.generate_conversation(
            content=content,
            num_turns=3,
            conversation_context="research"
        )
        
        # Verify the ollama interface was called correctly
        self.mock_ollama_interface.chat.assert_called_once()
        
        # Check that the system prompt contains our content
        system_prompt = self.mock_ollama_interface.chat.call_args[1]['messages'][0]['content']
        self.assertIn(content, system_prompt)
        self.assertIn("research", system_prompt)
        
        # Verify the returned conversation matches our mock
        self.assertEqual(conversation, self.mock_conversation)
    
    def test_validate_conversation_format(self):
        """Test the conversation format validation."""
        # Valid conversation
        valid_conversation = [
            {"from": "human", "value": "Question?"},
            {"from": "gpt", "value": "Answer."}
        ]
        
        # This should not raise an exception
        self.generator._validate_conversation_format(valid_conversation)
        
        # Test with different field names
        conversation_with_role = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."}
        ]
        
        # Clone to avoid modifying the original when validating
        import copy
        conv_to_validate = copy.deepcopy(conversation_with_role)
        
        # This should normalize the fields
        self.generator._validate_conversation_format(conv_to_validate)
        
        # Check that fields were normalized
        self.assertEqual(conv_to_validate[0]["from"], "human")
        self.assertEqual(conv_to_validate[0]["value"], "Question?")
        self.assertEqual(conv_to_validate[1]["from"], "gpt")
        self.assertEqual(conv_to_validate[1]["value"], "Answer.")
        
        # Test with invalid conversation (not a list)
        with self.assertRaises(ValueError):
            self.generator._validate_conversation_format("not a list")
        
        # Test with invalid turn (not a dict)
        with self.assertRaises(ValueError):
            self.generator._validate_conversation_format(["not a dict"])
    
    def test_generate_hedged_response(self):
        """Test generating a hedged response."""
        # Set up mock response for the hedged response
        hedged_response = "I believe attention mechanisms allow models to focus on relevant parts of the input."
        self.mock_ollama_interface.chat.return_value = {
            "message": {
                "content": hedged_response
            }
        }
        
        # Generate a hedged response
        prompt = "What are attention mechanisms?"
        response = self.generator.generate_hedged_response(
            prompt=prompt,
            hedging_profile="balanced",
            knowledge_level="medium",
            subject_expertise="general"
        )
        
        # Verify the ollama interface was called correctly
        self.mock_ollama_interface.chat.assert_called()
        
        # Verify the response matches our mock
        self.assertEqual(response, hedged_response)
        
        # Check that the system prompt contains our hedging instructions
        system_prompt = self.mock_ollama_interface.chat.call_args[1]['messages'][0]['content']
        self.assertIn("balanced", system_prompt.lower())
        self.assertIn("moderate knowledge", system_prompt)

    @patch('agentChef.conversation_generator.OllamaLlamaIndexIntegration')
    def test_analyze_conversation_hedging(self, mock_query_engine_class):
        """Test analyzing hedging patterns in conversations."""
        # Create mock conversations
        conversations = [
            [
                {"from": "human", "value": "What are attention mechanisms?"},
                {"from": "gpt", "value": "I think attention mechanisms allow models to focus on specific parts of input."}
            ],
            [
                {"from": "human", "value": "Are transformers better than RNNs?"},
                {"from": "gpt", "value": "It depends on the task. In my opinion, transformers excel at certain tasks."}
            ]
        ]
        
        # Set up the mock query engine
        mock_query_engine = mock_query_engine_class.return_value
        mock_query_engine.analyze_conversation_data.return_value = {
            "total_conversations": 2,
            "hedging_phrases_detected": ["I think", "in my opinion"],
            "hedging_rate": 0.5
        }
        
        # Create a generator with a mock query engine
        generator_with_query = OllamaConversationGenerator(model_name="llama3", enable_hedging=True)
        generator_with_query.query_engine = mock_query_engine
        
        # Analyze the conversations
        results = generator_with_query.analyze_conversation_hedging(conversations)
        
        # Verify the query engine was called correctly
        mock_query_engine.analyze_conversation_data.assert_called_once()
        
        # Verify the results
        self.assertEqual(results["total_conversations"], 2)
        self.assertIn("hedging_phrases_detected", results)
        self.assertIn("hedging_rate", results)
        
        # Test without query engine (fallback to basic analysis)
        generator_no_query = OllamaConversationGenerator(model_name="llama3", enable_hedging=True)
        generator_no_query.query_engine = None
        
        # Should still work using the basic approach
        basic_results = generator_no_query.analyze_conversation_hedging(conversations)
        
        # Verify we get basic stats
        self.assertEqual(basic_results["total_conversations"], 2)
        self.assertIn("hedging_counts", basic_results)
        self.assertIn("by_source", basic_results)
        self.assertTrue("I think" in basic_results["hedging_counts"])

    def test_basic_hedging_analysis(self):
        """Test the basic hedging analysis method."""
        conversations = [
            [
                {"from": "human", "value": "What are attention mechanisms?"},
                {"from": "gpt", "value": "I think attention mechanisms allow models to focus on specific parts of input."}
            ],
            [
                {"from": "human", "value": "Are transformers better than RNNs?"},
                {"from": "gpt", "value": "It depends on the task. In my opinion, transformers excel at certain tasks."}
            ]
        ]
        
        # Run basic analysis
        results = self.generator._basic_hedging_analysis(conversations)
        
        # Verify results structure
        self.assertEqual(results["total_conversations"], 2)
        self.assertEqual(results["total_turns"], 4)
        self.assertIn("hedging_counts", results)
        self.assertIn("by_source", results)
        self.assertIn("examples", results)
        
        # Check hedging counts
        self.assertTrue(results["hedging_counts"]["I think"] > 0)
        self.assertTrue(results["hedging_counts"]["in my opinion"] > 0)
        
        # Check source counts
        self.assertTrue(results["by_source"]["gpt"] > 0)
        
        # Check rates
        self.assertTrue(0 <= results["gpt_hedging_rate"] <= 1)
        self.assertTrue(0 <= results["human_hedging_rate"] <= 1)

    def test_conversations_to_df(self):
        """Test converting conversations to DataFrame."""
        conversations = [
            [
                {"from": "human", "value": "What are attention mechanisms?"},
                {"from": "gpt", "value": "I think attention mechanisms allow models to focus on specific parts."}
            ]
        ]
        
        # Convert to DataFrame
        df = self.generator._conversations_to_df(conversations)
        
        # Verify DataFrame structure
        self.assertEqual(len(df), 2)  # Two turns
        self.assertIn("conversation_idx", df.columns)
        self.assertIn("turn_idx", df.columns)
        self.assertIn("from", df.columns)
        self.assertIn("value", df.columns)
        
        # Verify values
        self.assertEqual(df.iloc[0]["from"], "human")
        self.assertEqual(df.iloc[1]["from"], "gpt")
        self.assertTrue("attention mechanisms" in df.iloc[0]["value"])
        
        # Check hedging feature columns
        self.assertTrue(any(col.startswith("has_") for col in df.columns))
        
        # Check that "I think" was detected
        self.assertTrue(df.iloc[1]["has_I_think"] == 1)

    @patch('agentChef.conversation_generator.OllamaConversationGenerator.generate_conversation')
    def test_generate_conversations_batch(self, mock_generate):
        """Test generating conversations in batch."""
        # Set up mock responses for generate_conversation
        mock_conversations = [
            [
                {"from": "human", "value": "Question 1?"},
                {"from": "gpt", "value": "Answer 1."}
            ],
            [
                {"from": "human", "value": "Question 2?"},
                {"from": "gpt", "value": "Answer 2."}
            ]
        ]
        mock_generate.side_effect = mock_conversations
        
        # Create content chunks
        content_chunks = ["Chunk 1", "Chunk 2"]
        
        # Generate batch conversations
        conversations = self.generator.generate_conversations_batch(
            content_chunks=content_chunks,
            num_turns=3,
            context="research"
        )
        
        # Verify generate_conversation was called for each chunk
        self.assertEqual(mock_generate.call_count, 2)
        
        # Verify calls had correct arguments
        mock_generate.assert_any_call(
            content_chunks[0], 3, "research", hedging_level="balanced"
        )
        mock_generate.assert_any_call(
            content_chunks[1], 3, "research", hedging_level="balanced"
        )
        
        # Verify results
        self.assertEqual(len(conversations), 2)
        self.assertEqual(conversations, mock_conversations)

if __name__ == "__main__":
    unittest.main()
