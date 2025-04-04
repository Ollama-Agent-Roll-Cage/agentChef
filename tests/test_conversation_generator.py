import unittest
from unittest.mock import patch, MagicMock

from agentChef.conversation_generator import OllamaConversationGenerator

class TestOllamaConversationGenerator(unittest.TestCase):
    
    @patch('agentChef.conversation_generator.ollama')
    def test_chunk_text(self, mock_ollama):
        # Test the static method directly
        generator = OllamaConversationGenerator(model_name="llama3")
        text = "This is a test. It has multiple sentences. We want to check chunking."
        chunks = generator.chunk_text(text, chunk_size=20, overlap=5)
        
        # Check that we got chunks
        self.assertTrue(len(chunks) > 1)
        
        # Check that all content is preserved
        combined = ''.join(chunks)
        self.assertEqual(len(combined), len(text))

if __name__ == "__main__":
    unittest.main()