import unittest
from unittest.mock import patch, MagicMock, Mock
import logging
import sys

# Import the module to test
from src.agentChef.ollama.ollama_interface import OllamaInterface

class TestOllamaInterface(unittest.TestCase):
    
    def setUp(self):
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_init_with_ollama_available(self):
        """Test initialization when Ollama is available."""
        # Mock the Ollama modules before importing
        with patch.dict('sys.modules', {'ollama': MagicMock()}), \
             patch('ollama.Client', return_value=MagicMock()), \
             patch('ollama.AsyncClient', return_value=MagicMock()):
                
            # Create interface
            interface = OllamaInterface(model_name="llama3")
            
            # Verify the instance was created correctly
            self.assertEqual(interface.model, "llama3")
            self.assertTrue(interface.ollama_available)
    
    def test_init_with_ollama_unavailable(self):
        """Test initialization when Ollama is not available."""
        
        def mock_import(name, *args, **kwargs):
            if name == 'ollama':
                raise ImportError("No module named 'ollama'")
            return __import__(name, *args, **kwargs)
        
        # Patch the import function
        with patch('builtins.__import__', side_effect=mock_import):
            # Create interface without ollama available
            interface = OllamaInterface(model_name="llama3")
            
            # Verify the instance was created correctly
            self.assertEqual(interface.model, "llama3")
            self.assertFalse(interface.ollama_available)
    
    @patch('ollama.chat')
    def test_chat(self, mock_ollama_chat):
        """Test the chat method."""
        # Configure the mock response
        mock_response = {
            "message": {
                "content": "This is a test response."
            }
        }
        mock_ollama_chat.return_value = mock_response
        
        # Create interface and call chat
        interface = OllamaInterface(model_name="llama3")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        
        response = interface.chat(messages=messages)
        
        # Verify ollama.chat was called correctly
        mock_ollama_chat.assert_called_once_with(model="llama3", messages=messages, stream=False)
        
        # Verify the response
        self.assertEqual(response, mock_response)
    
    @patch('ollama.chat')
    def test_chat_error_handling(self, mock_ollama_chat):
        """Test error handling in the chat method."""
        # Configure the mock to raise an exception
        mock_ollama_chat.side_effect = Exception("Test error")
        
        # Create interface and call chat
        interface = OllamaInterface(model_name="llama3")
        messages = [{"role": "user", "content": "Hello!"}]
        
        response = interface.chat(messages=messages)
        
        # Verify error handling creates expected response
        self.assertIn("error", response)
        self.assertIn("Test error", response["error"])
        self.assertIn("message", response)
        self.assertIn("content", response["message"])
        self.assertIn("Error communicating with Ollama", response["message"]["content"])
    
    def test_chat_ollama_unavailable(self):
        """Test chat method when Ollama is not available."""
        # Create interface with ollama unavailable
        interface = OllamaInterface(model_name="llama3")
        interface.ollama_available = False
        
        # Call chat
        messages = [{"role": "user", "content": "Hello!"}]
        response = interface.chat(messages=messages)
        
        # Verify error message about Ollama being unavailable
        self.assertIn("error", response)
        self.assertIn("Ollama is not available", response["error"])
        self.assertIn("message", response)
        self.assertIn("content", response["message"])
    
    @patch('ollama.embed')
    def test_embeddings(self, mock_ollama_embed):
        """Test the embeddings method."""
        # Configure the mock response
        mock_embedding = [0.1, 0.2, 0.3]
        mock_ollama_embed.return_value = {"embedding": mock_embedding}
        
        # Create interface and call embeddings
        interface = OllamaInterface(model_name="llama3")
        result = interface.embeddings(text="Hello, world!")
        
        # Verify ollama.embed was called correctly
        mock_ollama_embed.assert_called_once_with(model="llama3", input="Hello, world!")
        
        # Verify the result
        self.assertEqual(result, mock_embedding)
    
    @patch('ollama.embed')
    def test_embeddings_error_handling(self, mock_ollama_embed):
        """Test error handling in the embeddings method."""
        # Configure the mock to raise an exception
        mock_ollama_embed.side_effect = Exception("Test error")
        
        # Create interface and call embeddings
        interface = OllamaInterface(model_name="llama3")
        result = interface.embeddings(text="Hello, world!")
        
        # Verify error handling returns empty list
        self.assertEqual(result, [])
    
    def test_embeddings_ollama_unavailable(self):
        """Test embeddings method when Ollama is not available."""
        # Create interface with ollama unavailable
        interface = OllamaInterface(model_name="llama3")
        interface.ollama_available = False
        
        # Call embeddings
        result = interface.embeddings(text="Hello, world!")
        
        # Verify empty list is returned
        self.assertEqual(result, [])
    
    @patch('ollama.list')
    def test_is_available(self, mock_ollama_list):
        """Test the is_available method."""
        # Configure the mock
        mock_ollama_list.return_value = ["llama3", "mistral"]
        
        # Create interface and check availability
        interface = OllamaInterface(model_name="llama3")
        result = interface.is_available()
        
        # Verify ollama.list was called
        mock_ollama_list.assert_called_once()
        
        # Verify the result
        self.assertTrue(result)
    
    @patch('ollama.list')
    def test_is_available_error(self, mock_ollama_list):
        """Test is_available method when Ollama returns an error."""
        # Configure the mock to raise an exception
        mock_ollama_list.side_effect = Exception("Ollama server not running")
        
        # Create interface and check availability
        interface = OllamaInterface(model_name="llama3")
        result = interface.is_available()
        
        # Verify the result is False
        self.assertFalse(result)
    
    def test_is_available_ollama_unavailable(self):
        """Test is_available method when Ollama is not available."""
        # Create interface with ollama unavailable
        interface = OllamaInterface(model_name="llama3")
        interface.ollama_available = False
        
        # Check availability
        result = interface.is_available()
        
        # Verify the result is False
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()