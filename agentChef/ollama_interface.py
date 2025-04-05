import logging
from typing import List, Dict, Any, Optional

class OllamaInterface:
    """
    A unified interface for interacting with Ollama.
    This class provides a consistent way to access Ollama functionality
    throughout the agentChef package.
    """
    
    def __init__(self, model_name="llama3"):
        """
        Initialize the Ollama interface.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model = model_name
        self.logger = logging.getLogger(__name__)
        
        # Check if ollama is available
        try:
            import ollama
            self.ollama = ollama
            self.ollama_available = True
        except ImportError:
            self.ollama_available = False
            self.logger.warning("Ollama package not found. Please install with 'pip install ollama'")
    
    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Send a chat request to Ollama.
        
        Args:
            messages (List[Dict[str, str]]): List of message objects in the format:
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        
        Returns:
            Dict[str, Any]: Response from Ollama or an error message
        """
        if not self.ollama_available:
            error_msg = "Ollama is not available. Please install with 'pip install ollama'"
            self.logger.error(error_msg)
            return {"error": error_msg, "message": {"content": error_msg}}
        
        try:
            return self.ollama.chat(model=self.model, messages=messages)
        except Exception as e:
            error_msg = f"Error communicating with Ollama: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "message": {"content": error_msg}}
    
    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Ollama.
        
        Args:
            text (str): Text to create embeddings for
        
        Returns:
            List[float]: Embedding vector or empty list on error
        """
        if not self.ollama_available:
            self.logger.error("Ollama is not available. Please install with 'pip install ollama'")
            return []
        
        try:
            response = self.ollama.embeddings(model=self.model, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and working.
        
        Returns:
            bool: True if Ollama is available and working, False otherwise
        """
        if not self.ollama_available:
            return False
            
        try:
            # Try a simple request to check if Ollama server is responding
            self.ollama.list()
            return True
        except Exception as e:
            self.logger.error(f"Ollama is not accessible: {str(e)}")
            return False