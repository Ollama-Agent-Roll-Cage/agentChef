import logging
from typing import List, Dict, Any, Optional

try:
    import ollama
    from ollama import Client, AsyncClient, ResponseError
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama package not found. Please install with 'pip install ollama'")

class OllamaInterface:
    """
    A unified interface for interacting with Ollama using the official Python library.
    This class provides a consistent way to access Ollama functionality
    throughout the agentChef package.
    """
    
    def __init__(self, model_name="llama3", host="http://localhost:11434"):
        """
        Initialize the Ollama interface.
        
        Args:
            model_name (str): Name of the Ollama model to use
            host (str): Ollama API host URL
        """
        self.model = model_name
        self.host = host
        self.logger = logging.getLogger(__name__)
        self.ollama_available = OLLAMA_AVAILABLE
        
        # Set up clients if ollama is available
        if self.ollama_available:
            try:
                self.client = Client(host=self.host)
                self.async_client = AsyncClient(host=self.host)
            except Exception as e:
                self.logger.warning(f"Could not initialize Ollama clients: {e}")
                self.ollama_available = False
    
    def chat(self, messages: List[Dict[str, str]], stream=False) -> Dict[str, Any]:
        """
        Send a chat request to Ollama.
        
        Args:
            messages (List[Dict[str, str]]): List of message objects in the format:
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            stream (bool): Whether to stream the response
            
        Returns:
            Dict[str, Any]: Response from Ollama or an error message
        """
        if not self.ollama_available:
            error_msg = "Ollama is not available. Please install with 'pip install ollama'"
            self.logger.error(error_msg)
            return {"error": error_msg, "message": {"content": error_msg}}
        
        try:
            return ollama.chat(model=self.model, messages=messages, stream=stream)
        except ResponseError as e:
            error_msg = f"Ollama API error: {e.error} (Status code: {e.status_code})"
            self.logger.error(error_msg)
            return {"error": error_msg, "message": {"content": error_msg}}
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
            response = ollama.embed(model=self.model, input=text)
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
            # Use a simple list call to check if Ollama server is responding
            ollama.list()
            return True
        except Exception as e:
            self.logger.error(f"Ollama is not accessible: {str(e)}")
            return False
    
    async def async_chat(self, messages: List[Dict[str, str]], stream=False):
        """
        Asynchronously send a chat request to Ollama.
        
        Args:
            messages (List[Dict[str, str]]): List of message objects
            stream (bool): Whether to stream the response
            
        Returns:
            Response from Ollama or async generator if streaming
        """
        if not self.ollama_available:
            error_msg = "Ollama is not available. Please install with 'pip install ollama'"
            self.logger.error(error_msg)
            return {"error": error_msg, "message": {"content": error_msg}}
        
        try:
            return await self.async_client.chat(model=self.model, messages=messages, stream=stream)
        except Exception as e:
            error_msg = f"Error in async communication with Ollama: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "message": {"content": error_msg}}
    
    async def list_models(self):
        """Get list of available models."""
        try:
            self.logger.info("Calling ollama.list() to get available models")
            result = ollama.list()
            self.logger.info(f"ollama.list() result: {result}")
            
            # Check if it's the new ListResponse format
            if hasattr(result, 'models'):
                models = [model.model for model in result.models if hasattr(model, 'model')]
            # Check if it's the list of Model objects format
            elif isinstance(result, list):
                models = [model.model for model in result if hasattr(model, 'model')]
            # Check if it's the old format (dict with 'models' key)
            elif isinstance(result, dict) and 'models' in result:
                models = [model['name'] for model in result['models'] if 'name' in model]
            else:
                self.logger.warning(f"Unexpected response format from ollama.list(): {type(result)}")
                return ["llama2"]  # Return default model if format unknown
                
            return models or ["llama2"]  # Return default if no models found
                
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return ["llama2"]  # Return default model on error
    
    def set_model(self, model_name: str):
        """Set the active model."""
        self.model = model_name
        self.logger.info(f"Set active model to: {model_name}")