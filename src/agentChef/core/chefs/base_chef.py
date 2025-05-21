"""Base chef class for building custom AI agent pipelines."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Use relative import to avoid dependency issues
from ...utils.const import DEFAULT_DATA_DIR
from ..ollama.ollama_interface import OllamaInterface

class BaseChef:
    """Base class for creating custom AI agent pipelines ("chefs").
    
    This class provides core functionality that all chefs should inherit from:
    - Ollama integration 
    - UI support
    - Data persistence
    - Event handling
    - Progress tracking
    
    Example:
    ```python
    class MyCustomChef(BaseChef):
        def __init__(self):
            super().__init__(
                name="my_chef",
                model_name="llama3",
                enable_ui=True
            )
            self.setup_components()
            
        async def process(self, input_data):
            # Your chef's core logic here
            pass
            
        def setup_ui(self):
            # Custom UI setup here
            pass
    ```
    """
    
    def __init__(self, 
                 name: str,
                 model_name: str = "llama3",
                 data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
                 enable_ui: bool = False):
        """Initialize the base chef."""
        self.name = name
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.enable_ui = enable_ui
        
        # Initialize Ollama
        self.ollama = OllamaInterface(model_name=model_name)
        
        # Set up logging
        self.logger = logging.getLogger(f"chef.{name}")
        
        # Initialize UI if enabled
        self.ui = None
        if enable_ui:
            try:
                from agentChef.core.ui_components import ChefUI
                self.ui = ChefUI(self)
                self.setup_ui()
            except ImportError:
                self.logger.warning("UI components not available. Install PyQt6 for UI support.")
        
        # Event callbacks
        self.callbacks = {}
        
        # Component registry
        self.components = {}
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component for use in this chef."""
        self.components[name] = component
        
    def register_callback(self, event: str, callback) -> None:
        """Register a callback for chef events."""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def emit_event(self, event: str, data: Any = None) -> None:
        """Emit an event to registered callbacks."""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in {event} callback: {e}")
    
    def set_model(self, model_name: str) -> None:
        """Update the model used by this chef and its components."""
        self.model_name = model_name
        if self.ollama:
            self.ollama.set_model(model_name)
        
        # Update model for all registered components
        for component in self.components.values():
            if hasattr(component, 'set_model'):
                component.set_model(model_name)
    
    def setup_ui(self) -> None:
        """Set up the chef's UI. Override in subclasses."""
        if self.ui:
            self.ui.setup_basic_interface()
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement process()")
    
    async def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate content. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate()")
    
    def cleanup(self) -> None:
        """Clean up resources. Override in subclasses."""
        pass
    
    def show_ui(self) -> None:
        """Show the chef's UI if enabled."""
        if self.ui:
            self.ui.show()
        else:
            self.logger.warning("UI not enabled for this chef")
