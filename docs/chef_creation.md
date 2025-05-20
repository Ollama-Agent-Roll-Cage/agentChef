# Creating Custom Chefs with AgentChef

This guide explains how to create your own custom AI agent pipelines ("chefs") using the AgentChef toolkit.

## Chef Architecture

A chef consists of:
- Core processing logic
- Optional UI components
- Data handling utilities
- MCP protocol integration

### Basic Structure

```python
from agentChef.core.chefs import BaseChef
from agentChef.core.ui_components import ChefUI
from agentChef.core.ollama import OllamaInterface

class MyChef(BaseChef):
    def __init__(self):
        super().__init__()
        self.ollama = OllamaInterface()
        self.ui = ChefUI()  # Optional
        
    async def process(self, input_data):
        # Your chef's main logic
        pass
```

## Core Components

- **OllamaInterface**: LLM integration
- **DatasetExpander**: Data augmentation
- **ParquetStorage**: Data persistence
- **PandasQuery**: Data analysis
- **PyVis**: Visualization

## Adding a UI

1. Create UI class:
```python
from agentChef.core.ui_components import ChefUIBase

class MyChefUI(ChefUIBase):
    def setup_ui(self):
        # Define UI layout
        pass
```

2. Connect to chef:
```python
self.ui = MyChefUI(self)
self.ui.show()
```

## Best Practices

1. Use async/await for long operations
2. Implement progress callbacks
3. Follow the MCP protocol
4. Add comprehensive logging
5. Include usage examples
