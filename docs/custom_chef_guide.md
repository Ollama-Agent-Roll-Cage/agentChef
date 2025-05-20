# Creating Custom Chefs with AgentChef

## Core Functionality (UI-Independent)

AgentChef's core functionality can be used without any UI components. Here's how:

1. Create a headless chef:

```python
from agentChef.core.chefs import BaseChef

class HeadlessChef(BaseChef):
    def __init__(self):
        super().__init__(
            name="headless_chef",
            model_name="llama3",
            enable_ui=False  # Disable UI components
        )
        self.setup_components()
        
    def setup_components(self):
        """Set up only the core processing components."""
        # Dataset components
        self.ollama = OllamaInterface(model_name=self.model_name)
        self.expander = DatasetExpander(ollama_interface=self.ollama)
        self.cleaner = DatasetCleaner(ollama_interface=self.ollama)
        
        # Storage and analysis
        self.storage = ParquetStorageWrapper()
        self.pandas_query = PandasQueryIntegration()
        
        # Register components
        self.register_component("expander", self.expander)
        self.register_component("cleaner", self.cleaner)
```

2. Use core components directly:

```python
# Direct component usage
async def process_dataset(self, input_data: List[Dict[str, Any]]):
    # Generate conversations
    conversations = await self.ollama.chat(messages=[...])
    
    # Expand dataset
    expanded = await self.expander.expand_conversation_dataset(
        conversations=conversations,
        expansion_factor=2
    )
    
    # Clean data
    cleaned = await self.cleaner.clean_dataset(
        original_conversations=conversations,
        expanded_conversations=expanded
    )
    
    # Save results
    self.storage.save_to_parquet(
        cleaned, 
        "output.parquet"
    )
```

3. Handle events programmatically:

```python
def setup_callbacks(self):
    """Set up event handling without UI."""
    self.register_callback("process_start", self.log_start)
    self.register_callback("process_complete", self.log_complete)
    
def log_start(self, data):
    """Log process start."""
    logging.info(f"Processing started: {data}")
    
def log_complete(self, result):
    """Log process completion."""
    logging.info(f"Processing complete: {result}")
```

## Core Components (No UI Required)

These components work independently of any UI:

1. LLM Integration:
```python
from agentChef.core.ollama import OllamaInterface

ollama = OllamaInterface(model_name="llama3")
response = await ollama.chat(messages=[...])
```

2. Dataset Processing:
```python
from agentChef.core.augmentation import DatasetExpander
from agentChef.core.classification import DatasetCleaner

expander = DatasetExpander(ollama_interface=ollama)
cleaner = DatasetCleaner(ollama_interface=ollama)
```

3. Data Storage:
```python
from agentChef.core.crawlers import ParquetStorageWrapper

storage = ParquetStorageWrapper()
storage.save_to_parquet(data, "output.parquet")
```

4. Analysis Tools:
```python
from agentChef.core.llamaindex import PandasQueryIntegration

query_engine = PandasQueryIntegration()
results = query_engine.query_dataframe(df, "analyze this data")
```

## Example: Command-Line Chef

Here's a complete example of a chef that runs entirely from the command line:

```python
class CLIChef(BaseChef):
    def __init__(self):
        super().__init__(
            name="cli_chef",
            model_name="llama3",
            enable_ui=False
        )
        self.setup_components()
        
    async def run(self, input_file: str, output_file: str):
        """Process data from command line."""
        # Load data
        data = self.storage.load_from_parquet(input_file)
        
        # Process
        result = await self.process(data)
        
        # Save
        self.storage.save_to_parquet(result, output_file)
        return result

# Usage from command line
if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    chef = CLIChef()
    result = asyncio.run(chef.run(args.input, args.output))
```

## Optional UI Integration

If you later decide to add a UI, you can:

1. Enable UI support:
```python
chef = MyChef(enable_ui=True)
```

2. Add UI components:
```python
from agentChef.core.ui_components import ChefUI
chef.ui = ChefUI(chef)
```

## Creating Your First Chef

1. First, inherit from BaseChef:

```python
from agentChef.core.chefs import BaseChef

class MyCustomChef(BaseChef):
    def __init__(self):
        super().__init__(
            name="my_chef",
            model_name="llama3",
            enable_ui=True
        )
        self.setup_components()
```

2. Add your components:

```python
def setup_components(self):
    # Add dataset components
    self.register_component("expander", DatasetExpander(
        ollama_interface=self.ollama
    ))
    self.register_component("cleaner", DatasetCleaner(
        ollama_interface=self.ollama
    ))
    
    # Add crawlers if needed
    self.register_component("web_crawler", WebCrawlerWrapper())
```

3. Implement core methods:

```python
async def process(self, input_data: Any) -> Dict[str, Any]:
    """Process input through your pipeline."""
    self.emit_event("process_start", input_data)
    
    # Your processing logic here
    result = await self.ollama.chat(...)
    
    self.emit_event("process_complete", result)
    return {"result": result}

async def generate(self, **kwargs) -> Dict[str, Any]:
    """Generate content."""
    # Your generation logic here
    pass
```

4. Add a UI (optional):

```python
def setup_ui(self):
    """Set up custom UI elements."""
    if not self.ui:
        return
        
    # Add custom widgets
    self.ui.add_text_input("input", "Enter input:")
    self.ui.add_button("process", "Process", self.on_process_click)
    
    # Add progress tracking
    self.register_callback("process_start", self.ui.show_progress)
    self.register_callback("process_complete", self.ui.hide_progress)
```

## Using Components

AgentChef provides many built-in components:

- OllamaInterface: LLM integration
- DatasetExpander: Data augmentation  
- DatasetCleaner: Quality validation
- WebCrawlerWrapper: Web content fetching
- ParquetStorage: Data persistence
- PandasQuery: Data analysis
- PyVis: Visualization

Example using components:

```python
# Get component
expander = self.components["expander"]

# Use component
expanded_data = await expander.expand_conversation_dataset(
    conversations=conversations,
    expansion_factor=2
)
```

## Building UIs

AgentChef uses PyQt6 for UIs. Key features:

1. Basic Interface:
```python
# The base UI provides:
- Input/output areas
- Progress tracking
- Settings panel
- Status bar
```

2. Custom Widgets:
```python 
def setup_ui(self):
    # Add custom widgets
    self.ui.add_tab("Research")
    self.ui.add_tree_view("results")
    self.ui.add_graph_view("visualization")
```

3. Event Handling:
```python 
def on_process_click(self):
    """Handle process button click."""
    input_text = self.ui.get_input("input")
    asyncio.create_task(self.process(input_text))
```

## Best Practices

1. Use async/await for long operations
2. Emit events for progress tracking  
3. Handle errors gracefully
4. Clean up resources properly
5. Follow the MCP protocol
6. Add comprehensive logging

## Example: Complete Chef

Here's a complete example chef:

```python
class DataAnalysisChef(BaseChef):
    def __init__(self):
        super().__init__(
            name="data_analysis",
            model_name="llama3",
            enable_ui=True
        )
        
        # Setup components
        self.register_component(
            "pandas_query",
            PandasQueryIntegration(ollama_interface=self.ollama)
        )
        
        self.register_component(
            "visualizer",
            PyVisGraphBuilder()
        )
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input data."""
        self.emit_event("analysis_start")
        
        # Query the data
        query_results = await self.components["pandas_query"].query_dataframe(
            input_data["df"],
            input_data["query"]
        )
        
        # Visualize results
        graph = self.components["visualizer"].build_graph(query_results)
        
        self.emit_event("analysis_complete")
        
        return {
            "query_results": query_results,
            "visualization": graph
        }
        
    def setup_ui(self):
        """Setup analysis UI."""
        if not self.ui:
            return
            
        # Add data input
        self.ui.add_file_input("data", "Select Data File:")
        
        # Add query input
        self.ui.add_text_input("query", "Enter Query:")
        
        # Add visualization area
        self.ui.add_graph_view("results")
        
        # Add analyze button
        self.ui.add_button("analyze", "Analyze", self.on_analyze_click)
        
    def on_analyze_click(self):
        """Handle analyze button click."""
        data_file = self.ui.get_input("data")
        query = self.ui.get_input("query")
        
        df = pd.read_parquet(data_file)
        asyncio.create_task(self.process({
            "df": df,
            "query": query
        }))
```

## Publishing Your Chef

1. Create a package:
```bash
mkdir my_chef
cd my_chef
touch setup.py
```

2. Add to PyPI:
```bash
python -m build
python -m twine upload dist/*
```

3. Share with the community!

## Getting Help

- Join our Discord community
- Check the documentation
- File issues on GitHub
- Share your chefs!
