# Creating Custom Chefs with AgentChef

AgentChef v0.2.8 provides a modular architecture for building custom AI agent pipelines ("chefs"). This guide shows you how to create your own specialized agents.

## Quick Start - Simple Custom Chef

```python
from agentChef import BaseChef, PandasRAG
import pandas as pd

class MyDataChef(BaseChef):
    def __init__(self):
        super().__init__(
            name="my_data_chef",
            model_name="llama3.2:3b",
            enable_ui=False  # Start without UI
        )
        
        # Initialize PandasRAG for data analysis
        self.rag = PandasRAG(
            data_dir=f"./chef_data/{self.name}",
            model_name=self.model_name
        )
        
        # Register a specialized agent
        self.agent_id = self.rag.register_agent(
            "data_specialist",
            system_prompt="You are a data analysis specialist focused on business insights.",
            description="Analyzes business data and provides actionable recommendations"
        )
    
    async def process(self, dataframe: pd.DataFrame, question: str) -> str:
        """Process data analysis requests."""
        return self.rag.query(
            dataframe=dataframe,
            question=question,
            agent_id=self.agent_id,
            save_conversation=True
        )

# Usage
chef = MyDataChef()
df = pd.read_csv("sales_data.csv")
result = await chef.process(df, "What are our top performing products?")
print(result)
```

## Core Architecture

AgentChef chefs are built on these core components:

### 1. BaseChef Class
```python
from agentChef import BaseChef

class MyChef(BaseChef):
    def __init__(self):
        super().__init__(
            name="my_chef",
            model_name="llama3.2:3b",
            enable_ui=False  # UI is optional
        )
        self.setup_components()
    
    def setup_components(self):
        """Initialize your chef's components."""
        pass
    
    async def process(self, input_data):
        """Main processing method - implement your logic here."""
        raise NotImplementedError
```

### 2. Available Components

**PandasRAG** - For data analysis and conversation:
```python
from agentChef import PandasRAG

self.rag = PandasRAG(model_name=self.model_name)
agent_id = self.rag.register_agent("specialist", system_prompt="...")
```

**ResearchManager** - For research workflows:
```python
from agentChef import ResearchManager

self.research = ResearchManager(model_name=self.model_name)
```

**OllamaInterface** - For direct LLM interaction:
```python
from agentChef import OllamaInterface

self.ollama = OllamaInterface(model_name=self.model_name)
response = self.ollama.chat(messages=[...])
```

**Dataset Tools** - For data augmentation:
```python
from agentChef import DatasetExpander, DatasetCleaner

self.expander = DatasetExpander(ollama_interface=self.ollama)
self.cleaner = DatasetCleaner(ollama_interface=self.ollama)
```

## Example: Research Chef

```python
from agentChef import BaseChef, ResearchManager, PandasRAG
import asyncio

class ResearchChef(BaseChef):
    def __init__(self):
        super().__init__(name="research_chef", model_name="llama3.2:3b")
        
        # Research capabilities
        self.research_manager = ResearchManager(
            data_dir=f"./chef_data/{self.name}/research",
            model_name=self.model_name
        )
        
        # Conversational capabilities
        self.rag = PandasRAG(
            data_dir=f"./chef_data/{self.name}/conversations", 
            model_name=self.model_name
        )
        
        # Research assistant agent
        self.research_agent = self.rag.register_agent(
            "research_assistant",
            system_prompt="You are a research assistant specializing in academic paper analysis and synthesis.",
            description="Helps with research workflows and paper analysis"
        )
    
    async def research_topic(self, topic: str, max_papers: int = 3) -> str:
        """Research a topic and provide conversational summary."""
        # Do the research
        research_results = await self.research_manager.research_topic(
            topic=topic,
            max_papers=max_papers,
            max_search_results=5
        )
        
        # Create summary DataFrame for analysis
        papers_data = []
        for paper in research_results.get('processed_papers', []):
            papers_data.append({
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': ', '.join(paper.get('authors', [])),
                'url': paper.get('url', '')
            })
        
        if papers_data:
            df = pd.DataFrame(papers_data)
            
            # Get conversational summary
            summary = self.rag.query(
                dataframe=df,
                question=f"Provide a comprehensive summary of the research on '{topic}' based on these papers.",
                agent_id=self.research_agent,
                save_conversation=True
            )
            
            return summary
        else:
            return f"No research papers found for topic: {topic}"
    
    async def ask_about_research(self, question: str) -> str:
        """Ask questions about previous research."""
        # This will have context from previous research calls
        dummy_df = pd.DataFrame([{"context": "research_session"}])
        return self.rag.query(
            dataframe=dummy_df,
            question=question,
            agent_id=self.research_agent,
            save_conversation=True
        )

# Usage
async def main():
    chef = ResearchChef()
    
    # Research a topic
    summary = await chef.research_topic("transformer neural networks", max_papers=2)
    print("Research Summary:", summary)
    
    # Ask follow-up questions
    followup = await chef.ask_about_research("What are the main advantages of transformers?")
    print("Follow-up:", followup)

asyncio.run(main())
```

## Example: File Analysis Chef

```python
from agentChef import BaseChef, PandasRAG
from pathlib import Path
import pandas as pd

class FileAnalysisChef(BaseChef):
    def __init__(self):
        super().__init__(name="file_chef", model_name="llama3.2:3b")
        
        self.rag = PandasRAG(
            data_dir=f"./chef_data/{self.name}",
            model_name=self.model_name
        )
        
        # File analysis specialist
        self.file_agent = self.rag.register_agent(
            "file_analyst",
            system_prompt="You are a file analysis specialist. Analyze file contents and provide insights.",
            description="Specializes in analyzing and understanding file contents"
        )
    
    def ingest_file(self, file_path: str) -> bool:
        """Ingest a file for analysis."""
        path = Path(file_path)
        
        if not path.exists():
            return False
        
        # Read file content based on type
        if path.suffix == '.csv':
            content = f"CSV file with data:\n{pd.read_csv(path).head().to_string()}"
        elif path.suffix in ['.txt', '.md']:
            content = path.read_text(encoding='utf-8')
        else:
            content = f"File: {path.name} (type: {path.suffix})"
        
        # Add to knowledge base
        return self.rag.add_knowledge(
            agent_id=self.file_agent,
            content=content,
            source=f"file:{path.name}",
            metadata={"file_path": str(path), "file_type": path.suffix}
        )
    
    async def analyze_files(self, question: str) -> str:
        """Analyze ingested files."""
        dummy_df = pd.DataFrame([{"context": "file_analysis"}])
        return self.rag.query(
            dataframe=dummy_df,
            question=question,
            agent_id=self.file_agent,
            save_conversation=True
        )

# Usage
chef = FileAnalysisChef()
chef.ingest_file("data.csv")
chef.ingest_file("README.md")

result = await chef.analyze_files("What can you tell me about the files I've uploaded?")
print(result)
```

## Best Practices

### 1. Component Organization
```python
def setup_components(self):
    """Organize components logically."""
    # Core LLM interface
    self.ollama = OllamaInterface(model_name=self.model_name)
    
    # Conversational capabilities
    self.rag = PandasRAG(model_name=self.model_name)
    
    # Specialized components
    self.register_component("expander", DatasetExpander(self.ollama))
    self.register_component("cleaner", DatasetCleaner(self.ollama))
```

### 2. Error Handling
```python
async def process(self, input_data):
    try:
        result = await self.some_operation(input_data)
        self.emit_event("process_complete", result)
        return result
    except Exception as e:
        self.logger.error(f"Processing failed: {e}")
        self.emit_event("process_error", str(e))
        return {"error": str(e)}
```

### 3. Configuration
```python
def __init__(self, config: dict = None):
    config = config or {}
    super().__init__(
        name=config.get("name", "my_chef"),
        model_name=config.get("model", "llama3.2:3b")
    )
```

## Adding UI (Optional)

If you want to add a UI later:

```python
class MyChefWithUI(BaseChef):
    def __init__(self):
        super().__init__(
            name="my_chef",
            enable_ui=True  # Enable UI
        )
    
    def setup_ui(self):
        """Custom UI setup."""
        if self.ui:
            self.ui.add_button("process", "Process Data", self.on_process_click)
    
    def on_process_click(self):
        """Handle UI interactions."""
        asyncio.create_task(self.process(...))
```

## Testing Your Chef

```python
# Simple test script
async def test_chef():
    chef = MyChef()
    
    # Test basic functionality
    result = await chef.process("test input")
    assert result is not None
    
    # Test conversation capability
    if hasattr(chef, 'rag'):
        df = pd.DataFrame({"test": [1, 2, 3]})
        response = chef.rag.query(df, "Analyze this data", agent_id="test")
        assert isinstance(response, str)
    
    print("✅ Chef tests passed!")

asyncio.run(test_chef())
```

## Publishing Your Chef

1. **Package Structure:**
```
my_chef_package/
├── setup.py
├── my_chef/
│   ├── __init__.py
│   └── chef.py
└── README.md
```

2. **setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="my-chef-package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["agentChef>=0.2.8"],
    author="Your Name",
    description="My custom AgentChef implementation"
)
```

3. **Share:**
```bash
pip install build twine
python -m build
python -m twine upload dist/*
```

## Next Steps

- Explore the [examples directory](../src/examples/) for more complex implementations
- Check out [PandasRAG guide](pandas_rag_guide.md) for data analysis features
- Join the community to share your chefs!

## Getting Help

- File issues on GitHub for bugs
- Check existing examples for patterns
- Join our Discord for community support
