# PandasRAG - Script-Friendly AgentChef Interface

`PandasRAG` is a simple, script-friendly interface for AgentChef that allows you to work with agent-centric data storage, pandas querying, and conversation management without needing the web UI.

## Quick Start

```python
from agentChef import PandasRAG
import pandas as pd

# Initialize
rag = PandasRAG(data_dir="./my_data")

# Register an agent
agent_id = rag.register_agent("data_analyst", 
                             system_prompt="You are a skilled data analyst.")

# Query data
df = pd.read_csv("sales_data.csv")
response = rag.query(df, "What are the top selling products?", agent_id=agent_id)
print(response)
```

## Installation

Install AgentChef via pip:

```bash
pip install agentChef
```

**Prerequisites:**
1. **Ollama** installed and running:
   ```bash
   # Install from https://ollama.ai/
   ollama pull llama3.2:3b
   ollama serve  # Keep this running
   ```

## Core Features

### 1. Agent Management

```python
from agentChef import PandasRAG

rag = PandasRAG()

# Register agents with specific roles
research_agent = rag.register_agent(
    "research_assistant",
    system_prompt="You are a research assistant specializing in academic papers.",
    description="Helps with literature review and research analysis"
)

data_agent = rag.register_agent(
    "data_scientist", 
    system_prompt="You are a data scientist focused on statistical analysis.",
    description="Provides statistical insights and data interpretations"
)

# List all agents
agents = rag.list_agents()
print("Available agents:", agents)
```

### 2. Data Querying with Conversation History

```python
import pandas as pd

# Load your data
df = pd.read_csv("customer_data.csv")

# Progressive conversation with context building
response1 = rag.query(df, "What are the top selling products?", agent_id="data_scientist")
response2 = rag.query(df, "For those top products, what are the profit margins?", agent_id="data_scientist")
response3 = rag.query(df, "Based on our analysis, what should we prioritize?", agent_id="data_scientist")

# Each query builds on previous context automatically!
```

### 3. Interactive Chat Sessions

```python
# Start a persistent chat session
chat = rag.chat_with_data(df, agent_id="data_scientist")

# Natural conversation flow
chat.ask("What patterns do you see in the sales data?")
chat.ask("Focus on the seasonal trends you mentioned")
chat.ask("How should this influence our Q4 strategy?")

# Get conversation summary
summary = chat.get_summary()
print("Conversation Summary:", summary)
```

### 4. Knowledge Management

```python
# Add domain knowledge
rag.add_knowledge(
    agent_id="research_assistant",
    content="The university follows APA style guidelines for all publications.",
    source="institutional_policy",
    metadata={"category": "formatting", "priority": "high"}
)

# Retrieve knowledge base
knowledge = rag.get_knowledge("research_assistant")
```

### 5. Data Export and Backup

```python
# Export all agent data
exported_files = rag.export_data(
    agent_id="data_scientist",
    export_dir="./backups"
)
```

## Configuration Options

```python
# Custom settings
rag = PandasRAG(
    data_dir="./custom_location",        # Where to store data
    model_name="llama3.2:1b",           # Ollama model to use
    max_history_turns=15,               # Conversation context length
    log_level="DEBUG"                   # Logging level
)
```

## 🧠 Conversation History & Context

PandasRAG automatically maintains conversation history to provide context-aware responses:

```python
# Each query remembers previous exchanges
response1 = rag.query(df, "What are the top selling products?", agent_id)
response2 = rag.query(df, "For those top products, analyze their profit margins", agent_id)
# The agent remembers "those top products" from the first query!

# Control history behavior
response = rag.query(df, "Fresh analysis", include_history=False)  # No context
response = rag.query(df, "Quick summary", max_history=3)  # Last 3 turns only
```

## Storage Format

PandasRAG uses efficient Parquet storage:
- **Conversations**: `{data_dir}/conversations/{agent_id}_conversations.parquet`
- **Knowledge**: `{data_dir}/knowledge/{agent_id}_knowledge.parquet`  
- **Agent Profiles**: `{data_dir}/agents/{agent_id}_profile.json`

## Error Handling

```python
try:
    response = rag.query(df, "Complex analysis question", "my_agent")
except Exception as e:
    print(f"Query failed: {e}")
    # System gracefully handles errors and continues working
```

## Performance Tips

1. **Reuse PandasRAG instances**: Create once, use multiple times
2. **Use appropriate models**: `llama3.2:1b` for speed, `llama3.2:3b` for quality
3. **Limit conversation history**: Adjust `max_history_turns` for performance
4. **Export periodically**: Regular exports ensure data backup

## Next Steps

- Check out the [complete examples](../src/examples/) for advanced usage
- See [custom chef guide](custom_chef_guide.md) for building your own agents
- Use the [file ingestion examples](../src/examples/personal_assistant_rag_example.py) for document workflows

## Troubleshooting

**"Ollama not available"**
```bash
ollama serve
ollama pull llama3.2:3b
```

**"No conversations found"**
- Check your `data_dir` path
- Ensure `save_conversation=True` in queries
- Verify agent_id is correct

**Performance issues**
- Try smaller model: `llama3.2:1b`
- Reduce `max_history_turns`
- Use `include_history=False` for simple queries
