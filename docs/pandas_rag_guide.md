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

Make sure you have Ollama installed and running with at least one model (e.g., `llama3.2`).

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

# Get agent information
info = rag.get_agent_info("research_assistant")
print("Agent info:", info)
```

### 2. Data Querying

```python
import pandas as pd

# Load your data
df = pd.read_csv("customer_data.csv")

# Query with natural language
response = rag.query(
    dataframe=df,
    question="What is the customer retention rate by region?",
    agent_id="data_scientist",
    save_conversation=True  # Automatically saves to conversation history
)

print("Analysis:", response)
```

### 3. Conversation Management

```python
# Manual conversation logging
rag.save_conversation("research_assistant", "user", 
                     "What methodology should I use for this study?")
rag.save_conversation("research_assistant", "assistant", 
                     "I recommend a mixed-methods approach...")

# Retrieve conversation history
conversations = rag.get_conversations("research_assistant", limit=10)
print(conversations[['role', 'content', 'timestamp']])

# Start fresh conversations
rag.create_empty_conversation("data_scientist")
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

# Add research findings
rag.add_knowledge(
    agent_id="data_scientist",
    content="Customer churn typically increases by 15% during Q4 due to budget cycles.",
    source="historical_analysis",
    metadata={"category": "business_insights", "date": "2024-01-15"}
)

# Retrieve knowledge base
knowledge = rag.get_knowledge("research_assistant")
for _, item in knowledge.iterrows():
    print(f"[{item['source']}] {item['content']}")
```

### 5. Data Export and Backup

```python
# Export all agent data
exported_files = rag.export_data(
    agent_id="data_scientist",
    export_dir="./backups",
    include_conversations=True,
    include_knowledge=True
)

print("Exported files:", exported_files)
# Output: {
#   'conversations': './backups/data_scientist_conversations.parquet',
#   'knowledge': './backups/data_scientist_knowledge.parquet',
#   'profile': './backups/data_scientist_profile.json'
# }
```

### 6. System Summary

```python
# Get overview of your PandasRAG instance
summary = rag.get_summary()
print(f"Total agents: {summary['total_agents']}")
print(f"Data directory: {summary['data_directory']}")

for agent_id, info in summary['agents'].items():
    print(f"\nAgent: {agent_id}")
    print(f"  Conversations: {info['conversation_turns']}")
    print(f"  Knowledge items: {info['knowledge_items']}")
```

## Advanced Usage

### Custom Data Directory and Model

```python
# Use custom settings
rag = PandasRAG(
    data_dir="/path/to/my/agent_data",
    model_name="llama3.1",  # Use different Ollama model
    log_level="DEBUG"  # Enable debug logging
)
```

### Working with Multiple DataFrames

```python
# Financial data analysis
financial_df = pd.read_csv("financial_reports.csv")
market_df = pd.read_csv("market_data.csv")

# Register a financial analyst
analyst = rag.register_agent(
    "financial_analyst",
    system_prompt="You are a financial analyst with expertise in market trends."
)

# Analyze different datasets
profit_analysis = rag.query(financial_df, "What are the profit margins by quarter?", analyst)
market_analysis = rag.query(market_df, "How do market indices correlate with our performance?", analyst)

# The conversation history now contains both analyses
full_conversation = rag.get_conversations(analyst)
```

### Integration with Existing Workflows

```python
def analyze_weekly_reports():
    """Weekly automated analysis workflow"""
    rag = PandasRAG(data_dir="./weekly_analysis")
    
    # Load this week's data
    weekly_data = pd.read_csv(f"data/week_{get_current_week()}.csv")
    
    # Get or create weekly analyst
    if "weekly_analyst" not in rag.list_agents():
        rag.register_agent(
            "weekly_analyst",
            system_prompt="You provide weekly business insights and trend analysis."
        )
    
    # Perform analysis
    insights = rag.query(
        weekly_data, 
        "Provide a summary of this week's key metrics and trends.",
        agent_id="weekly_analyst"
    )
    
    # Save additional context
    rag.add_knowledge(
        "weekly_analyst",
        f"Week {get_current_week()} analysis completed. Key findings: {insights[:200]}...",
        source="automated_analysis"
    )
    
    return insights

# Use in your existing scripts
weekly_insights = analyze_weekly_reports()
```

## Storage Format

PandasRAG uses efficient Parquet storage for all data:

- **Conversations**: `{data_dir}/conversations/{agent_id}_conversations.parquet`
- **Knowledge**: `{data_dir}/knowledge/{agent_id}_knowledge.parquet`  
- **Agent Profiles**: `{data_dir}/agents/{agent_id}_profile.json`

This format is:
- Fast to read/write
- Efficient storage
- Compatible with pandas, Spark, and other data tools
- Version-controllable (with git-lfs)

## Error Handling

```python
try:
    response = rag.query(df, "Complex analysis question", "my_agent")
except Exception as e:
    print(f"Query failed: {e}")
    # Fallback to manual conversation entry
    rag.save_conversation("my_agent", "user", "Complex analysis question")
    rag.save_conversation("my_agent", "system", f"Error occurred: {e}")
```

## Integration with AgentChef UI

PandasRAG is fully compatible with AgentChef's web UI. Agents and data created with PandasRAG will appear in the UI, and vice versa.

```python
# Create agents in script
rag = PandasRAG()
rag.register_agent("script_agent", system_prompt="Created via script")

# Later, use AgentChef UI to interact with the same data
# The script_agent will be available in the web interface
```

## Performance Tips

1. **Reuse PandasRAG instances**: Create once, use multiple times
2. **Batch operations**: Group multiple queries when possible
3. **Limit conversation history**: Use `limit` parameter when retrieving conversations
4. **Export periodically**: Regular exports ensure data backup

## Next Steps

- Explore the full AgentChef documentation for advanced features
- Check out `examples/pandas_rag_example.py` for a complete working example
- See `examples/modular_system_demo.py` for integration with crawlers and data management
- Use the AgentChef web UI for visual interaction with your agents and data

## 🧠 Conversation History & Context

PandasRAG automatically maintains conversation history to provide context-aware responses, similar to how ChatGPT remembers your conversation.

### Basic History Usage

```python
from agentChef import PandasRAG
import pandas as pd

# Initialize with history support
rag = PandasRAG(max_history_turns=10)  # Keep last 10 conversation turns

# Register an agent
analyst = rag.register_agent("data_analyst", 
                           system_prompt="You are a data analyst who builds on previous insights.")

# Load your data
df = pd.read_csv("sales_data.csv")

# First query establishes context
response1 = rag.query(df, "What are the top selling products?", analyst)

# Second query builds on the first - the agent remembers!
response2 = rag.query(df, "For those top products, analyze their profit margins", analyst)

# Third query continues the analysis
response3 = rag.query(df, "Based on our findings, what's your recommendation?", analyst)
```

### Interactive Chat Sessions

```python
# Start a persistent chat session
chat = rag.chat_with_data(df, agent_id="analyst")

# Natural conversation flow
chat.ask("What patterns do you see in the sales data?")
chat.ask("Focus on the seasonal trends you mentioned")
chat.ask("How should this influence our Q4 strategy?")

# Get conversation summary
summary = chat.get_summary()
print("Conversation Summary:", summary)

# Save the session for later
chat.save_session("quarterly_analysis_2025")
```

### Controlling History Behavior

```python
# Query without history context (for fresh analysis)
fresh_response = rag.query(df, "Analyze this data", 
                          include_history=False)

# Limit history context for specific queries
limited_response = rag.query(df, "Quick summary", 
                           max_history=3)  # Only last 3 turns

# Clear conversation history
chat.clear_history()
```

### How Conversation History Works

1. **Automatic Context**: Each query includes relevant previous exchanges
2. **Smart Truncation**: Only recent, relevant conversation is included
3. **Agent Memory**: Each agent maintains separate conversation history
4. **Persistent Storage**: Conversations are saved and can be resumed
5. **Context Building**: Responses build naturally on previous insights

### Example: Progressive Analysis

```python
# This conversation builds context progressively:

# Q1: "What are our best-selling products?"
# A1: "Based on the data, laptops, phones, and tablets are top sellers..."

# Q2: "What about profit margins on those items?"  
# A2: "Looking at the top sellers we identified (laptops, phones, tablets), 
#      the profit margins are..."

# Q3: "Should we expand the laptop line?"
# A3: "Given our earlier analysis showing laptops as top sellers with 
#      strong margins, expansion would be recommended because..."
```

The agent naturally references "top sellers we identified" and "our earlier analysis" because it has conversation context!

### Memory Management

```python
# Configure memory settings
rag = PandasRAG(
    max_history_turns=15,        # Keep 15 conversation turns
    data_dir="./agent_memory"    # Persistent storage location
)

# Check conversation history
conversations = rag.get_conversations("agent_name", limit=10)
print(conversations[['role', 'content', 'timestamp']])

# Get conversation summary
summary = rag.get_conversation_summary("agent_name", num_exchanges=5)
```
