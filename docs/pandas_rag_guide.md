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

# Query data - PandasRAG works with any DataFrame
df = pd.read_parquet("sales_data.parquet")  # Efficient Parquet format
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

2. **Python 3.8+** with basic packages:
   ```bash
   pip install pandas pyarrow  # pyarrow needed for Parquet storage
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

# Load your data (supports multiple formats)
df = pd.read_parquet("customer_data.parquet")  # Recommended: Fast Parquet
# df = pd.read_csv("customer_data.csv")        # Also works: CSV
# df = pd.read_json("customer_data.json")      # Also works: JSON

# Progressive conversation with context building
response1 = rag.query(df, "What are the top selling products?", agent_id="data_scientist")
response2 = rag.query(df, "For those top products, what are the profit margins?", agent_id="data_scientist")
response3 = rag.query(df, "Based on our analysis, what should we prioritize?", agent_id="data_scientist")

# Each query builds on previous context automatically!
```

### 3. Interactive Chat Sessions

```python
# Start a persistent chat session
df = pd.read_parquet("sales_data.parquet")
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
# Export all agent data (saved as efficient Parquet files)
exported_files = rag.export_data(
    agent_id="data_scientist",
    export_dir="./backups"
)

# Files exported:
# - conversations: backup/data_scientist_conversations.parquet
# - knowledge: backup/data_scientist_knowledge.parquet
# - profile: backup/data_scientist_profile.json
```

## 🔧 Integration with Pandas Query Engine

PandasRAG integrates with AgentChef's advanced pandas query engine for sophisticated data analysis:

### Using PandasQueryIntegration

```python
from agentChef.core.llamaindex.pandas_query import PandasQueryIntegration

# Initialize with agent-specific prompts
query_engine = PandasQueryIntegration(
    agent_name="data_scientist",
    verbose=True,
    synthesize_response=True
)

# Advanced DataFrame querying
df = pd.read_parquet("complex_dataset.parquet")

# Query with agent-specific analysis prompts
result = query_engine.query_dataframe(
    df=df,
    query="What are the key patterns in customer behavior?",
    prompt_type="conversation_analysis",  # Uses agent-specific prompts
    save_result=True  # Saves to agent's conversation history
)

print(f"Analysis: {result['response']}")
print(f"Pandas Code: {result['pandas_instructions']}")
```

### Agent-Specific Prompt Types

```python
# Different prompt types for different analysis needs
analysis_types = [
    "conversation_analysis",    # For analyzing conversation patterns
    "knowledge_extraction",     # For extracting key knowledge points
    "template_generation",      # For creating conversation templates
    "performance_analysis"      # For evaluating agent performance
]

# Use specific prompt types for targeted analysis
insights = query_engine.query_dataframe(
    df, 
    "Extract key insights for agent training",
    prompt_type="knowledge_extraction"
)
```

### Generating Agent Insights

```python
# Generate multiple insights using agent-specific queries
insights = query_engine.generate_agent_insights(df, num_insights=5)

for insight in insights:
    print(f"Query: {insight['query']}")
    print(f"Insight: {insight['insight']}")
    print(f"Code: {insight['pandas_code']}")
    print("---")
```

### Comparing Datasets

```python
# Compare agent performance across different datasets
df_before = pd.read_parquet("agent_data_before.parquet")
df_after = pd.read_parquet("agent_data_after.parquet")

comparison = query_engine.compare_agent_datasets(
    df1=df_before,
    df2=df_after,
    df1_name="Before Training",
    df2_name="After Training",
    aspects=["conversation_quality", "response_accuracy", "knowledge_coverage"]
)

print(f"Comparison Summary: {comparison['overall_summary']}")
```

## 🗄️ Advanced Storage Integration

PandasRAG uses sophisticated agent-focused storage for persistence and organization:

### ConversationStorage Features

```python
from agentChef.core.storage.conversation_storage import ConversationStorage, KnowledgeEntry

# Access the underlying storage system
storage = rag.storage  # PandasRAG exposes its storage system

# Advanced conversation management
conversations_df = storage.get_conversations("data_scientist", limit=50)
print(f"Retrieved {len(conversations_df)} conversation turns")

# Get detailed agent statistics
stats = storage.get_agent_stats("data_scientist")
print(f"Agent Stats: {stats}")
```

### Knowledge Base Management

```python
# Create structured knowledge entries
knowledge_entries = [
    KnowledgeEntry(
        agent_name="research_assistant",
        entry_id="insight_001",
        topic="data_analysis_patterns",
        content="Users often ask for top-N queries followed by trend analysis",
        knowledge_type="pattern",
        confidence=0.95,
        source="conversation_analysis",
        tags=["user_behavior", "query_patterns"]
    ),
    KnowledgeEntry(
        agent_name="research_assistant", 
        entry_id="template_001",
        topic="response_templates",
        content="When showing data insights, always include specific numbers and percentages",
        knowledge_type="template",
        confidence=1.0,
        source="best_practices",
        tags=["response_quality", "templates"]
    )
]

# Save knowledge with metadata
success = storage.save_knowledge("research_assistant", knowledge_entries)
print(f"Knowledge saved: {success}")

# Query knowledge by type
all_knowledge = storage.load_knowledge("research_assistant")
patterns = [k for k in all_knowledge if k.knowledge_type == "pattern"]
templates = [k for k in all_knowledge if k.knowledge_type == "template"]
```

### Template Management

```python
# Save conversation templates for reuse
template_data = {
    "template_type": "data_analysis_response",
    "structure": [
        {"role": "analysis_summary", "pattern": "Based on the data, I found {key_findings}"},
        {"role": "specific_numbers", "pattern": "The top {n} items are: {ranked_list}"},
        {"role": "recommendation", "pattern": "I recommend focusing on {action_items}"}
    ],
    "usage_context": "responding to top-N queries with actionable insights",
    "quality_score": 0.92
}

storage.save_template("data_scientist", "top_n_analysis", template_data)

# Load and use templates
template = storage.load_template("data_scientist", "top_n_analysis")
print(f"Template structure: {template['structure']}")
```

## 🕷️ Crawler Integration for Data Collection

Integrate PandasRAG with AgentChef's crawling system for automatic data collection:

### Web Crawling Integration

```python
from agentChef.core.crawlers.crawlers_module import AgentDataManager

# Initialize agent data manager
data_manager = AgentDataManager(
    agent_name="research_assistant",
    data_dir="./crawler_data"
)

# Crawl and store web data for agent training
web_results = data_manager.crawl_and_store_for_agent(
    source_type="web",
    source_params={
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "extract_text": True
    }
)

print(f"Web crawl results: {web_results}")

# The data is automatically stored and available to PandasRAG
agent_summary = data_manager.get_agent_knowledge_summary()
print(f"Agent now has {agent_summary['total_knowledge_entries']} knowledge entries")
```

### ArXiv Paper Integration

```python
# Research paper crawling and analysis
arxiv_results = data_manager.crawl_and_store_for_agent(
    source_type="arxiv",
    source_params={
        "query": "machine learning interpretability",
        "max_results": 5
    }
)

# Convert crawled papers to DataFrame for analysis
if not arxiv_results.get("error"):
    # Load papers into PandasRAG for analysis
    papers_df = pd.read_parquet("./crawler_data/arxiv_papers.parquet")
    
    # Analyze papers with PandasRAG
    analysis = rag.query(
        papers_df,
        "What are the main themes in these machine learning papers?",
        agent_id="research_assistant"
    )
    print(f"Paper Analysis: {analysis}")
```

### GitHub Repository Analysis

```python
# Crawl GitHub repositories for code analysis
github_results = data_manager.crawl_and_store_for_agent(
    source_type="github", 
    source_params={
        "repo_url": "https://github.com/scikit-learn/scikit-learn",
        "analysis_type": "code_patterns"
    }
)

# Analyze code patterns with PandasRAG
if not github_results.get("error"):
    print(f"GitHub analysis completed: {github_results['storage_result']}")
```

### Search Integration

```python
# DuckDuckGo search integration
search_results = data_manager.crawl_and_store_for_agent(
    source_type="ddg",
    source_params={
        "query": "pandas data analysis best practices",
        "max_results": 10
    }
)

print(f"Search results stored: {search_results}")
```

## 🔄 Complete Workflow Example

Here's a complete example combining all components:

```python
# Complete PandasRAG workflow with crawlers and advanced querying
from agentChef import PandasRAG
from agentChef.core.crawlers.crawlers_module import AgentDataManager
from agentChef.core.llamaindex.pandas_query import PandasQueryIntegration
import pandas as pd

# 1. Initialize systems
rag = PandasRAG(data_dir="./complete_workflow")
data_manager = AgentDataManager("data_scientist", "./complete_workflow")
query_engine = PandasQueryIntegration("data_scientist")

# 2. Register specialized agent
agent_id = rag.register_agent(
    "data_scientist",
    system_prompt="You are a data scientist who combines web research with data analysis to provide comprehensive insights.",
    description="Specializes in research-backed data analysis"
)

# 3. Collect external data
web_data = data_manager.crawl_and_store_for_agent(
    "web", 
    {"url": "https://kaggle.com/datasets"}
)

arxiv_data = data_manager.crawl_and_store_for_agent(
    "arxiv",
    {"query": "data science methodology", "max_results": 3}
)

# 4. Load and analyze your data
df = pd.read_parquet("your_dataset.parquet")

# 5. Perform sophisticated analysis
insights = query_engine.generate_agent_insights(df, num_insights=3)

# 6. Have contextual conversations
response1 = rag.query(df, "What patterns do you see in this data?", agent_id)
response2 = rag.query(df, "How do these patterns relate to current research?", agent_id)

# 7. Export everything for backup
exported = rag.export_data(agent_id, "./backup")
print(f"Exported: {exported}")

# 8. Get comprehensive agent statistics
stats = rag.storage.get_agent_stats(agent_id)
print(f"Final agent stats: {stats}")
```

## Configuration Options

```python
# Custom settings with advanced options
rag = PandasRAG(
    data_dir="./custom_location",        # Where to store data
    model_name="llama3.2:1b",           # Ollama model to use
    max_history_turns=15,               # Conversation context length
    log_level="DEBUG"                   # Logging level
)

# Advanced storage configuration
rag.storage.conversations_dir  # Access conversation storage
rag.storage.knowledge_dir      # Access knowledge storage
rag.storage.templates_dir      # Access template storage
```

## 🧠 Conversation History & Context

PandasRAG automatically maintains conversation history to provide context-aware responses:

```python
# Load your analysis data
df = pd.read_parquet("quarterly_sales.parquet")

# Each query remembers previous exchanges
response1 = rag.query(df, "What are the top selling products?", agent_id)
response2 = rag.query(df, "For those top products, analyze their profit margins", agent_id)
# The agent remembers "those top products" from the first query!

# Control history behavior
response = rag.query(df, "Fresh analysis", include_history=False)  # No context
response = rag.query(df, "Quick summary", max_history=3)  # Last 3 turns only
```

## Storage Format

PandasRAG uses efficient Parquet storage for fast I/O:

```
{data_dir}/
├── conversations/
│   ├── data_scientist_conversations.parquet    # Fast conversation history
│   └── research_assistant_conversations.parquet
├── knowledge/
│   ├── data_scientist_knowledge.parquet        # Efficient knowledge storage
│   └── research_assistant_knowledge.parquet
├── agents/
│   ├── data_scientist_profile.json             # Agent configuration
│   └── research_assistant_profile.json
└── prompts/
    └── agent_prompts.json                       # System prompts
```

**Why Parquet?**
- **10x faster** than CSV for large datasets
- **Smaller file sizes** with built-in compression
- **Type preservation** (no data type guessing)
- **Column-wise storage** for efficient queries

## Working with Different Data Formats

```python
# PandasRAG accepts any pandas DataFrame
import pandas as pd

# From Parquet (recommended)
df = pd.read_parquet("data.parquet")

# From CSV (also works)
df = pd.read_csv("data.csv")

# From JSON
df = pd.read_json("data.json")

# From database
import sqlite3
conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT * FROM sales", conn)

# All work the same way with PandasRAG
response = rag.query(df, "Analyze this data", agent_id)

# Save results efficiently
df.to_parquet("analysis_results.parquet")  # Fast save
```

## Error Handling

```python
try:
    df = pd.read_parquet("data.parquet")
    response = rag.query(df, "Complex analysis question", "my_agent")
except FileNotFoundError:
    print("Data file not found - check your path")
except Exception as e:
    print(f"Query failed: {e}")
    # System gracefully handles errors and continues working
```

## Performance Tips

1. **Use Parquet format**: 10x faster than CSV for large datasets
2. **Reuse PandasRAG instances**: Create once, use multiple times
3. **Use appropriate models**: `llama3.2:1b` for speed, `llama3.2:3b` for quality
4. **Limit conversation history**: Adjust `max_history_turns` for performance
5. **Install pyarrow**: `pip install pyarrow` for optimal Parquet performance

## Data Format Conversion

```python
# Convert existing CSV data to Parquet for better performance
import pandas as pd

# One-time conversion
df = pd.read_csv("large_dataset.csv")
df.to_parquet("large_dataset.parquet")  # Much faster subsequent loads

# Verify the conversion worked
df_parquet = pd.read_parquet("large_dataset.parquet")
print(f"Rows: {len(df_parquet)}, Columns: {len(df_parquet.columns)}")
```

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

**"pyarrow not found" (for Parquet support)**
```bash
pip install pyarrow
```

**"No conversations found"**
- Check your `data_dir` path
- Ensure `save_conversation=True` in queries
- Verify agent_id is correct

**"Query engine not available"**
```bash
pip install llama-index llama-index-experimental
```

**Performance issues**
- Convert CSV to Parquet: `df.to_parquet("file.parquet")`
- Try smaller model: `llama3.2:1b`
- Reduce `max_history_turns`
- Use `include_history=False` for simple queries

**Storage issues**
- Check data directory permissions
- Ensure enough disk space for Parquet files
- Verify agent names don't contain invalid characters
