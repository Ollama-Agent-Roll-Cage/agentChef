# AgentChef Modular Architecture

This document explains the new modular architecture implemented for AgentChef, which provides better abstraction and agent-specific functionality.

## Overview

The modular system introduces three key components that work together to provide agent-focused functionality:

1. **Agent Prompt Manager** - Manages agent-specific prompts and templates
2. **Conversation Storage** - Agent-focused storage for conversations, knowledge, and templates  
3. **Abstract Query Engine** - Abstracted pandas query engine with agent-specific prompt support

## Key Improvements

### 1. Agent-Centric Design

Instead of being focused on web crawling, the system now centers around agents and their specific needs:

- Each agent can have its own prompts, knowledge base, and conversation history
- Storage is organized by agent rather than by data source
- Query engines adapt to different agent contexts and expertise domains

### 2. Prompt Management System

The `AgentPromptManager` provides:

- **Template-based prompts** with variable substitution
- **Agent-specific prompt customization**
- **Dynamic prompt generation** via callable functions
- **Persistent prompt storage** in JSON format

```python
from agentChef.core.prompts.agent_prompt_manager import AgentPromptManager

# Initialize prompt manager
pm = AgentPromptManager()

# Register an agent with custom prompts
agent_config = {
    "domain": "research",
    "personality": "analytical", 
    "prompts": {
        "paper_analysis": "You are analyzing research papers for {agent_name}..."
    }
}
pm.register_agent("research_agent", agent_config)

# Get a formatted prompt
prompt = pm.get_prompt("research_agent", "paper_analysis", 
                      query="What are the main findings?")
```

### 3. Conversation Storage System

The `ConversationStorage` class provides:

- **Agent-specific data organization**
- **Structured conversation storage** with metadata
- **Knowledge base management** for each agent
- **Template storage** for reusable conversation patterns

```python
from agentChef.core.storage.conversation_storage import ConversationStorage, KnowledgeEntry

# Initialize storage
storage = ConversationStorage()

# Save a conversation
conversation = [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is..."}
]
conv_id = storage.save_conversation("my_agent", conversation)

# Save knowledge entries
knowledge = [
    KnowledgeEntry(
        agent_name="my_agent",
        entry_id="ml_001", 
        topic="machine learning",
        content="ML is a subset of AI...",
        knowledge_type="definition"
    )
]
storage.save_knowledge("my_agent", knowledge)
```

### 4. Abstract Query Engine

The `PandasQueryIntegration` class now supports:

- **Agent-specific prompts** for different analysis contexts
- **Automatic conversation storage** of query results
- **Customizable prompt types** (conversation_analysis, knowledge_extraction, etc.)
- **Agent insights generation** based on domain expertise

```python
from agentChef.core.llamaindex.pandas_query import PandasQueryIntegration

# Create agent-specific query engine
query_engine = PandasQueryIntegration(agent_name="data_scientist")

# Query with agent-specific prompts
result = query_engine.query_dataframe(
    df, 
    "What patterns do you see in this customer data?",
    prompt_type="data_analysis",
    save_result=True
)

# Generate agent-specific insights
insights = query_engine.generate_agent_insights(df, num_insights=5)
```

### 5. Agent Data Manager

The `AgentDataManager` provides a unified interface:

- **Integrated crawling and storage** workflow
- **Agent-specific data processing**
- **Knowledge extraction** from crawled content
- **Automatic conversation generation** from crawled data

```python
from agentChef.core.crawlers.crawlers_module import AgentDataManager

# Initialize for specific agent
manager = AgentDataManager("research_agent")

# Crawl and store data in agent-specific format
result = manager.crawl_and_store_for_agent("arxiv", {
    "query": "machine learning transformers",
    "max_results": 5
})

# Get knowledge summary
summary = manager.get_agent_knowledge_summary()
```

## File Structure

```
src/agentChef/core/
├── prompts/
│   └── agent_prompt_manager.py    # Agent-specific prompt management
├── storage/  
│   └── conversation_storage.py    # Agent-focused storage system
├── llamaindex/
│   └── pandas_query.py           # Abstract query engine with agent support
└── crawlers/
    └── crawlers_module.py        # Updated with agent data manager

examples/
└── modular_system_demo.py        # Complete demonstration
```

## Migration Guide

### From Old System

**Before:**
```python
# Old approach - crawler-focused
pandas_query = PandasQueryIntegration()
result = pandas_query.query_dataframe(df, "analyze this")
```

**After:**
```python
# New approach - agent-focused
query_engine = PandasQueryIntegration(agent_name="my_agent")
result = query_engine.query_dataframe(
    df, 
    "analyze this", 
    prompt_type="conversation_analysis",
    save_result=True
)
```

### Key Changes

1. **Import paths updated** to use new modular structure
2. **Agent name required** for most operations
3. **Prompt types** replace custom instructions
4. **Automatic storage** of conversations and knowledge
5. **Agent-specific configuration** instead of global settings

## Benefits

### For Agent Development

- **Specialized prompts** for different agent types and domains
- **Persistent knowledge** that grows with each interaction
- **Conversation history** for context and learning
- **Template reuse** for consistent agent behavior

### For Data Management

- **Better organization** by agent rather than data source
- **Rich metadata** for conversations and knowledge
- **Query tracking** and result storage
- **Cross-agent comparison** capabilities

### For System Architecture

- **Modular design** allows independent component development
- **Abstract interfaces** enable easy extension and customization
- **Agent-centric** approach aligns with actual use cases
- **Storage efficiency** through parquet format and metadata separation

## Example Use Cases

### Research Agent
- Crawl academic papers
- Extract key findings and methodologies  
- Generate research summaries
- Build domain-specific knowledge base

### Customer Service Agent
- Analyze customer conversation data
- Identify common issues and solutions
- Generate response templates
- Track conversation quality metrics

### Data Analysis Agent  
- Process various datasets
- Generate insights and visualizations
- Build feature engineering knowledge
- Create analysis templates

## Running the Demo

```bash
cd examples
python modular_system_demo.py
```

This will demonstrate all the new functionality including agent registration, prompt management, storage, and query engine capabilities.

## Future Enhancements

- **Multi-agent conversations** between different specialized agents
- **Knowledge sharing** between related agents
- **Automated prompt optimization** based on conversation outcomes
- **Agent performance metrics** and improvement tracking
- **Visual interfaces** for agent management and monitoring
