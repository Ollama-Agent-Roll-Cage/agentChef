<p align="center">
  <img src="https://github.com/Leoleojames1/agentChef/blob/86caca2e8b300f3fdee64ee56f02e318992a8f92/assets/Untitled-removebg-preview.png" alt="agentChef logo" width="250"/>
</p>
<p align="center">
  <a href="https://ko-fi.com/theborch"><img src="https://github.com/Leoleojames1/agentChef/blob/7b3e5e86589add6a792acfc1de4538d82918942f/assets/Discord%20button.png" height="48"></a>
  <a href="https://discord.gg/dAzSYcnpdF"><img src="assets/Discord button.png" height="48"></a>
</p>

# AgentChef

AgentChef is a comprehensive Python library for AI research, dataset generation, and conversation management using large language models. It provides tools for crawling, processing, and analyzing data sources including web pages, ArXiv papers, and GitHub repositories, and generating high-quality conversation datasets for AI training.

[![PyPI version](https://badge.fury.io/py/agentChef.svg)](https://badge.fury.io/py/agentChef)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installing from PyPI](#installing-from-pypi)
  - [Development Installation](#development-installation)
  - [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
  - [Research and Data Collection](#research-and-data-collection)
  - [Conversation Generation](#conversation-generation)
  - [Dataset Expansion](#dataset-expansion)
  - [Dataset Cleaning](#dataset-cleaning)
  - [Web Crawling and Paper Analysis](#web-crawling-and-paper-analysis)
  - [GitHub Repository Analysis](#github-repository-analysis)
  - [Using the Unified System (UDRAGS)](#using-the-unified-system-udrags)
- [Advanced Usage](#advanced-usage)
  - [Working with Custom Dataset Formats](#working-with-custom-dataset-formats)
  - [Integration with Pandas Query Interface](#integration-with-pandas-query-interface)
- [Command-line Interface](#command-line-interface)
- [Building Custom Workflows](#building-custom-workflows)
- [Contributing](#contributing)
- [License](#license)

## Features

- **AI-Powered Research**: Search and process web content, ArXiv papers, and GitHub repositories
- **Conversation Generation**: Create synthetic conversation data from research papers and other content
- **Dataset Expansion**: Generate variations and paraphrases to expand training datasets
- **Quality Control**: Clean and validate generated datasets to ensure high quality
- **Integrated Ollama Support**: Seamless integration with local Ollama models
- **Pandas Integration**: Natural language querying of pandas DataFrames
- **Flexible Output Formats**: Export data in JSONL, Parquet, CSV, and other formats
- **Command-line Interface**: Full-featured CLI for research and dataset generation workflows
- **Optional GUI**: User-friendly interface for non-technical users (requires PyQt6)

## Overview

Agent Chef provides an end-to-end solution for:

- Researching topics from multiple sources (ArXiv, web search, GitHub repositories)
- Generating high-quality conversation datasets
- Expanding and augmenting existing datasets
- Analyzing and cleaning data to ensure quality
- Querying and analyzing datasets using natural language

Built on top of local Ollama models, Agent Chef enables researchers and developers to work with conversation data efficiently without requiring external API access.

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and configured with models of your choice

### Installing from PyPI

```bash
# Install the base package
pip install agentchef

# Install optional dependencies for UI
pip install 'agentchef[ui]'

# Install dependencies for advanced analysis
pip install 'agentchef[analysis]'

# Install all dependencies
pip install 'agentchef[full]'
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/Leoleojames1/agentchef.git
cd agentchef

# Create and activate a virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.8+
- Ollama (for LLM-based features)
- Optional dependencies:
  - PyQt6 (for GUI features)
  - LlamaIndex (for advanced data querying)
  - pyarrow (for Parquet support)

## Core Components

AgentChef consists of several core modules:

- **conversation_generator.py**: Generate conversations from content
- **dataset_expander.py**: Expand datasets with variations
- **dataset_cleaner.py**: Clean and validate generated datasets
- **crawlers_module.py**: Web, ArXiv, and GitHub data collection
- **ollama_interface.py**: Interface to Ollama models
- **pandas_query.py**: Natural language querying for pandas DataFrames
- **udrags.py**: Unified Dataset Research, Augmentation, & Generation System

## Quick Start

```python
from agentChef.conversation_generator import OllamaConversationGenerator
from agentChef.dataset_expander import DatasetExpander
from agentChef.ollama_interface import OllamaInterface

# Create a shared Ollama interface for consistent access across components
ollama_interface = OllamaInterface(model_name="llama3")

# Initialize conversation generator with the shared interface
generator = OllamaConversationGenerator(
    model_name="llama3", 
    ollama_interface=ollama_interface
)

# Generate a conversation about a topic
conversation = generator.generate_conversation(
    content="Attention mechanisms have become an integral part of compelling sequence modeling...",
    num_turns=3,
    conversation_context="AI research"
)

# Initialize dataset expander with the same interface
expander = DatasetExpander(
    ollama_interface=ollama_interface, 
    output_dir="./expanded_data"
)

# Expand the generated conversation
expanded_conversations = expander.expand_conversation_dataset(
    conversations=[conversation],
    expansion_factor=3,
    static_fields={'human': True, 'gpt': False}  # Keep human questions static
)

# Save the expanded conversations
expander.save_conversations_to_jsonl(expanded_conversations, "expanded_conversations")

# Analyze the expanded dataset
analysis = expander.analyze_expanded_dataset([conversation], expanded_conversations)
print(analysis['basic_statistics'])
```

## Usage Examples

### Research and Data Collection

```python
import asyncio
from agentChef.udrags import ResearchManager

async def research_topic():
    # Initialize the research manager
    manager = ResearchManager(model_name="llama3")
    
    # Research a topic
    results = await manager.research_topic(
        topic="Transformer neural networks",
        max_papers=3,
        max_search_results=5,
        include_github=True,
        github_repos=["https://github.com/huggingface/transformers"]
    )
    
    # Print the research summary
    print(results["summary"])
    
    # Access ArXiv paper information
    for paper in results["arxiv_papers"]:
        print(f"Paper: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Abstract: {paper['abstract'][:200]}...\n")

# Run the async function
asyncio.run(research_topic())
```

### Conversation Generation

```python
from agentChef.ollama_interface import OllamaInterface
from agentChef.conversation_generator import OllamaConversationGenerator

# Initialize the Ollama interface
ollama_interface = OllamaInterface(model_name="llama3")

# Initialize the conversation generator
generator = OllamaConversationGenerator(model_name="llama3", ollama_interface=ollama_interface)

# Sample content to generate a conversation about
content = """
Attention mechanisms have become an integral part of compelling sequence modeling
and transduction models in various tasks, allowing modeling of dependencies without
regard to their distance in the input or output sequences. In this paper we present the
Transformer, a model architecture eschewing recurrence and instead relying entirely
on an attention mechanism to draw global dependencies between input and output.
"""

# Generate a conversation with 3 turns
conversation = generator.generate_conversation(
    content=content,
    num_turns=3,
    conversation_context="AI research",
    hedging_level="balanced"
)

# Print the formatted conversation
import json
print(json.dumps(conversation, indent=2))

# Generate a hedged response to a specific question
hedged_response = generator.generate_hedged_response(
    prompt="Explain how transformer models work in simple terms",
    hedging_profile="balanced",
    knowledge_level="high",
    subject_expertise="machine learning"
)

print("\nHedged Response:")
print(hedged_response)
```

### Dataset Expansion

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.dataset_expander import DatasetExpander
from agentChef.conversation_generator import OllamaConversationGenerator

async def expand_dataset():
    # Initialize components
    ollama_interface = OllamaInterface(model_name="llama3")
    generator = OllamaConversationGenerator(model_name="llama3", ollama_interface=ollama_interface)
    expander = DatasetExpander(ollama_interface=ollama_interface, output_dir="./expanded_data")
    
    # Sample paper content
    paper_content = """
    Attention mechanisms have become an integral part of compelling sequence modeling
    and transduction models in various tasks, allowing modeling of dependencies without
    regard to their distance in the input or output sequences. In this paper we present the
    Transformer, a model architecture eschewing recurrence and instead relying entirely
    on an attention mechanism to draw global dependencies between input and output.
    """
    
    # Generate and expand conversations
    orig_conversations, expanded_conversations = await expander.generate_conversations_from_paper(
        paper_content=paper_content,
        conversation_generator=generator,
        num_chunks=2,
        num_turns=3,
        expansion_factor=2,
        static_fields={'human': True, 'gpt': False},  # Keep human questions static, vary gpt responses
        reference_fields=['human']  # Use human questions as reference when generating gpt responses
    )
    
    # Save in multiple formats
    output_files = expander.convert_to_multi_format(
        expanded_conversations, 
        "transformer_paper_conversations",
        formats=['jsonl', 'parquet', 'csv']
    )
    
    print(f"Original conversations: {len(orig_conversations)}")
    print(f"Expanded conversations: {len(expanded_conversations)}")
    print(f"Generated files: {output_files}")
    
    # Analyze the expanded dataset
    analysis = expander.analyze_expanded_dataset(orig_conversations, expanded_conversations)
    print("Expansion analysis:", analysis["basic_statistics"])

# Run the async function
asyncio.run(expand_dataset())
```

### Dataset Cleaning

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.dataset_cleaner import DatasetCleaner

async def clean_dataset():
    # Initialize the Ollama interface
    ollama_interface = OllamaInterface(model_name="llama3")
    
    # Initialize the dataset cleaner
    cleaner = DatasetCleaner(ollama_interface=ollama_interface, output_dir="./cleaned_output")
    
    # Sample conversations (original and expanded)
    original_conversations = [
        # Example original conversation
        [
            {"from": "human", "value": "What are the key components of a transformer model?"},
            {"from": "gpt", "value": "The key components of a transformer model include self-attention mechanisms, feed-forward neural networks, positional encodings, and layer normalization."}
        ]
    ]
    
    expanded_conversations = [
        # Example expanded conversation with some quality issues
        [
            {"from": "human", "value": "Can you tell me about the main parts of transformer architectures?"},
            {"from": "gpt", "value": "The transformer architecture have several key components. They is including the self-attention mechanism, feed-forward networks, position encodings, and normalization layers."}
        ]
    ]
    
    # Analyze the dataset
    analysis = await cleaner.analyze_dataset(original_conversations, expanded_conversations)
    print("Analysis results:")
    print(f"Issues found: {analysis['issues_by_type']}")
    
    # Clean the dataset
    cleaned_conversations = await cleaner.clean_dataset(
        original_conversations=original_conversations,
        expanded_conversations=expanded_conversations,
        cleaning_criteria={
            "fix_hallucinations": True,
            "normalize_style": True,
            "correct_grammar": True,
            "ensure_coherence": True
        }
    )
    
    print("\nCleaned conversation:")
    import json
    print(json.dumps(cleaned_conversations[0], indent=2))

# Run the async function
asyncio.run(clean_dataset())
```

### Web Crawling and Paper Analysis

```python
import asyncio
from agentChef.crawlers_module import WebCrawler, ArxivSearcher, DuckDuckGoSearcher

async def crawl_and_analyze():
    # Initialize components
    web_crawler = WebCrawler()
    arxiv_searcher = ArxivSearcher()
    ddg_searcher = DuckDuckGoSearcher()
    
    # Fetch content from a web page
    url = "https://example.com"
    html_content = await web_crawler.fetch_url_content(url)
    if html_content:
        text_content = await web_crawler.extract_text_from_html(html_content)
        print(f"Web page content: {text_content[:200]}...\n")
    
    # Fetch a paper from ArXiv
    try:
        paper_info = await arxiv_searcher.fetch_paper_info("1706.03762")  # Attention Is All You Need
        formatted_paper = await arxiv_searcher.format_paper_for_learning(paper_info)
        print("ArXiv Paper Information:")
        print(f"Title: {paper_info['title']}")
        print(f"Authors: {', '.join(paper_info['authors'])}")
        print(f"Abstract: {paper_info['abstract'][:200]}...\n")
    except Exception as e:
        print(f"Error fetching ArXiv paper: {e}")
    
    # Perform a DuckDuckGo search
    search_results = await ddg_searcher.text_search("transformer neural networks", max_results=3)
    print("DuckDuckGo Search Results:")
    print(search_results)

# Run the async function
asyncio.run(crawl_and_analyze())
```

### GitHub Repository Analysis

```python
import asyncio
from agentChef.crawlers_module import GitHubCrawler

async def analyze_github_repo():
    # Initialize the GitHub crawler
    github_crawler = GitHubCrawler()
    
    # Get a summary of a repository
    repo_url = "https://github.com/huggingface/transformers"
    try:
        repo_summary = await github_crawler.get_repo_summary(repo_url)
        print("GitHub Repository Summary:")
        print(repo_summary)
        
        # Query the repository content
        query_result = await github_crawler.query_repo_content(
            repo_url=repo_url,
            query="Find Python files related to attention mechanisms"
        )
        print("\nQuery Results:")
        print(query_result)
    except Exception as e:
        print(f"Error analyzing GitHub repository: {e}")

# Run the async function
asyncio.run(analyze_github_repo())
```

### Using the Unified System (UDRAGS)

```python
import asyncio
from agentChef.udrags import ResearchManager

async def unified_research_and_generation():
    # Initialize the research manager
    manager = ResearchManager(model_name="llama3")
    
    # Define a progress callback function
    def progress_callback(message):
        print(f"Progress: {message}")
    
    # Step 1: Research a topic
    research_results = await manager.research_topic(
        topic="Transformer neural networks",
        max_papers=3,
        callback=progress_callback
    )
    
    # Step 2: Generate conversation dataset from research
    dataset_results = await manager.generate_conversation_dataset(
        num_turns=3,
        expansion_factor=2,
        clean=True,
        callback=progress_callback
    )
    
    # Print results
    print(f"Generated {len(dataset_results['conversations'])} original conversations")
    print(f"Generated {len(dataset_results['expanded_conversations'])} expanded conversations")
    print(f"Generated {len(dataset_results['cleaned_conversations'])} cleaned conversations")
    print(f"Output saved to: {dataset_results.get('output_path', 'unknown')}")

# Run the async function
asyncio.run(unified_research_and_generation())
```

## Advanced Usage

### Working with Custom Dataset Formats

```python
from agentChef.dataset_expander import DatasetExpander
from agentChef.ollama_interface import OllamaInterface
import pandas as pd

# Initialize components
ollama_interface = OllamaInterface(model_name="llama3")
expander = DatasetExpander(ollama_interface=ollama_interface)

# Convert conversations to DataFrame
conversations = [
    # Example conversations
    [
        {"from": "human", "value": "What are transformer models?"},
        {"from": "gpt", "value": "Transformer models are a type of neural network architecture..."}
    ],
    [
        {"from": "human", "value": "Explain attention mechanisms."},
        {"from": "gpt", "value": "Attention mechanisms allow models to focus on different parts..."}
    ]
]

# Convert to DataFrame
df = expander.convert_conversations_to_dataframe(conversations)
print("Conversation DataFrame:")
print(df.head())

# Convert back to conversation format
# (This would require a custom function, not directly provided by AgentChef)

# Save in multiple formats
output_files = expander.convert_to_multi_format(
    conversations,
    "custom_conversations",
    formats=['jsonl', 'parquet', 'csv', 'df']
)

# Access the DataFrame directly
dataframe = output_files.get('df')
```

### Integration with Pandas Query Interface

```python
import pandas as pd
from agentChef.pandas_query import PandasQueryIntegration, OllamaLlamaIndexIntegration

# Sample DataFrame
df = pd.DataFrame({
    "city": ["Toronto", "Tokyo", "Berlin", "Sydney", "New York"],
    "population": [2930000, 13960000, 3645000, 5312000, 8419000],
    "country": ["Canada", "Japan", "Germany", "Australia", "USA"],
    "continent": ["North America", "Asia", "Europe", "Oceania", "North America"]
})

# Using OpenAI-based integration
try:
    pandas_query = PandasQueryIntegration(openai_api_key="your-api-key")
    
    # Execute a natural language query
    result = pandas_query.query_dataframe(df, "What is the city with the highest population?")
    print(f"Query result: {result['response']}")
    print(f"Pandas code: {result['pandas_instructions']}")
    
    # Generate insights from the DataFrame
    insights = pandas_query.generate_dataset_insights(df, num_insights=2)
    for insight in insights:
        print(f"\nQuery: {insight['query']}")
        print(f"Insight: {insight['insight']}")
except ImportError:
    print("LlamaIndex not installed")

# Using Ollama-based integration
try:
    ollama_query = OllamaLlamaIndexIntegration(ollama_model="llama3")
    
    # Execute a query using Ollama
    result = ollama_query.query_dataframe_with_ollama(df, "What is the city with the highest population?")
    print(f"Ollama query result: {result['response']}")
    print(f"Pandas code: {result['pandas_code']}")
except ImportError:
    print("Ollama integration not available")
```

## Command-line Interface

AgentChef provides a comprehensive command-line interface through the `udrags.py` module:

### Research Mode

```bash
python -m agentChef.udrags --mode research --topic "Transformer neural networks" --max-papers 5 --max-search 10
```

### Generate Mode

```bash
python -m agentChef.udrags --mode generate --topic "Transformer neural networks" --turns 3 --expand 3 --clean --format jsonl
```

### Process Mode

```bash
python -m agentChef.udrags --mode process --input papers_dir/ --turns 3 --expand 3 --clean --format all
```

### UI Mode (if PyQt6 is installed)

```bash
python -m agentChef.udrags --mode ui
```

## Building Custom Workflows

AgentChef is designed to be modular, allowing you to build custom workflows by combining different components. The UDRAGS system (Unified Dataset Research, Augmentation, & Generation System) itself is an example of a custom workflow built on top of AgentChef's core components.

Here's an example of how you can create your own research-generate-augment-analyze-clean pipeline, mirroring the UDRAGS approach:

### 1. Research Phase

First, collect and process research data:

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.crawlers_module import ArxivSearcher, DuckDuckGoSearcher

async def research_phase(topic):
    # Set up components with shared interface
    ollama = OllamaInterface(model_name="llama3")
    arxiv = ArxivSearcher()
    search = DuckDuckGoSearcher()
    
    # Collect data from multiple sources
    search_results = await search.text_search(topic, max_results=5)
    print(f"Web search completed: {len(search_results)} results")
    
    # Get relevant papers (using a sample ArXiv ID for demonstration)
    try:
        paper_id = "2201.08239"  # You might use topic keywords to find relevant IDs
        paper = await arxiv.fetch_paper_info(paper_id)
        formatted_paper = await arxiv.format_paper_for_learning(paper)
        print(f"Retrieved paper: {paper['title']}")
        
        # Return collected research
        return {
            "search_results": search_results,
            "papers": [formatted_paper]
        }
    except Exception as e:
        print(f"Error retrieving paper: {e}")
        return {"search_results": search_results, "papers": []}

# Run the research phase
research_data = asyncio.run(research_phase("attention mechanisms in neural networks"))
```

### 2. Generation Phase

Next, generate conversations based on the research:

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.conversation_generator import OllamaConversationGenerator

async def generation_phase(research_data):
    # Set up shared components
    ollama = OllamaInterface(model_name="llama3")
    generator = OllamaConversationGenerator(model_name="llama3", ollama_interface=ollama)
    
    all_conversations = []
    
    # Generate conversations from each paper
    for paper_content in research_data["papers"]:
        # Chunk the paper content
        chunks = generator.chunk_text(paper_content, chunk_size=2000, overlap=200)
        
        # Generate a conversation for each chunk
        for i, chunk in enumerate(chunks[:3]):  # Process first 3 chunks
            conversation = generator.generate_conversation(
                content=chunk,
                num_turns=3,
                conversation_context="research paper",
                hedging_level="balanced"
            )
            if conversation:
                all_conversations.append(conversation)
    
    return {
        "original_conversations": all_conversations
    }

# Run the generation phase
generation_results = asyncio.run(generation_phase(research_data))
```

### 3. Augmentation Phase

Now, expand and augment the generated conversations:

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.dataset_expander import DatasetExpander

async def augmentation_phase(generation_results):
    # Set up components
    ollama = OllamaInterface(model_name="llama3")
    expander = DatasetExpander(ollama_interface=ollama, output_dir="./custom_workflow_output")
    
    original_conversations = generation_results["original_conversations"]
    
    if original_conversations:
        # Expand the dataset with variations
        expanded_conversations = expander.expand_conversation_dataset(
            conversations=original_conversations,
            expansion_factor=3,
            static_fields={'human': True, 'gpt': False}  # Keep human questions static
        )
        
        # Save the expanded dataset
        output_path = expander.save_conversations_to_parquet(
            expanded_conversations, 
            "augmented_dataset"
        )
        
        # Analyze the expansion results
        expansion_analysis = expander.analyze_expanded_dataset(
            original_conversations, 
            expanded_conversations
        )
        
        return {
            "original_conversations": original_conversations,
            "expanded_conversations": expanded_conversations,
            "expansion_analysis": expansion_analysis,
            "output_path": output_path
        }
    else:
        return {"error": "No conversations to augment"}

# Run the augmentation phase
augmentation_results = asyncio.run(augmentation_phase(generation_results))
```

### 4. Analysis Phase

Analyze the augmented dataset to identify quality issues and patterns:

```python
import pandas as pd
from agentChef.ollama_interface import OllamaInterface
from agentChef.pandas_query import OllamaLlamaIndexIntegration

def analysis_phase(augmentation_results):
    # Load the dataset
    try:
        df = pd.read_parquet(augmentation_results["output_path"])
        print(f"Loaded dataset with {len(df)} records")
        
        # Set up analysis tools
        ollama = OllamaInterface(model_name="llama3")
        analyzer = OllamaLlamaIndexIntegration(ollama_model="llama3")
        
        # Define analysis queries
        analysis_queries = [
            "What's the distribution of conversation lengths in the dataset?",
            "What are the most common topics discussed in these conversations?",
            "Are there any quality issues or inconsistencies in the dataset?"
        ]
        
        # Run each analysis query
        analysis_results = {}
        for query in analysis_queries:
            result = analyzer.query_dataframe_with_ollama(df, query)
            analysis_results[query] = result["response"]
            print(f"\nQuery: {query}")
            print(f"Response: {result['response']}")
        
        return analysis_results
    except Exception as e:
        print(f"Error in analysis phase: {e}")
        return {"error": str(e)}

# Run the analysis phase
analysis_results = analysis_phase(augmentation_results)
```

### 5. Cleaning Phase

Finally, clean the dataset based on analysis findings:

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.dataset_cleaner import DatasetCleaner

async def cleaning_phase(augmentation_results, analysis_results):
    # Set up components
    ollama = OllamaInterface(model_name="llama3")
    cleaner = DatasetCleaner(ollama_interface=ollama, output_dir="./custom_workflow_output/cleaned")
    
    original_conversations = augmentation_results["original_conversations"]
    expanded_conversations = augmentation_results["expanded_conversations"]
    
    # Clean the dataset
    cleaned_conversations = await cleaner.clean_dataset(
        original_conversations=original_conversations,
        expanded_conversations=expanded_conversations,
        cleaning_criteria={
            "fix_hallucinations": True,
            "normalize_style": True,
            "correct_grammar": True,
            "ensure_coherence": True
        }
    )
    
    # Save the cleaned conversations
    output_base = "cleaned_dataset"
    cleaned_output_path = f"./custom_workflow_output/cleaned/{output_base}.jsonl"
    
    with open(cleaned_output_path, 'w', encoding='utf-8') as f:
        for conversation in cleaned_conversations:
            f.write(json.dumps(conversation) + '\n')
            
    print(f"Saved {len(cleaned_conversations)} cleaned conversations to {cleaned_output_path}")
    
    return {
        "cleaned_conversations": cleaned_conversations,
        "output_path": cleaned_output_path,
        "cleaning_stats": {
            "original_count": len(original_conversations),
            "expanded_count": len(expanded_conversations),
            "cleaned_count": len(cleaned_conversations)
        }
    }

# Import json for saving
import json

# Run the cleaning phase
cleaning_results = asyncio.run(cleaning_phase(augmentation_results, analysis_results))
```

### 6. Putting It All Together

Combine these five phases into a complete UDRAGS-style workflow:

```python
import asyncio

async def custom_udrags_workflow(topic):
    print(f"Starting custom UDRAGS workflow for topic: {topic}")
    
    # Phase 1: Research
    print("\n=== RESEARCH PHASE ===")
    research_data = await research_phase(topic)
    
    # Phase 2: Generation
    print("\n=== GENERATION PHASE ===")
    generation_results = await generation_phase(research_data)
    
    # Phase 3: Augmentation
    print("\n=== AUGMENTATION PHASE ===")
    augmentation_results = await augmentation_phase(generation_results)
    
    # Phase 4: Analysis
    print("\n=== ANALYSIS PHASE ===")
    analysis_results = analysis_phase(augmentation_results)
    
    # Phase 5: Cleaning
    print("\n=== CLEANING PHASE ===")
    cleaning_results = await cleaning_phase(augmentation_results, analysis_results)
    
    return {
        "research": research_data,
        "generation": generation_results,
        "augmentation": augmentation_results,
        "analysis": analysis_results,
        "cleaning": cleaning_results
    }

# Run the complete workflow
workflow_results = asyncio.run(custom_udrags_workflow("attention mechanisms in neural networks"))
```

This custom workflow demonstrates how you can combine AgentChef's components to create a specialized pipeline following the UDRAGS approach: research-generate-augment-analyze-clean. Each phase builds on the previous one, creating a comprehensive system for dataset generation and processing that mirrors the core functionality of the built-in UDRAGS system.

You can adapt this pattern to create workflows for your specific needs, focusing on any part of the pipeline or extending it with additional processing steps.

## Package Structure

```md
agentChef/
├── pyproject.toml
├── setup.py
├── LICENSE
├── README.md
├── agentChef/
│   ├── __init__.py
│   ├── udrags.py
│   ├── conversation_generator.py
│   ├── dataset_expander.py
│   ├── dataset_cleaner.py
│   ├── pandas_query_integration.py 
│   ├── crawlers_module.py
│   ├── ui_module.py
│   └── assets/
│       ├── Untitled-removebg-preview.png
│       ├── buy me a coffee button.png
│       └── Discord button.png
└── tests/
    ├── __init__.py
    ├── test_conversation_generator.py
    └── test_dataset_expander.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0

## Acknowledgments

Agent Chef builds upon the UDRAGS framework and integrates with several open-source projects:
- Ollama for local LLM access
- LlamaIndex for natural language querying of structured data
- PyQt6 for the graphical user interface
