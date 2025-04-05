# UDRAGS Documentation

U.D.R.A.G.S. (Unified Dataset Research, Augmentation, & Generation System) replaces the previous research modules by leveraging functionality provided in the crawlers_module.py utilities. It provides a complete pipeline for researching topics, converting research papers to conversation format, expanding datasets, and cleaning the expanded datasets.

## Class: `ResearchManager`

Manages the research and dataset generation workflow.

### Constructor

```python
def __init__(self, data_dir=DEFAULT_DATA_DIR, model_name="llama3")
```

Initialize the research manager.

**Parameters:**
- `data_dir`: Directory to store research data and generated datasets
- `model_name`: Name of the Ollama model to use for generations

### Methods

#### `research_topic`

```python
async def research_topic(self, topic, max_papers=5, max_search_results=10, 
                        include_github=False, github_repos=None, callback=None)
```

Research a topic using ArXiv, web search, and optionally GitHub.

**Parameters:**
- `topic`: Research topic to investigate
- `max_papers`: Maximum number of papers to process
- `max_search_results`: Maximum number of web search results
- `include_github`: Whether to include GitHub repositories
- `github_repos`: Optional list of GitHub repositories to include
- `callback`: Optional callback function for progress updates

**Returns:**
- Dictionary with research results

#### `generate_conversation_dataset`

```python
async def generate_conversation_dataset(self, papers=None, num_turns=3, 
                                       expansion_factor=3, clean=True, callback=None)
```

Generate a conversation dataset from research papers.

**Parameters:**
- `papers`: List of papers to process (if None, uses the papers from research_state)
- `num_turns`: Number of conversation turns to generate
- `expansion_factor`: Factor by which to expand the dataset
- `clean`: Whether to clean the expanded dataset
- `callback`: Optional callback function for progress updates

**Returns:**
- Dictionary with generated dataset information

#### `process_paper_files`

```python
async def process_paper_files(self, paper_files, output_format='jsonl', 
                             num_turns=3, expansion_factor=3, clean=True, callback=None)
```

Process paper files to generate conversation datasets.

**Parameters:**
- `paper_files`: List of file paths to papers
- `output_format`: Output format ('jsonl', 'parquet', or 'csv')
- `num_turns`: Number of conversation turns to generate
- `expansion_factor`: Factor by which to expand the dataset
- `clean`: Whether to clean the expanded dataset
- `callback`: Optional callback function for progress updates

**Returns:**
- Dictionary with generated dataset information

#### `cleanup`

```python
def cleanup(self)
```

Clean up temporary files.

### Private Methods

#### `_generate_arxiv_queries`

```python
async def _generate_arxiv_queries(self, topic)
```

Generate specific queries for ArXiv based on the research topic.

**Parameters:**
- `topic`: Research topic

**Returns:**
- List of ArXiv queries

#### `_generate_research_summary`

```python
def _generate_research_summary(self)
```

Generate a summary of the research results.

**Returns:**
- Formatted markdown summary

## Class: `ResearchThread`

Thread for running research operations in the background (for UI mode).

### Constructor

```python
def __init__(self, manager, operation, **kwargs)
```

Initialize the research thread.

**Parameters:**
- `manager`: ResearchManager instance
- `operation`: Operation to perform ('research' or 'generate')
- `**kwargs`: Additional arguments for the operation

### Methods

#### `run`

```python
def run(self)
```

Run the research thread.

#### `stop`

```python
def stop(self)
```

Request the thread to stop.

## Command-line Interface

The UDRAGS module provides a comprehensive command-line interface with the following modes:

### Research Mode

```bash
python -m agentChef.udrags --mode research --topic "Your research topic" --max-papers 5 --max-search 10
```

### Generate Mode

```bash
python -m agentChef.udrags --mode generate --topic "Your topic" --turns 3 --expand 3 --clean --format jsonl
```

### Process Mode

```bash
python -m agentChef.udrags --mode process --input papers_dir/ --turns 3 --expand 3 --clean --format all
```

### UI Mode (if PyQt6 is installed)

```bash
python -m agentChef.udrags --mode ui
```
