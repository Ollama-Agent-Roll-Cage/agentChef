# AgentChef Usage Guide

## Core Components

### RAGChef

The Research Augmentation Generation Chef (RAGChef) is the main component for:
- Research content collection
- Dataset generation
- Content augmentation

```python
from agentChef.core.chefs.ragchef import ResearchManager

# Initialize
manager = ResearchManager()

# Research a topic
result = await manager.research_topic("quantum computing")

# Generate dataset
dataset = await manager.generate_conversation_dataset(
    num_turns=3,
    expansion_factor=2
)
```

### Dataset Management

```python
from agentChef.core.generation.conversation_generator import OllamaConversationGenerator
from agentChef.core.augmentation.dataset_expander import DatasetExpander
from agentChef.core.classification.dataset_cleaner import DatasetCleaner

# Generate conversations
generator = OllamaConversationGenerator()
conversations = generator.generate_conversation(content)

# Expand dataset
expander = DatasetExpander(ollama_interface)
expanded = expander.expand_conversation_dataset(conversations)

# Clean dataset
cleaner = DatasetCleaner(ollama_interface)
cleaned = cleaner.clean_dataset(expanded)
```

## Interfaces

### FastAPI

```bash
uvicorn agentChef.api.agent_chef_api:app --reload
```

### CLI

```bash
# Research
agentchef research --topic "topic"

# Generate
agentchef generate --input papers/ --turns 3

# Clean
agentchef clean --input dataset.jsonl
```

### MCP Server

```bash
agentchef mcp start --port 50505
```

## Command-Line Interface

### Research Mode
```bash
# Basic research
python -m agentChef.ragchef --mode research \
    --topic "Your research topic" \
    --max-papers 5 \
    --max-search 10 \
    --include-github \
    --github-repos "repo1_url" "repo2_url"

# Query datasets
python -m agentChef.ragchef research query \
    --query "Find all conversations about transformers" \
    --dataset data/conversations.parquet \
    --output results.json

# Classify content
python -m agentChef.ragchef research classify \
    --text "Content to analyze" \
    --categories harm bias quality
```

### Generation Mode
```bash
python -m agentChef.ragchef --mode generate \
    --topic "Your topic" \
    --turns 3 \
    --expand 3 \
    --clean \
    --format all \
    --hedging balanced \
    --model llama3
```

### Process Mode
```bash
python -m agentChef.ragchef --mode process \
    --input /path/to/papers/ \
    --turns 3 \
    --expand 3 \
    --clean \
    --format all \
    --output-dir ./output
```

### Analysis Mode
```bash
python -m agentChef.ragchef --mode analyze \
    --orig-dataset original.jsonl \
    --exp-dataset expanded.jsonl \
    --basic-stats \
    --quality \
    --comparison \
    --output analysis_report.json
```

### Storage Operations
```bash
# Save to Parquet
python -m agentChef.ragchef storage save \
    --data input.json \
    --format parquet \
    --output data.parquet

# Query stored data
python -m agentChef.ragchef storage query \
    --query "Find all..." \
    --file data.parquet
```

### Model Management
```bash
# List available models
python -m agentChef.ragchef models list

# Set active model
python -m agentChef.ragchef models set \
    --model llama3
```

## Graphical Interface

1. Launch the UI:
```bash
python -m agentChef.ragchef --mode ui
```

2. Research Tab:
   - Enter research topic
   - Set max papers and search results
   - Enable GitHub integration if needed
   - Click "Start Research"
   - View progress and results
   - Save results or proceed to generation

3. Generate Tab:
   - Set conversation parameters (turns, expansion)
   - Choose hedging level
   - Select output format
   - Click "Generate Dataset"
   - Monitor progress
   - Open output directory or analyze results

4. Process Tab:
   - Select input papers directory
   - Configure generation settings
   - Choose output format
   - Click "Process Files"
   - View processing results

5. Analyze Tab:
   - Select original and expanded datasets
   - Choose analysis options
   - Click "Analyze Datasets"
   - View analysis results
   - Save analysis report

## Common Options

- `--model`: Choose Ollama model (default: "llama3")
- `--output-dir`: Set custom output directory
- `--format`: Choose output format (jsonl/parquet/csv/all)
- `--verbose`: Enable detailed logging
- `--config`: Use custom configuration file

## Examples

1. Basic Research:
```bash
python -m agentChef.ragchef --mode research --topic "transformer models"
```

2. Generate Dataset with Research:
```bash
python -m agentChef.ragchef --mode generate \
    --topic "transformer models" \
    --turns 3 \
    --expand 3 \
    --clean \
    --format all
```

3. Process Papers:
```bash
python -m agentChef.ragchef --mode process \
    --input papers/ \
    --turns 3 \
    --expand 3 \
    --clean \
    --format jsonl
```

4. Dataset Analysis:
```bash
python -m agentChef.ragchef --mode analyze \
    --orig-dataset original.jsonl \
    --exp-dataset expanded.jsonl \
    --basic-stats \
    --quality \
    --comparison
```

## Requirements

1. Required packages:
   - ollama
   - PyQt6 (for UI mode)
   - pandas
   - numpy
   - tqdm

2. Install dependencies:
```bash
pip install ollama PyQt6 pandas numpy tqdm
```

3. Make sure Ollama is running:
```bash
ollama serve
```

## Troubleshooting

1. If UI doesn't launch:
   - Check PyQt6 installation
   - Try command-line mode instead

2. If research fails:
   - Verify Ollama is running
   - Check internet connection
   - Try with fewer papers/results

3. If generation is slow:
   - Reduce number of turns/expansion
   - Use faster Ollama model
   - Process fewer papers at once
