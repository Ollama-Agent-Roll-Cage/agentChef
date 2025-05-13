# RagChef Usage Guide

## Command-Line Interface

### Research Mode
```bash
python -m agentChef.ragchef --mode research \
    --topic "Your research topic" \
    --max-papers 5 \
    --max-search 10 \
    --include-github \
    --github-repos "repo1_url" "repo2_url"
```

### Generate Mode
```bash
python -m agentChef.ragchef --mode generate \
    --topic "Your topic" \
    --turns 3 \
    --expand 3 \
    --clean \
    --format all \
    --hedging balanced
```

### Process Mode
```bash
python -m agentChef.ragchef --mode process \
    --input /path/to/papers/ \
    --turns 3 \
    --expand 3 \
    --clean \
    --format all
```

### Analyze Mode
```bash
python -m agentChef.ragchef --mode analyze \
    --orig-dataset original.jsonl \
    --exp-dataset expanded.jsonl \
    --basic-stats \
    --quality \
    --comparison
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
