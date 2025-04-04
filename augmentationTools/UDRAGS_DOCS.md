# UDRAGS - Unified Dataset Research, Augmentation, & Generation System

The UDRAGS package is a comprehensive suite of tools for researching, generating, expanding, and cleaning conversation datasets. All powered by local Ollama models, making it accessible and efficient without requiring external API access.

## Key Features

- **Research Pipeline**: Extract and process content from ArXiv papers, web searches, and GitHub repositories
- **Conversation Generation**: Create realistic conversations from research papers and other content
- **Dataset Expansion**: Paraphrase and expand datasets with controlled variations
- **NLP Hedging**: Generate nuanced responses with appropriate levels of confidence and hedging
- **Data Analysis**: Query and analyze datasets using natural language with Ollama-powered pandas integration
- **Dataset Cleaning**: Identify and fix quality issues in expanded conversation datasets

## Installation

```bash
pip install udrags
```

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and configured with your models of choice
- `pandas` and `numpy` for data handling

## Quick Start

```python
from udrags.conversation_generator import OllamaConversationGenerator
from udrags.dataset_expander import DatasetExpander
from udrags.pandas_query_integration import OllamaPandasQuery

# Initialize conversation generator
generator = OllamaConversationGenerator(model_name="llama3", enable_hedging=True)

# Generate a conversation about a topic
conversation = generator.generate_conversation(
    content="Attention mechanisms have become an integral part of compelling sequence modeling...",
    num_turns=3,
    conversation_context="AI research",
    hedging_level="balanced"
)

# Create an Ollama interface (simple wrapper for the ollama library)
class OllamaInterface:
    def __init__(self, model_name="llama3"):
        self.model = model_name
        
    def chat(self, messages):
        import ollama
        return ollama.chat(model=self.model, messages=messages)

# Initialize dataset expander
ollama_interface = OllamaInterface(model_name="llama3")
expander = DatasetExpander(ollama_interface, output_dir="./expanded_data")

# Expand the generated conversation
expanded_conversations = expander.expand_conversation_dataset(
    conversations=[conversation],
    expansion_factor=3,
    static_fields={'human': True, 'gpt': False}  # Keep human questions static
)

# Save the expanded conversations
expander.save_conversations_to_jsonl(expanded_conversations, "expanded_conversations")
```

## Core Components

### OllamaConversationGenerator

Generate realistic conversations from text content using Ollama LLMs.

```python
generator = OllamaConversationGenerator(model_name="llama3", enable_hedging=True)

# Create a conversation about research content
conversation = generator.generate_conversation(
    content=paper_abstract,
    num_turns=3,
    conversation_context="research paper"
)

# Generate a hedged response with controlled confidence
response = generator.generate_hedged_response(
    prompt="Explain transformers in simple terms",
    hedging_profile="balanced",
    knowledge_level="high",
    subject_expertise="machine learning"
)
```

### OllamaPandasQuery

Natural language querying of pandas DataFrames using Ollama models.

```python
import pandas as pd
from udrags.pandas_query_integration import OllamaPandasQuery

# Create the query engine
query_engine = OllamaPandasQuery(ollama_interface)

# Query your DataFrame with natural language
result = query_engine.query_dataframe(
    df, 
    "What's the average message length by participant type?"
)

# Generate dataset insights automatically
insights = query_engine.generate_dataset_insights(df, num_insights=5)

# Compare two datasets
comparison = query_engine.compare_datasets(
    orig_df, expanded_df, 
    df1_name="Original", df2_name="Expanded"
)
```

### DatasetExpander

Expand existing conversation datasets by generating paraphrases and variations.

```python
# Initialize the expander
expander = DatasetExpander(ollama_interface, output_dir="./expanded_data")

# Expand a conversation dataset
expanded_conversations = expander.expand_conversation_dataset(
    conversations=original_conversations,
    expansion_factor=3,
    static_fields={'human': False, 'gpt': False},  # Make both dynamic
    reference_fields=['human']  # Use human messages as reference context
)

# Save in multiple formats
output_files = expander.convert_to_multi_format(
    expanded_conversations, 
    "my_dataset",
    formats=['jsonl', 'parquet', 'csv']
)
```

### DatasetCleaner

Clean and validate expanded datasets by comparing to originals and fixing quality issues.

```python
from udrags.dataset_cleaner import DatasetCleaner

# Initialize the cleaner
cleaner = DatasetCleaner(ollama_interface, output_dir="./cleaned_data")

# Analyze dataset quality
analysis = cleaner.analyze_dataset(
    original_conversations=original_conversations,
    expanded_conversations=expanded_conversations
)

# Clean the expanded dataset
cleaned_conversations = cleaner.clean_dataset(
    original_conversations=original_conversations,
    expanded_conversations=expanded_conversations,
    cleaning_criteria={
        "fix_hallucinations": True,
        "normalize_style": True,
        "correct_grammar": True,
        "ensure_coherence": True
    }
)

# Save the analysis report
cleaner.save_cleaning_report(analysis, "cleaning_report.json")
```

## Command Line Interface

UDRAGS includes a comprehensive CLI for easy integration into workflows:

```bash
# Research a topic
python -m udrags --mode research --topic "transformer models" --max-papers 5

# Generate conversations from research
python -m udrags --mode generate --topic "transformer models" --turns 3 --expand 5 --clean

# Process existing papers
python -m udrags --mode process --input papers_dir/ --format jsonl --expand 3
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
