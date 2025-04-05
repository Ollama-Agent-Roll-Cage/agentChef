# DatasetExpander Documentation

## Class: `DatasetExpander`

A class to expand datasets by generating paraphrases and variations of conversation data, with control over which fields remain static and which are dynamically generated. Works with conversation data in the format produced by OllamaConversationGenerator.

### Constructor

```python
def __init__(self, ollama_interface, output_dir="./output", use_llama_index=True, openai_api_key=None)
```

Initialize the DatasetExpander.

**Parameters:**
- `ollama_interface`: An interface to Ollama for generating text
- `output_dir` (str): Directory to save expanded datasets
- `use_llama_index` (bool): Whether to use LlamaIndex for advanced DataFrame analysis
- `openai_api_key` (str, optional): OpenAI API key for LlamaIndex integration

### Methods

#### `expand_conversation_dataset`

```python
def expand_conversation_dataset(self, conversations, expansion_factor=3, 
                              static_fields=None, reference_fields=None)
```

Expand a dataset of conversations by generating paraphrases.

**Parameters:**
- `conversations`: List of conversations, where each conversation is a list of turns. Each turn is a dict with 'from' and 'value' keys
- `expansion_factor`: Number of variations to generate for each conversation
- `static_fields`: Dict mapping field names ("human", "gpt") to boolean indicating if they should remain static. If None, defaults to {'human': False, 'gpt': False} (all fields are dynamic)
- `reference_fields`: List of fields to use as reference values when generating paraphrases

**Returns:**
- List of expanded conversations

#### `paraphrase_text`

```python
def paraphrase_text(self, text, reference_values=None, is_question=None)
```

Generate a paraphrase of the given text.

**Parameters:**
- `text`: Text to paraphrase
- `reference_values`: Dictionary of reference values to incorporate
- `is_question`: Whether the text is a question (if None, will be detected automatically)

**Returns:**
- Paraphrased text

#### `verify_paraphrase`

```python
def verify_paraphrase(self, original, paraphrased, reference, is_question)
```

Verify that the paraphrased text maintains the meaning of the original.

**Parameters:**
- `original`: Original text
- `paraphrased`: Paraphrased text
- `reference`: Reference values
- `is_question`: Whether the text is a question

**Returns:**
- Verified or corrected paraphrased text

#### `clean_generated_content`

```python
def clean_generated_content(self, text, is_question)
```

Clean generated content by removing explanatory phrases, normalizing punctuation, etc.

**Parameters:**
- `text`: Text to clean
- `is_question`: Whether the text is a question

**Returns:**
- Cleaned text

#### `generate_conversations_from_paper`

```python
def generate_conversations_from_paper(self, paper_content, conversation_generator,
                                     num_chunks=5, num_turns=3, expansion_factor=2,
                                     static_fields=None, reference_fields=None)
```

Generate conversations from a paper and then expand the dataset.

**Parameters:**
- `paper_content`: The content of the research paper
- `conversation_generator`: An instance of OllamaConversationGenerator
- `num_chunks`: Number of chunks to create from the paper
- `num_turns`: Number of turns per conversation
- `expansion_factor`: Number of variations to create per conversation
- `static_fields`: Dict mapping field names to boolean indicating if they should remain static
- `reference_fields`: List of fields to use as reference when generating paraphrases

**Returns:**
- Tuple of (original conversations, expanded conversations)

#### `save_conversations_to_jsonl`

```python
def save_conversations_to_jsonl(self, conversations, filename)
```

Save conversations to a JSONL file.

**Parameters:**
- `conversations`: List of conversations to save
- `filename`: Name of the output file (without extension)

**Returns:**
- Path to the saved file

#### `save_conversations_to_parquet`

```python
def save_conversations_to_parquet(self, conversations, filename)
```

Save conversations to a Parquet file.

**Parameters:**
- `conversations`: List of conversations to save
- `filename`: Name of the output file (without extension)

**Returns:**
- Path to the saved file

#### `load_conversations_from_jsonl`

```python
def load_conversations_from_jsonl(self, file_path)
```

Load conversations from a JSONL file.

**Parameters:**
- `file_path`: Path to the JSONL file

**Returns:**
- List of conversations

#### `convert_conversations_to_dataframe`

```python
def convert_conversations_to_dataframe(self, conversations)
```

Convert conversations to a DataFrame format for analysis.

**Parameters:**
- `conversations`: List of conversations

**Returns:**
- DataFrame with structured conversation data

#### `convert_to_multi_format`

```python
def convert_to_multi_format(self, conversations, base_filename, 
                           formats=['jsonl', 'parquet', 'csv', 'df'])
```

Convert conversations to multiple formats and save them.

**Parameters:**
- `conversations`: List of conversations
- `base_filename`: Base name for the output files
- `formats`: List of output formats to generate

**Returns:**
- Dictionary mapping format names to output paths

#### `analyze_expanded_dataset`

```python
def analyze_expanded_dataset(self, original_conversations, expanded_conversations)
```

Analyze the expanded dataset in comparison to the original using natural language queries.

**Parameters:**
- `original_conversations`: List of original conversations
- `expanded_conversations`: List of expanded conversations

**Returns:**
- Dictionary with analysis results

### Private Methods

#### `_perform_advanced_analysis`

```python
def _perform_advanced_analysis(self, orig_df, expanded_df)
```

Perform advanced analysis using PandasQueryIntegration.

**Parameters:**
- `orig_df`: DataFrame with original conversations
- `expanded_df`: DataFrame with expanded conversations

**Returns:**
- Dictionary with advanced analysis results

#### `_is_question`

```python
def _is_question(self, text)
```

Determine if the text is a question.

**Parameters:**
- `text`: Text to analyze

**Returns:**
- True if the text is a question, False otherwise
